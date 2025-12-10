#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <cuda_runtime.h>

namespace fs = std::filesystem;

struct Box {
    float x1, y1, x2, y2;
    float score;
    float class_id;
};

// Helper class for memory reuse
// -----------------------------
// This struct manages GPU memory to avoid repeated allocation/deallocation overhead.
// It keeps track of the current capacity and only reallocates if the new requirement exceeds it.
struct GpuMemory {
    Box* d_boxes = nullptr;
    unsigned char* d_mask = nullptr;
    size_t capacity_boxes = 0;
    size_t capacity_mask = 0;

    ~GpuMemory() {
        if (d_boxes) cudaFree(d_boxes);
        if (d_mask) cudaFree(d_mask);
    }

    // Ensures that enough memory is allocated for 'n' boxes
    void reserve(size_t n) {
        if (n > capacity_boxes) {
            if (d_boxes) cudaFree(d_boxes);
            cudaMalloc(&d_boxes, n * sizeof(Box));
            capacity_boxes = n;
        }
        size_t mask_size = n * n;
        if (mask_size > capacity_mask) {
            if (d_mask) cudaFree(d_mask);
            cudaMalloc(&d_mask, mask_size * sizeof(unsigned char));
            capacity_mask = mask_size;
        }
    }
};

// Device function to compute IoU
__device__ float devIoU(const Box& a, const Box& b) {
    float xx1 = fmaxf(a.x1, b.x1);
    float yy1 = fmaxf(a.y1, b.y1);
    float xx2 = fminf(a.x2, b.x2);
    float yy2 = fminf(a.y2, b.y2);

    float w = fmaxf(0.0f, xx2 - xx1);
    float h = fmaxf(0.0f, yy2 - yy1);
    float inter_area = w * h;

    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    float union_area = area_a + area_b - inter_area;

    if (union_area <= 0.0f) return 0.0f;
    return inter_area / union_area;
}

// Optimized NMS Kernel (Shared Memory Tiling)
// -------------------------------------------
// This kernel optimizes global memory access by loading blocks (tiles) of boxes into shared memory.
//
// Logic:
// 1. Each thread 'i' loads its own box into registers.
// 2. The kernel iterates over the list of boxes in tiles (chunks of blockDim.x).
// 3. For each tile, threads cooperatively load the tile's boxes into shared memory.
// 4. Thread 'i' compares its box against all boxes in the current shared memory tile.
// 5. This reduces global memory reads significantly, as each box in a tile is read from global memory
//    only once per block, and then accessed multiple times from fast shared memory.
__global__ void nms_kernel_opt(const Box* boxes, int n, float thresh, unsigned char* mask) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load my box into registers
    Box my_box;
    if (i < n) {
        my_box = boxes[i];
    }

    // Shared memory buffer for the current tile of boxes
    __shared__ Box tile_boxes[256]; // Assumes block size is 256

    // Loop over all tiles
    for (int tile = 0; tile < (n + blockDim.x - 1) / blockDim.x; ++tile) {
        int tile_idx = tile * blockDim.x + threadIdx.x;
        
        // Cooperative load: Each thread loads one box of the tile into shared memory
        if (tile_idx < n) {
            tile_boxes[threadIdx.x] = boxes[tile_idx];
        }
        __syncthreads(); // Wait for tile to be loaded

        if (i < n) {
            // Compare my_box against all boxes in the loaded tile
            // We only need to check j > i
            int tile_start = tile * blockDim.x;
            int tile_end = min(tile_start + blockDim.x, n);

            // Optimization: Skip tile if all boxes in it are <= i
            if (tile_end > i + 1) {
                for (int k = 0; k < blockDim.x; ++k) {
                    int j = tile_start + k;
                    if (j >= n) break;
                    
                    if (j > i) {
                        Box other_box = tile_boxes[k]; // Access from shared memory
                        if (my_box.class_id == other_box.class_id) {
                            if (devIoU(my_box, other_box) > thresh) {
                                mask[i * n + j] = 1;
                            }
                        }
                    }
                }
            }
        }
        __syncthreads(); // Wait before loading the next tile
    }
}

std::vector<Box> load_boxes(const std::string& filepath) {
    // ... (implementation remains same)
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) return {};

    int num_boxes;
    file.read(reinterpret_cast<char*>(&num_boxes), sizeof(int));

    std::vector<Box> boxes(num_boxes);
    std::vector<float> buffer(num_boxes * 6);
    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size() * sizeof(float));

    for(int i=0; i<num_boxes; ++i) {
        boxes[i].x1       = buffer[i*6 + 0];
        boxes[i].y1       = buffer[i*6 + 1];
        boxes[i].x2       = buffer[i*6 + 2];
        boxes[i].y2       = buffer[i*6 + 3];
        boxes[i].score    = buffer[i*6 + 4];
        boxes[i].class_id = buffer[i*6 + 5];
    }
    return boxes;
}

// Main NMS Function (Optimized CUDA)
// ----------------------------------
// Uses GpuMemory for memory reuse and calls the optimized tiled kernel.
std::vector<Box> nms_cuda_opt(std::vector<Box>& boxes, float iou_threshold, GpuMemory& mem) {
    if (boxes.empty()) return {};

    // 1. Sort on CPU
    std::sort(boxes.begin(), boxes.end(), [](const Box& a, const Box& b) {
        return a.score > b.score;
    });

    int n = boxes.size();
    
    // 2. Prepare GPU memory (reuse if possible)
    mem.reserve(n);

    // 3. Initialize mask and copy boxes
    cudaMemset(mem.d_mask, 0, (size_t)n * n * sizeof(unsigned char)); 
    cudaMemcpy(mem.d_boxes, boxes.data(), n * sizeof(Box), cudaMemcpyHostToDevice);

    // 4. Launch Optimized Kernel
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    nms_kernel_opt<<<numBlocks, blockSize>>>(mem.d_boxes, n, iou_threshold, mem.d_mask);
    cudaDeviceSynchronize();

    // 5. Copy mask back
    std::vector<unsigned char> h_mask(n * n);
    cudaMemcpy(h_mask.data(), mem.d_mask, (size_t)n * n * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // 6. Suppress on CPU
    std::vector<Box> kept_boxes;
    std::vector<bool> suppressed(n, false);

    for (int i = 0; i < n; ++i) {
        if (suppressed[i]) continue;
        kept_boxes.push_back(boxes[i]);
        
        for (int j = i + 1; j < n; ++j) {
            if (suppressed[j]) continue;
            if (h_mask[i * n + j]) {
                suppressed[j] = true;
            }
        }
    }

    return kept_boxes;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <path to data folder>" << std::endl;
        return 1;
    }

    std::string input_folder = argv[1];
    float iou_thresh = 0.5f;    

    if (!fs::exists(input_folder)) {
        std::cerr << "Folder not found!" << std::endl;
        return 1;
    }

    std::vector<std::string> file_paths;
    for (const auto& entry : fs::directory_iterator(input_folder)) {
        if (entry.path().extension() == ".bin") {
            file_paths.push_back(entry.path().string());
        }
    }

    long long total_boxes_before = 0;
    long long total_boxes_after = 0;

    // Initialize GPU Memory Manager
    GpuMemory mem;

    auto start_total = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < file_paths.size(); ++i) {
        std::vector<Box> boxes = load_boxes(file_paths[i]);
        std::vector<Box> result = nms_cuda_opt(boxes, iou_thresh, mem);
        total_boxes_before += boxes.size();
        total_boxes_after += result.size();
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double, std::milli>(end_total - start_total).count();

    std::cout << "=== CUDA Optimized Results ===" << std::endl;
    std::cout << "Processed " << file_paths.size() << " images." << std::endl;
    std::cout << "Total Boxes Processed: " << total_boxes_before << std::endl;
    std::cout << "Total Boxes Kept:      " << total_boxes_after << std::endl;
    std::cout << "Total Time:             " << total_time << " ms" << std::endl;

    return 0;
}
