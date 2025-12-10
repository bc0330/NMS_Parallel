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

// Device function to compute Intersection over Union (IoU) between two boxes
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

// Naive NMS Kernel
// ----------------
// This kernel computes the IoU matrix for all pairs of boxes.
// Each thread 'i' is responsible for checking if box 'i' suppresses any box 'j' where j > i.
//
// Arguments:
// - boxes: Array of sorted bounding boxes (on device)
// - n: Number of boxes
// - thresh: IoU threshold for suppression
// - mask: Output collision mask (flattened 2D array of size n*n)
//         mask[i * n + j] = 1 means box 'i' suppresses box 'j'
__global__ void nms_kernel_naive(const Box* boxes, int n, float thresh, unsigned char* mask) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Box my_box = boxes[i];

    // Check against all subsequent boxes
    for (int j = i + 1; j < n; ++j) {
        Box other_box = boxes[j];
        if (my_box.class_id == other_box.class_id) {
            if (devIoU(my_box, other_box) > thresh) {
                mask[i * n + j] = 1; // Mark j as suppressed by i
            } else {
                mask[i * n + j] = 0;
            }
        } else {
            mask[i * n + j] = 0;
        }
    }
}

std::vector<Box> load_boxes(const std::string& filepath) {
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

// Main NMS Function (Naive CUDA)
// ------------------------------
// 1. Sorts boxes by score on CPU.
// 2. Copies boxes to GPU.
// 3. Launches kernel to compute IoU matrix.
// 4. Copies mask back to CPU.
// 5. Performs suppression sequentially on CPU using the mask.
std::vector<Box> nms_cuda_naive(std::vector<Box>& boxes, float iou_threshold) {
    if (boxes.empty()) return {};

    // 1. Sort on CPU (Descending Score)
    std::sort(boxes.begin(), boxes.end(), [](const Box& a, const Box& b) {
        return a.score > b.score;
    });

    int n = boxes.size();
    Box* d_boxes;
    unsigned char* d_mask;
    
    // Allocate device memory
    cudaMalloc(&d_boxes, n * sizeof(Box));
    cudaMalloc(&d_mask, (size_t)n * n * sizeof(unsigned char));

    // Copy data to device
    cudaMemcpy(d_boxes, boxes.data(), n * sizeof(Box), cudaMemcpyHostToDevice);

    // Launch Kernel
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    nms_kernel_naive<<<numBlocks, blockSize>>>(d_boxes, n, iou_threshold, d_mask);
    cudaDeviceSynchronize();

    // Copy mask back to host
    std::vector<unsigned char> h_mask(n * n);
    cudaMemcpy(h_mask.data(), d_mask, (size_t)n * n * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Cleanup device memory
    cudaFree(d_boxes);
    cudaFree(d_mask);

    // 2. Suppress on CPU using the computed mask
    std::vector<Box> kept_boxes;
    std::vector<bool> suppressed(n, false);

    for (int i = 0; i < n; ++i) {
        if (suppressed[i]) continue;
        kept_boxes.push_back(boxes[i]);
        
        // Mark all boxes suppressed by 'i'
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

    auto start_total = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < file_paths.size(); ++i) {
        std::vector<Box> boxes = load_boxes(file_paths[i]);
        std::vector<Box> result = nms_cuda_naive(boxes, iou_thresh);
        total_boxes_before += boxes.size();
        total_boxes_after += result.size();
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double, std::milli>(end_total - start_total).count();

    std::cout << "=== CUDA Naive Results ===" << std::endl;
    std::cout << "Processed " << file_paths.size() << " images." << std::endl;
    std::cout << "Total Boxes Processed: " << total_boxes_before << std::endl;
    std::cout << "Total Boxes Kept:      " << total_boxes_after << std::endl;
    std::cout << "Total Time:             " << total_time << " ms" << std::endl;

    return 0;
}
