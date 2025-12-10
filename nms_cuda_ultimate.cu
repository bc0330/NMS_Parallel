#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <immintrin.h> // For AVX2

namespace fs = std::filesystem;

struct Box {
    float x1, y1, x2, y2;
    float score;
    float class_id;
};

// ==========================================
// CPU SIMD Implementation (Fallback)
// ==========================================

// ==========================================
// CPU SIMD Implementation (Fallback)
// ----------------------------------
// This struct converts Array of Structures (AoS) to Structure of Arrays (SoA)
// to optimize SIMD operations by ensuring contiguous memory access for each field.
struct SoABoxes {
    std::vector<float> x1, y1, x2, y2, area, scores, class_ids;
    size_t size;

    void reserve(size_t n) {
        x1.reserve(n); y1.reserve(n); x2.reserve(n); y2.reserve(n);
        area.reserve(n); scores.reserve(n); class_ids.reserve(n);
    }

    void push_back(const Box& b) {
        x1.push_back(b.x1);
        y1.push_back(b.y1);
        x2.push_back(b.x2);
        y2.push_back(b.y2);
        area.push_back((b.x2 - b.x1) * (b.y2 - b.y1));
        scores.push_back(b.score);
        class_ids.push_back(b.class_id);
        size++;
    }
};

// SIMD NMS on CPU
// ---------------
// Uses AVX2 intrinsics to process 8 boxes at a time.
// This is used as a fallback when the number of boxes is too small to justify GPU overhead.
std::vector<Box> nms_simd_cpu(std::vector<Box>& boxes, float iou_threshold) {
    if (boxes.empty()) return {};

    std::sort(boxes.begin(), boxes.end(), [](const Box& a, const Box& b) {
        return a.score > b.score;
    });

    SoABoxes soa;
    soa.size = 0;
    soa.reserve(boxes.size());
    for (const auto& box : boxes) {
        soa.push_back(box);
    }

    std::vector<Box> kept_boxes;
    std::vector<bool> suppressed(boxes.size(), false);
    int n = boxes.size();

    __m256 t_threshold = _mm256_set1_ps(iou_threshold);
    __m256 zero = _mm256_setzero_ps();

    for (int i = 0; i < n; ++i) {
        if (suppressed[i]) continue;
        kept_boxes.push_back(boxes[i]);

        // Broadcast current box parameters
        float ix1 = soa.x1[i];
        float iy1 = soa.y1[i];
        float ix2 = soa.x2[i];
        float iy2 = soa.y2[i];
        float iarea = soa.area[i];
        float iclass = soa.class_ids[i];

        __m256 v_ix1 = _mm256_set1_ps(ix1);
        __m256 v_iy1 = _mm256_set1_ps(iy1);
        __m256 v_ix2 = _mm256_set1_ps(ix2);
        __m256 v_iy2 = _mm256_set1_ps(iy2);
        __m256 v_iarea = _mm256_set1_ps(iarea);
        __m256 v_iclass = _mm256_set1_ps(iclass);

        // Check against subsequent boxes in chunks of 8
        int j = i + 1;
        for (; j <= n - 8; j += 8) {
            __m256 v_jx1 = _mm256_loadu_ps(&soa.x1[j]);
            __m256 v_jy1 = _mm256_loadu_ps(&soa.y1[j]);
            __m256 v_jx2 = _mm256_loadu_ps(&soa.x2[j]);
            __m256 v_jy2 = _mm256_loadu_ps(&soa.y2[j]);
            __m256 v_jarea = _mm256_loadu_ps(&soa.area[j]);
            __m256 v_jclass = _mm256_loadu_ps(&soa.class_ids[j]);

            // Class check
            __m256 class_mask = _mm256_cmp_ps(v_iclass, v_jclass, _CMP_EQ_OQ);
            int mask_int = _mm256_movemask_ps(class_mask);

            if (mask_int == 0) continue;

            // IoU Calculation
            __m256 xx1 = _mm256_max_ps(v_ix1, v_jx1);
            __m256 yy1 = _mm256_max_ps(v_iy1, v_jy1);
            __m256 xx2 = _mm256_min_ps(v_ix2, v_jx2);
            __m256 yy2 = _mm256_min_ps(v_iy2, v_jy2);

            __m256 w = _mm256_max_ps(zero, _mm256_sub_ps(xx2, xx1));
            __m256 h = _mm256_max_ps(zero, _mm256_sub_ps(yy2, yy1));
            __m256 inter = _mm256_mul_ps(w, h);

            __m256 union_area = _mm256_sub_ps(_mm256_add_ps(v_iarea, v_jarea), inter);
            __m256 iou = _mm256_div_ps(inter, union_area);

            __m256 iou_mask = _mm256_cmp_ps(iou, t_threshold, _CMP_GT_OQ);
            __m256 final_mask = _mm256_and_ps(class_mask, iou_mask);

            int final_mask_int = _mm256_movemask_ps(final_mask);

            if (final_mask_int) {
                for (int k = 0; k < 8; ++k) {
                    if ((final_mask_int >> k) & 1) {
                         if (!suppressed[j + k]) suppressed[j + k] = true;
                    }
                }
            }
        }

        // Handle remaining items sequentially
        for (; j < n; ++j) {
            if (suppressed[j]) continue;
            if (boxes[i].class_id == boxes[j].class_id) {
                float xx1 = std::max(ix1, boxes[j].x1);
                float yy1 = std::max(iy1, boxes[j].y1);
                float xx2 = std::min(ix2, boxes[j].x2);
                float yy2 = std::min(iy2, boxes[j].y2);

                float w = std::max(0.0f, xx2 - xx1);
                float h = std::max(0.0f, yy2 - yy1);
                float inter = w * h;
                float union_area = iarea + soa.area[j] - inter;

                if (inter / union_area > iou_threshold) {
                    suppressed[j] = true;
                }
            }
        }
    }
    return kept_boxes;
}


// ==========================================
// GPU Implementation
// ==========================================

// Functor for sorting boxes by score descending (used by Thrust)
struct BoxComparator {
    __host__ __device__
    bool operator()(const Box& a, const Box& b) {
        return a.score > b.score;
    }
};

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

// Ultimate NMS Kernel
// -------------------
// Uses bitmasks to store the suppression results compactly.
// Each thread 'i' checks its box against all other boxes 'j' in chunks of 64.
// The result is stored as a 64-bit integer (unsigned long long), where each bit represents a suppression flag.
//
// Output: mask[i * num_chunks + chunk]
//         If bit 'k' in chunk 'c' is set, it means box 'i' suppresses box 'c*64 + k'.
__global__ void nms_kernel_ultimate(const Box* boxes, int n, float thresh, unsigned long long* mask) {
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row index (my box)
    if (i >= n) return;

    Box my_box = boxes[i];
    int num_chunks = (n + 63) / 64;

    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        unsigned long long my_mask = 0;
        int start_j = chunk * 64;
        
        // Optimization: Only check chunks that might contain j > i
        if (start_j > i) { 
             for (int bit = 0; bit < 64; ++bit) {
                int j = start_j + bit;
                if (j >= n) break;
                
                if (my_box.class_id == boxes[j].class_id) {
                    if (devIoU(my_box, boxes[j]) > thresh) {
                        my_mask |= (1ULL << bit);
                    }
                }
            }
        } else if (start_j + 64 > i) { // Chunk overlaps with i
            for (int bit = 0; bit < 64; ++bit) {
                int j = start_j + bit;
                if (j >= n) break;
                if (j > i) {
                    if (my_box.class_id == boxes[j].class_id) {
                        if (devIoU(my_box, boxes[j]) > thresh) {
                            my_mask |= (1ULL << bit);
                        }
                    }
                }
            }
        }
        mask[i * num_chunks + chunk] = my_mask;
    }
}

// Helper class for memory reuse (Thrust)
// --------------------------------------
// Wraps thrust::device_vector to keep memory allocated on the GPU across function calls.
struct GpuMemory {
    thrust::device_vector<Box> d_boxes;
    thrust::device_vector<unsigned long long> d_mask;
    
    // We don't need explicit reserve logic as thrust::vector handles it,
    // but calling resize is enough to reuse the underlying capacity if sufficient.
};

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

// Main NMS Function (Ultimate CUDA)
// ---------------------------------
// 1. Hybrid Check: If boxes < 1000, use CPU SIMD.
// 2. Sort on GPU using Thrust.
// 3. Launch Bitmask Kernel.
// 4. Suppress on CPU using the bitmask.
std::vector<Box> nms_cuda_ultimate(std::vector<Box>& boxes, float iou_threshold, GpuMemory& mem) {
    if (boxes.empty()) return {};

    // Hybrid Strategy: Use CPU SIMD for small number of boxes
    // Threshold can be tuned. 1000 is a reasonable starting point.
    if (boxes.size() < 1000) {
        return nms_simd_cpu(boxes, iou_threshold);
    }

    int n = boxes.size();

    // 1. Copy to GPU (Thrust handles allocation/resizing)
    if (mem.d_boxes.size() < n) mem.d_boxes.resize(n);
    thrust::copy(boxes.begin(), boxes.end(), mem.d_boxes.begin());

    // 2. Sort on GPU
    thrust::sort(mem.d_boxes.begin(), mem.d_boxes.begin() + n, BoxComparator());

    // Get raw pointer
    Box* d_boxes_ptr = thrust::raw_pointer_cast(mem.d_boxes.data());

    // 3. Allocate mask
    int num_chunks = (n + 63) / 64;
    if (mem.d_mask.size() < n * num_chunks) mem.d_mask.resize(n * num_chunks);
    unsigned long long* d_mask_ptr = thrust::raw_pointer_cast(mem.d_mask.data());

    // 4. Launch Kernel
    // 1D grid of threads, each thread handles one row 'i'
    int blockSize = 256;
    int numBlocksY = (n + blockSize - 1) / blockSize;
    dim3 grid(1, numBlocksY);
    dim3 block(1, blockSize);

    nms_kernel_ultimate<<<grid, block>>>(d_boxes_ptr, n, iou_threshold, d_mask_ptr);
    cudaDeviceSynchronize();

    // 5. Copy back
    // We need the sorted boxes to return the correct ones
    std::vector<Box> sorted_boxes(n);
    thrust::copy(mem.d_boxes.begin(), mem.d_boxes.begin() + n, sorted_boxes.begin());

    std::vector<unsigned long long> h_mask(n * num_chunks);
    thrust::copy(mem.d_mask.begin(), mem.d_mask.begin() + n * num_chunks, h_mask.begin());

    std::vector<Box> kept_boxes;
    std::vector<bool> suppressed(n, false);

    for (int i = 0; i < n; ++i) {
        if (suppressed[i]) continue;
        kept_boxes.push_back(sorted_boxes[i]);
        
        // Use the bitmask to suppress
        for (int chunk = 0; chunk < num_chunks; ++chunk) {
            unsigned long long mask_chunk = h_mask[i * num_chunks + chunk];
            if (mask_chunk == 0) continue;

            for (int bit = 0; bit < 64; ++bit) {
                if (mask_chunk & (1ULL << bit)) {
                    int j = chunk * 64 + bit;
                    if (j < n) {
                        suppressed[j] = true;
                    }
                }
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
        std::vector<Box> result = nms_cuda_ultimate(boxes, iou_thresh, mem);
        total_boxes_before += boxes.size();
        total_boxes_after += result.size();
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double, std::milli>(end_total - start_total).count();

    std::cout << "=== CUDA Ultimate Results ===" << std::endl;
    std::cout << "Processed " << file_paths.size() << " images." << std::endl;
    std::cout << "Total Boxes Processed: " << total_boxes_before << std::endl;
    std::cout << "Total Boxes Kept:      " << total_boxes_after << std::endl;
    std::cout << "Total Time:             " << total_time << " ms" << std::endl;

    return 0;
}
