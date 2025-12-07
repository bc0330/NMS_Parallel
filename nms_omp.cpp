#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <numeric>
#include <omp.h>

namespace fs = std::filesystem;

struct Box {
    float x1, y1, x2, y2;
    float score;
    float class_id;
};

inline float computeIoU(const Box& a, const Box& b) {
    /*
        Calculate IoU given two bbox
    */

    // Calculate intersection coordinates
    float xx1 = std::max(a.x1, b.x1);
    float yy1 = std::max(a.y1, b.y1);
    float xx2 = std::min(a.x2, b.x2);
    float yy2 = std::min(a.y2, b.y2);

    // Calculate intersection area
    float w = std::max(0.0f, xx2 - xx1);
    float h = std::max(0.0f, yy2 - yy1);
    float inter_area = w * h;

    // Calculate union area
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    float union_area = area_a + area_b - inter_area;

    // Avoid division by zero
    if (union_area <= 0.0f) return 0.0f;

    return inter_area / union_area;
}


std::vector<Box> nms_parallel(std::vector<Box>& boxes, float iou_threshold) {
    /*
        The OpenMP version of the NMS algorithm
    */
    if (boxes.empty()) return {};

    // 1. Sort Descending
    std::sort(boxes.begin(), boxes.end(), [](const Box& a, const Box& b) {
        return a.score > b.score;
    });

    size_t n = boxes.size();
    
    // 2. Prepare Bit Matrix
    // We use uint64_t to store 64 bits per element. 
    // Rows = n, Cols = (n + 63) / 64 blocks
    int col_blocks = (n + 63) / 64;
    std::vector<uint64_t> conflict(static_cast<size_t>(n) * col_blocks, 0);

    // 3. Parallel Conflict Detection
    #pragma omp parallel for schedule(dynamic, 32)
    for (size_t i = 0; i < n; ++i) {
        uint64_t *row_ptr = &conflict[i * col_blocks];

        for (size_t j = i + 1; j < n; ++j) {
            if (boxes[i].class_id != boxes[j].class_id) continue;


            if (computeIoU(boxes[i], boxes[j]) > iou_threshold) {
                size_t col_block = j / 64;
                size_t bit_pos = j % 64;
                row_ptr[col_block] |= (1ULL << bit_pos);
            }
        }
    }

    // 4. Sequential Reduction
    std::vector<uint8_t> suppressed(n, 0);
    std::vector<Box> kept_boxes;
    kept_boxes.reserve(n);

    for (size_t i = 0; i < n; ++i) {
        if (suppressed[i]) continue;

        kept_boxes.emplace_back(boxes[i]);

        // Process conflicts for box i
        uint64_t *row_ptr = &conflict[i * col_blocks];
        int start_block = i / 64;

        for (int b = start_block; b < col_blocks; ++b) {
            uint64_t block = row_ptr[b];
            if (block == 0) continue;

            while (block) {
                int bit_pos = __builtin_ctzll(block); // Count Trailing Zeros
                size_t j = static_cast<size_t>(b) * 64 + bit_pos;

                if (j > i && j < n) {
                    suppressed[j] = 1;
                }
                
                // Clear the bit
                block &= ~(1ULL << bit_pos);
            }
        }
    }

    return kept_boxes;
}

std::vector<Box> load_boxes(const std::string& filepath) {
    /*
    Helper function to load bounding boxes from a binary file
    */
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

int main() {
    std::string input_folder = "coco_val_bins";
    float iou_thresh = 0.5f;    
    float conf_thresh = 0.0f;  

    // Determine max threads
    int max_threads = omp_get_max_threads();
    std::cout << "Threads: " << max_threads << std::endl;

    std::vector<double> times;
    int total_boxes_before = 0;
    int total_boxes_after = 0;

    if (!fs::exists(input_folder)) {
        std::cerr << "Folder not found!" << std::endl;
        return 1;
    }

    for (const auto& entry : fs::directory_iterator(input_folder)) {
        if (entry.path().extension() == ".bin") {
            std::vector<Box> boxes = load_boxes(entry.path().string());
            
            std::vector<Box> filtered_boxes;
            filtered_boxes.reserve(boxes.size());
            for(const auto& b : boxes) {
                if(b.score >= conf_thresh) filtered_boxes.push_back(b);
            }

            auto start = std::chrono::high_resolution_clock::now();
            std::vector<Box> result = nms_parallel(filtered_boxes, iou_thresh);
            auto end = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double, std::milli> duration = end - start;
            times.push_back(duration.count());

            total_boxes_before += filtered_boxes.size();
            total_boxes_after += result.size();
        }
    }

    if (times.empty()) return 0;

    double total_time = std::accumulate(times.begin(), times.end(), 0.0);
    double avg_time = total_time / times.size();

    std::cout << "=== OpenMP Results ===" << std::endl;
    std::cout << "Processed " << times.size() << " images." << std::endl;
    std::cout << "Total Boxes Processed: " << total_boxes_before << std::endl;
    std::cout << "Total Boxes Kept:      " << total_boxes_after << std::endl;
    std::cout << "Average Time per Image: " << avg_time << " ms" << std::endl;
    std::cout << "Total Time:             " << total_time << " ms" << std::endl;


    return 0;
}