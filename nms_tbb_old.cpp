// try to implement Non-Maximum Suppression using SIMD and TBB for parallelism
// but found that it is slower than pure SIMD version
#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <numeric>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

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
        The TBB version of the NMS algorithm
    */
    if (boxes.empty()) return {};

    sort(boxes.begin(), boxes.end(), [](const Box& a, const Box& b) {
        return a.score > b.score;
    });

    size_t n = boxes.size();
    
    // 2. Prepare Bit Matrix
    int col_blocks = (n + 63) / 64;
    std::vector<uint64_t> conflict(static_cast<size_t>(n) * col_blocks, 0);

    // 3. Parallel Conflict Detection using TBB
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n), 
        [&](const tbb::blocked_range<size_t>& r) {
            // Iterate over the range assigned to this task
            for (size_t i = r.begin(); i != r.end(); ++i) {
                uint64_t *row_ptr = &conflict[i * col_blocks];

                // Inner loop performs comparisons
                for (size_t j = i + 1; j < n; ++j) {
                    if (boxes[i].class_id != boxes[j].class_id) continue;

                    if (computeIoU(boxes[i], boxes[j]) > iou_threshold) {
                        size_t col_block = j / 64;
                        size_t bit_pos = j % 64;

                        row_ptr[col_block] |= (1ULL << bit_pos);
                    }
                }
            }
        }
    );

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
        
        std::vector<Box> result = nms_parallel(boxes, iou_thresh);

        total_boxes_before += boxes.size();
        total_boxes_after += result.size();
    }

    auto end_total = std::chrono::high_resolution_clock::now();

    double total_time = std::chrono::duration<double, std::milli>(end_total - start_total).count();

    std::cout << "=== TBB Old Results ===" << std::endl;
    std::cout << "Processed " << file_paths.size() << " images." << std::endl;
    std::cout << "Total Boxes Processed: " << total_boxes_before << std::endl;
    std::cout << "Total Boxes Kept:      " << total_boxes_after << std::endl;
    std::cout << "Total Time:             " << total_time << " ms" << std::endl;

    return 0;
}