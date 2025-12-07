#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <numeric>
#include <immintrin.h>

namespace fs = std::filesystem;

// Bounding Box structure
struct Box {
    float x1, y1, x2, y2;
    float score;
    float class_id;
};

// Sructure of Arrays for bboxes
struct SoABoxes {
    std::vector<float> x1;
    std::vector<float> y1;
    std::vector<float> x2;
    std::vector<float> y2;
    std::vector<float> area;
    std::vector<float> scores;
    std::vector<float> class_ids;
    size_t size;

    void reserve(size_t n) {
        x1.reserve(n);
        y1.reserve(n);
        x2.reserve(n);
        y2.reserve(n);
        area.reserve(n);
        scores.reserve(n);
        class_ids.reserve(n);
        size = 0;
    }

    void push_back(const Box& box) {
        x1.push_back(box.x1);
        y1.push_back(box.y1);
        x2.push_back(box.x2);
        y2.push_back(box.y2);
        float w = std::max(0.0f, box.x2 - box.x1);
        float h = std::max(0.0f, box.y2 - box.y1);
        area.push_back(w * h);
        scores.push_back(box.score);
        class_ids.push_back(box.class_id);
        size++;
    }
};

inline float computeIoU_Single(const SoABoxes& boxes, size_t i, size_t j) {
    /*
        Helper function to compute IoU for single box pair
    */

    // Calculate intersection coordinates
    float xx1 = std::max(boxes.x1[i], boxes.x1[j]);
    float yy1 = std::max(boxes.y1[i], boxes.y1[j]);
    float xx2 = std::min(boxes.x2[i], boxes.x2[j]);
    float yy2 = std::min(boxes.y2[i], boxes.y2[j]);

    // Calculate intersection area
    float w = std::max(0.0f, xx2 - xx1);
    float h = std::max(0.0f, yy2 - yy1);
    float inter_area = w * h;

    // Calculate union area
    float union_area = boxes.area[i] + boxes.area[j] - inter_area;
    return (union_area > 0.0f) ? (inter_area / union_area) : 0.0f;
}

std::vector<Box> nms_simd(std::vector<Box>& boxes, float iou_threshold) {
    /*
        The SIMD-optimized version of the NMS algorithm
    */
    if (boxes.empty()) return {};
    size_t n = boxes.size();

    // Sort boxes by score (Descending)
    std::sort(boxes.begin(), boxes.end(), [](const Box& a, const Box& b) {
        return a.score > b.score;
    });

    // Convert to Structure of Arrays (SoA) for SIMD processing
    SoABoxes soa_boxes;
    soa_boxes.reserve(boxes.size());
    for (const auto& box : boxes) {
        soa_boxes.push_back(box);
    }

    std::vector<uint8_t> suppressed(boxes.size(), 0);  // suppression flags
    std::vector<Box> kept_boxes;

    __m256 v_thres = _mm256_set1_ps(iou_threshold);
    __m256 v_zero = _mm256_set1_ps(0.0f);

    for (size_t i = 0; i < n; ++i) {
        if (suppressed[i]) continue;

        kept_boxes.push_back(boxes[i]);
        
        __m256  v_x1_i = _mm256_set1_ps(soa_boxes.x1[i]);
        __m256  v_y1_i = _mm256_set1_ps(soa_boxes.y1[i]);
        __m256  v_x2_i = _mm256_set1_ps(soa_boxes.x2[i]);
        __m256  v_y2_i = _mm256_set1_ps(soa_boxes.y2[i]);
        __m256  v_area_i = _mm256_set1_ps(soa_boxes.area[i]);
        __m256  v_class_i = _mm256_set1_ps(soa_boxes.class_ids[i]);

        size_t j = i + 1;;
        for (; j + 8 <= n; j += 8) {
            // check suppression for 8 boxes at once
            __m256 v_x1_j = _mm256_loadu_ps(&soa_boxes.x1[j]);
            __m256 v_y1_j = _mm256_loadu_ps(&soa_boxes.y1[j]);
            __m256 v_x2_j = _mm256_loadu_ps(&soa_boxes.x2[j]);
            __m256 v_y2_j = _mm256_loadu_ps(&soa_boxes.y2[j]);
            __m256 v_area_j = _mm256_loadu_ps(&soa_boxes.area[j]);
            __m256 v_class_j = _mm256_loadu_ps(&soa_boxes.class_ids[j]);

            // Intersection
            __m256 v_xx1 = _mm256_max_ps(v_x1_i, v_x1_j);
            __m256 v_yy1 = _mm256_max_ps(v_y1_i, v_y1_j);
            __m256 v_xx2 = _mm256_min_ps(v_x2_i, v_x2_j);
            __m256 v_yy2 = _mm256_min_ps(v_y2_i, v_y2_j);

            // Width and Height
            __m256 v_w = _mm256_max_ps(v_zero, _mm256_sub_ps(v_xx2, v_xx1));
            __m256 v_h = _mm256_max_ps(v_zero, _mm256_sub_ps(v_yy2, v_yy1));

            // Intersection Area
            __m256 v_inter_area = _mm256_mul_ps(v_w, v_h);

            // Union Area
            __m256 v_union_area = _mm256_sub_ps(_mm256_add_ps(v_area_i, v_area_j), v_inter_area);

            // IoU Calculation
            __m256 v_iou = _mm256_div_ps(v_inter_area, v_union_area);

            // Comparisons
            // 1. IoU > threshold
            __m256 v_mask_iou = _mm256_cmp_ps(v_iou, v_thres, _CMP_GT_OQ);
            // 2. class_id match   
            __m256 v_mask_class = _mm256_cmp_ps(v_class_i, v_class_j, _CMP_EQ_OQ);
            // Combine masks: (IoU > thres) AND (class_id match)
            __m256 v_mask = _mm256_and_ps(v_mask_iou, v_mask_class);
            // Store suppression flags
            int mask = _mm256_movemask_ps(v_mask);
            for (int k = 0; k < 8; ++k) {
                if (mask & (1 << k)) {
                    suppressed[j + k] = 1;
                }
            }
        }

        // Handle remaining boxes
        for (; j < n; ++j) {
            if (suppressed[j]) continue;
            if (soa_boxes.class_ids[i] != soa_boxes.class_ids[j]) continue;
            if (computeIoU_Single(soa_boxes, i, j) > iou_threshold) {
                suppressed[j] = 1;
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
    if (!file.is_open()) {
        std::cerr << "Error opening " << filepath << std::endl;
        return {};
    }

    int num_boxes;
    file.read(reinterpret_cast<char*>(&num_boxes), sizeof(int));

    std::vector<Box> boxes(num_boxes);
    file.read(reinterpret_cast<char*>(boxes.data()), num_boxes * sizeof(Box));
    
    return boxes;
}

int main() {
    std::string input_folder = "coco_val_bins";
    float iou_thresh = 0.5f;    
    float conf_thresh = 0.0f;  

    std::vector<double> times;
    int total_boxes_before = 0;
    int total_boxes_after = 0;

    if (!fs::exists(input_folder)) {
        std::cerr << "Folder not found: " << input_folder << std::endl;
        return 1;
    }

    for (const auto& entry : fs::directory_iterator(input_folder)) {
        if (entry.path().extension() == ".bin") {
            std::vector<Box> boxes = load_boxes(entry.path().string());
            
            // Optional: Pre-filtering low confidence boxes
            // (This is usually done before NMS to save time)
            std::vector<Box> filtered_boxes;
            filtered_boxes.reserve(boxes.size());
            for(const auto& b : boxes) {
                if(b.score >= conf_thresh) filtered_boxes.push_back(b);
            }

            // --- BENCHMARK START ---
            auto start = std::chrono::high_resolution_clock::now();
            
            std::vector<Box> result = nms_simd(filtered_boxes, iou_thresh);
            
            auto end = std::chrono::high_resolution_clock::now();
            // --- BENCHMARK END ---

            std::chrono::duration<double, std::milli> duration = end - start;
            times.push_back(duration.count());

            total_boxes_before += filtered_boxes.size();
            total_boxes_after += result.size();
        }
    }

    if (times.empty()) {
        std::cout << "No files processed." << std::endl;
        return 0;
    }

    double total_time = std::accumulate(times.begin(), times.end(), 0.0);
    double avg_time = total_time / times.size();

    std::cout << "=== SIMD Results ===" << std::endl;
    std::cout << "Processed " << times.size() << " images." << std::endl;
    std::cout << "Total Boxes Processed: " << total_boxes_before << std::endl;
    std::cout << "Total Boxes Kept:      " << total_boxes_after << std::endl;
    std::cout << "Average Time per Image: " << avg_time << " ms" << std::endl;
    std::cout << "Total Time:             " << total_time << " ms" << std::endl;

    return 0;
}