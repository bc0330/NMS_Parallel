#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <numeric>

namespace fs = std::filesystem;

// Bounding Box structure
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


std::vector<Box> nms_sequential(std::vector<Box>& boxes, float iou_threshold) {
    /*
        The sequential version of the NMS algorithm
    */

    if (boxes.empty()) return {};

    // Step 1: Sort boxes by score (Descending)
    std::sort(boxes.begin(), boxes.end(), [](const Box& a, const Box& b) {
        return a.score > b.score;
    });

    std::vector<Box> kept_boxes;                        // the final selected set of bboxes
    std::vector<bool> suppressed(boxes.size(), false);

    // Step 2: Iterate through the boxes 
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (suppressed[i]) continue;

        kept_boxes.push_back(boxes[i]);

        // Check it against all subsequent boxes
        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (suppressed[j]) continue;

            // CLASS-AWARE CHECK:
            // If they are different classes (e.g., Cat vs Dog), do not suppress
            if (boxes[i].class_id != boxes[j].class_id) continue;

            // If overlap is high, suppress the lower score box (j)
            if (computeIoU(boxes[i], boxes[j]) > iou_threshold) {
                suppressed[j] = true;
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
            
            std::vector<Box> result = nms_sequential(filtered_boxes, iou_thresh);
            
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

    std::cout << "=== Sequential Baseline Results ===" << std::endl;
    std::cout << "Processed " << times.size() << " images." << std::endl;
    std::cout << "Total Boxes Processed: " << total_boxes_before << std::endl;
    std::cout << "Total Boxes Kept:      " << total_boxes_after << std::endl;
    std::cout << "Average Time per Image: " << avg_time << " ms" << std::endl;
    std::cout << "Total Time:             " << total_time << " ms" << std::endl;

    return 0;
}