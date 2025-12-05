CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -Wall

all: nms_seq nms_simd

nms_seq: nms_seq.cpp
	$(CXX) $(CXXFLAGS) -o nms_seq nms_seq.cpp

nms_simd: nms_simd.cpp
	$(CXX) $(CXXFLAGS) -mavx2 -o nms_simd nms_simd.cpp

clean:
	rm -f nms_seq nms_simd