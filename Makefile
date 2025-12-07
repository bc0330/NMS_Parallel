CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -Wall
LDFLAGS = -fopenmp

all: nms_seq nms_simd nms_omp nms_tbb nms_simd_tbb

nms_seq: nms_seq.cpp
	$(CXX) $(CXXFLAGS) -o nms_seq.out nms_seq.cpp

nms_simd: nms_simd.cpp
	$(CXX) $(CXXFLAGS) -mavx2 -o nms_simd.out nms_simd.cpp

nms_omp: nms_omp.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o nms_omp.out nms_omp.cpp

nms_tbb: nms_tbb.cpp
	$(CXX) $(CXXFLAGS) nms_tbb.cpp -o nms_tbb.out $$(pkg-config --cflags --libs tbb)

nms_simd_tbb: nms_simd_tbb.cpp
	$(CXX) $(CXXFLAGS) -mavx2 nms_simd_tbb.cpp -o nms_simd_tbb.out $$(pkg-config --cflags --libs tbb)

clean:
	rm -f nms_seq.out nms_simd.out nms_omp.out nms_tbb.out nms_simd_tbb.out