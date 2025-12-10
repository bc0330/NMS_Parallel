CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -Wall
# NVCC Compiler
NVCC = nvcc
NVCCFLAGS = -O3 -arch=sm_61 -std=c++17 -Xcompiler -mavx2
LDFLAGS = -fopenmp

all: nms_seq nms_simd nms_omp nms_cuda_naive nms_cuda_opt nms_cuda_ultimate # nms_tbb nms_simd_tbb nms_tbb_old nms_simd_tbb_old

nms_seq: nms_seq.cpp
	$(CXX) $(CXXFLAGS) -o nms_seq.out nms_seq.cpp

nms_simd: nms_simd.cpp
	$(CXX) $(CXXFLAGS) -mavx2 -o nms_simd.out nms_simd.cpp

nms_omp: nms_omp.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o nms_omp.out nms_omp.cpp

nms_cuda_naive: nms_cuda_naive.cu
	$(NVCC) $(NVCCFLAGS) -o nms_cuda_naive.out nms_cuda_naive.cu

nms_cuda_opt: nms_cuda_opt.cu
	$(NVCC) $(NVCCFLAGS) -o nms_cuda_opt.out nms_cuda_opt.cu

nms_cuda_ultimate: nms_cuda_ultimate.cu
	$(NVCC) $(NVCCFLAGS) -o nms_cuda_ultimate.out nms_cuda_ultimate.cu

# nms_tbb: nms_tbb.cpp
# 	$(CXX) $(CXXFLAGS) nms_tbb.cpp -o nms_tbb.out $$(pkg-config --cflags --libs tbb)

# nms_simd_tbb: nms_simd_tbb.cpp
# 	$(CXX) $(CXXFLAGS) -mavx2 nms_simd_tbb.cpp -o nms_simd_tbb.out $$(pkg-config --cflags --libs tbb)

# nms_tbb_old: nms_tbb_old.cpp
# 	$(CXX) $(CXXFLAGS) nms_tbb_old.cpp -o nms_tbb_old.out $$(pkg-config --cflags --libs tbb)

# nms_simd_tbb_old: nms_simd_tbb_old.cpp
# 	$(CXX) $(CXXFLAGS) -mavx2 nms_simd_tbb_old.cpp -o nms_simd_tbb_old.out $$(pkg-config --cflags --libs tbb)

clean:
	rm -f nms_seq.out nms_simd.out nms_omp.out nms_tbb.out nms_simd_tbb.out nms_tbb_old.out nms_simd_tbb_old.out nms_cuda_naive.out nms_cuda_opt.out nms_cuda_ultimate.out

# Remote Execution
# Usage: make run_remote_seq CCID=your_ccid

# Check if CCID is defined, default to wjchiang if not
CCID ?= wjchiang

# Randomly select a login node (1-3)
RANDOM_NODE := $(shell shuf -i 1-3 -n 1)
REMOTE_HOST := hpclogin0$(RANDOM_NODE).cs.nycu.edu.tw
# Default remote directory, can be overridden
REMOTE_DIR := ~/NMS_Parallel

# SSH command prefix
# We use bash -l -c to force a login shell so that 'module' and other environment variables are loaded
# We use a macro to handle the quoting correctly
REMOTE_RUN = ssh -t $(CCID)@$(REMOTE_HOST) "bash -l -c 'cd $(REMOTE_DIR) && module load pp && $(1)'"

# Sync data to remote
sync_data:
	scp -r coco_val_bins $(CCID)@$(REMOTE_HOST):$(REMOTE_DIR)/

# Remote Compile
compile_remote:
	$(call REMOTE_RUN,make all)

# Default cores for OpenMP/TBB
CORES ?= 4

run_remote_seq:
	$(call REMOTE_RUN,run -- ./nms_seq.out coco_val_bins)

run_remote_simd:
	$(call REMOTE_RUN,run -- ./nms_simd.out coco_val_bins)

run_remote_omp:
	$(call REMOTE_RUN,run -c $(CORES) -- ./nms_omp.out coco_val_bins)

# run_remote_tbb:
# 	$(call REMOTE_RUN,run -c $(CORES) -- ./nms_tbb.out coco_val_bins)

# run_remote_simd_tbb:
# 	$(call REMOTE_RUN,run -c $(CORES) -- ./nms_simd_tbb.out coco_val_bins)

# Note: nms_mpi.out must be compiled separately or added to the build targets
run_remote_mpi:
	$(call REMOTE_RUN,run --mpi=pmix -N 4 ./nms_mpi.out coco_val_bins)

run_remote_cuda_naive:
	$(call REMOTE_RUN,run -- ./nms_cuda_naive.out coco_val_bins)

run_remote_cuda_opt:
	$(call REMOTE_RUN,run -- ./nms_cuda_opt.out coco_val_bins)

run_remote_cuda_ultimate:
	$(call REMOTE_RUN,run -- ./nms_cuda_ultimate.out coco_val_bins)