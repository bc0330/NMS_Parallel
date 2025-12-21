# Local Benchmark Usage Guide

## Overview
`benchmark_local.py` runs NMS benchmarks locally on your machine (no remote execution needed).

## Prerequisites
1. Build all executables:
   ```bash
   make all
   ```

2. Ensure test data is extracted:
   ```bash
   tar -xzf coco_val_bins.tar.gz
   ```

## Usage

### Run all benchmarks (excluding CUDA if not available):
```bash
python3 benchmark_local.py --disable CUDA
```

### Run with custom data directory:
```bash
python3 benchmark_local.py --data-dir /path/to/data
```

### Disable specific implementations:
```bash
# Disable OpenMP and TBB benchmarks
python3 benchmark_local.py --disable OpenMP TBB

# Disable only CUDA benchmarks
python3 benchmark_local.py --disable CUDA
```

### Run only specific implementations:
```bash
# Run only Sequential and SIMD
python3 benchmark_local.py --disable OpenMP TBB CUDA
```

## How It Works

The script:
- Runs each executable directly (no SSH required)
- Sets `OMP_NUM_THREADS` and `TBB_NUM_THREADS` environment variables for parallel implementations
- Parses "Total Time: X ms" from each program's output
- Calculates speedup relative to Sequential baseline
- Supports selective enabling/disabling of benchmarks

## Comparison with Remote Benchmark

| Feature | `benchmark_local.py` | `benchmark.py` |
|---------|---------------------|----------------|
| Execution | Local machine | Remote server via SSH |
| Setup | Simple | Requires CCID and remote access |
| Thread Control | Environment variables | remote `run -c` command |
| Use Case | Development/testing | Production benchmarking |

## Example Output

```
======================================================================
Benchmark Results
======================================================================
Implementation            | Time (ms)       | Speedup   
----------------------------------------------------------------------
Sequential                | 1903.46         | 1.00      x
SIMD                      | 644.03          | 2.96      x
OpenMP (4 cores)          | 494.04          | 3.85      x
TBB (4 cores)             | 223.97          | 8.50      x
SIMD+TBB (4 cores)        | 75.66           | 25.16     x
======================================================================
```

## Notes
- CUDA benchmarks are disabled by default in `benchmark_local.py` (set `"enabled": True` to enable)
- Thread counts are controlled via environment variables
- Results may vary based on your CPU and available cores
