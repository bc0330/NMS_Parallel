import subprocess
import re
import argparse

# Configuration: Define targets here. 
# Set 'enabled' to False to skip a target (e.g., if TBB is not installed).
TARGETS = [
    {"name": "Sequential",      "target": "run_remote_seq",         "enabled": True},
    {"name": "SIMD",            "target": "run_remote_simd",        "enabled": True},
    {"name": "OpenMP (1 core)", "target": "run_remote_omp CORES=1", "enabled": True},
    {"name": "OpenMP (2 cores)","target": "run_remote_omp CORES=2", "enabled": True},
    {"name": "OpenMP (4 cores)","target": "run_remote_omp CORES=4", "enabled": True},
    {"name": "CUDA Naive",      "target": "run_remote_cuda_naive",  "enabled": True},
    {"name": "CUDA Optimized",  "target": "run_remote_cuda_opt",    "enabled": True},
    {"name": "CUDA Ultimate",   "target": "run_remote_cuda_ultimate","enabled": True},
    # To enable TBB, uncomment the lines below and ensure the targets exist in Makefile
    # {"name": "TBB (4 cores)",   "target": "run_remote_tbb CORES=4", "enabled": False},
]

def run_target(target_name, make_target, ccid=None):
    print(f"Running {target_name} ({make_target})...")
    try:
        # Split the make target string into arguments
        args = ["make"] + make_target.split()
        
        # Add CCID if provided
        if ccid:
            args.append(f"CCID={ccid}")
            
        # Run the make command and capture output
        result = subprocess.run(
            args, 
            capture_output=True, 
            text=True, 
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running {target_name}:")
        print(e.stderr)
        return None

def parse_time(output):
    # Look for "Total Time: <number> ms"
    match = re.search(r"Total Time:\s+([\d\.]+)\s+ms", output)
    if match:
        return float(match.group(1))
    return None

def main():
    parser = argparse.ArgumentParser(description="Benchmark NMS implementations.")
    parser.add_argument("--ccid", type=str, help="CCID for remote execution", default=None)
    args = parser.parse_args()

    results = []

    print("Starting Benchmark...")
    if args.ccid:
        print(f"Using CCID: {args.ccid}")
    print("=" * 40)

    for config in TARGETS:
        if not config["enabled"]:
            continue
        
        output = run_target(config["name"], config["target"], args.ccid)
        if output:
            time_ms = parse_time(output)
            if time_ms is not None:
                results.append({
                    "name": config["name"],
                    "time": time_ms
                })
                print(f"  -> Time: {time_ms:.2f} ms")
            else:
                print("  -> Could not parse time from output.")
        else:
            print("  -> Execution failed.")
        print("-" * 40)

    print("\nBenchmark Results")
    print("=" * 60)
    print(f"{'Implementation':<20} | {'Time (ms)':<15} | {'Speedup':<10}")
    print("-" * 60)

    if not results:
        print("No results to display.")
        return

    # Find baseline (Sequential) or use the first one as baseline
    baseline = next((r for r in results if r["name"] == "Sequential"), results[0])
    baseline_time = baseline["time"]

    for res in results:
        speedup = baseline_time / res["time"] if res["time"] > 0 else 0.0
        print(f"{res['name']:<20} | {res['time']:<15.2f} | {speedup:<10.2f}x")
    print("=" * 60)

if __name__ == "__main__":
    main()
