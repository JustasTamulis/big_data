import timeit_wrapper as tw
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import psutil
import multiprocessing as mp
from detection import process_file_in_chunks
from joblib import Memory

# Set up joblib Memory for caching
memory = Memory(location="output/joblib_cache", verbose=0)


def measure_time_and_resources(func):
    """Decorator to measure execution time, CPU and memory usage"""

    def wrapper(*args, **kwargs):
        # Memory before
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        cpu_percent = psutil.cpu_percent()

        # Time measurement
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time

        # Memory after
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_used = mem_after - mem_before

        return {
            "execution_time": execution_time,
            "cpu_percent": cpu_percent,
            "memory_used": mem_used,
        }

    return wrapper


@measure_time_and_resources
def timed_process_file_in_chunks(file_path, chunk_size, num_processes):
    """Process vessel data with specific parameters"""
    return process_file_in_chunks(
        file_path, chunk_size=chunk_size, num_processes=num_processes
    )


@memory.cache
def process_with_params(file_path, chunk_size, num_processes):
    """Cached version of process_file_in_chunks"""
    return timed_process_file_in_chunks(
        file_path, chunk_size=chunk_size, num_processes=num_processes
    )


def analyze_performance(
    file_path,
    chunk_sizes=[10000, 50000, 100000, 500000, 1000000],
    process_counts=[2, 4, 8, 12, 16],
):
    """Comprehensive performance analysis across different configurations"""

    results = []

    # Run sequential first (using 1 process)
    print("\nMeasuring sequential processing...")
    sequential_metrics = process_with_params(file_path, chunk_sizes[-1], 1)
    sequential_time = sequential_metrics["execution_time"]

    results.append(
        {
            "processes": 1,
            "chunk_size": chunk_sizes[0],
            "time": sequential_time,
            "speedup": 1.0,
            "efficiency": 1.0,
            "cpu_percent": sequential_metrics["cpu_percent"],
            "memory_used": sequential_metrics["memory_used"],
        }
    )

    for chunk_size in chunk_sizes:
        for num_processes in process_counts:
            print(
                f"\nTesting with {num_processes} processes, chunk size {chunk_size}..."
            )

            metrics = process_with_params(file_path, chunk_size, num_processes)
            print(metrics)
            parallel_time = metrics["execution_time"]

            speedup = sequential_time / parallel_time
            efficiency = (
                speedup / num_processes
            )  # How effectively we use each processor

            results.append(
                {
                    "processes": num_processes,
                    "chunk_size": chunk_size,
                    "time": parallel_time,
                    "speedup": speedup,
                    "efficiency": efficiency,
                    "cpu_percent": metrics["cpu_percent"],
                    "memory_used": metrics["memory_used"],
                }
            )

    return pd.DataFrame(results)


def plot_performance_metrics(performance_df):
    """Create comprehensive visualizations of performance metrics"""

    # 1. Execution Time vs Processes for different chunk sizes
    plt.figure(figsize=(10, 6))
    for chunk_size in performance_df["chunk_size"].unique():
        df_chunk = performance_df[performance_df["chunk_size"] == chunk_size]
        plt.plot(
            df_chunk["processes"],
            df_chunk["time"],
            marker="o",
            label=f"Chunk size {chunk_size}",
        )
    plt.xlabel("Number of Processes")
    plt.ylabel("Execution Time (s)")
    plt.title("Execution Time vs Number of Processes")
    plt.legend()
    plt.grid(True)
    plt.savefig("output/execution_time_plot.png")
    plt.close()

    # 2. Speedup Analysis
    plt.figure(figsize=(10, 6))
    for chunk_size in performance_df["chunk_size"].unique():
        df_chunk = performance_df[performance_df["chunk_size"] == chunk_size]
        plt.plot(
            df_chunk["processes"],
            df_chunk["speedup"],
            marker="o",
            label=f"Chunk size {chunk_size}",
        )
    # Add ideal speedup line
    max_procs = performance_df["processes"].max()
    #plt.plot([1, max_procs], [1, max_procs], "k--", label="Ideal Speedup")
    plt.xlabel("Number of Processes")
    plt.ylabel("Speedup")
    plt.title("Speedup vs Number of Processes")
    plt.legend()
    plt.grid(True)
    plt.savefig("output/speedup_plot.png")
    plt.close()

    # 3. CPU Usage
    plt.figure(figsize=(10, 6))
    for chunk_size in performance_df["chunk_size"].unique():
        df_chunk = performance_df[performance_df["chunk_size"] == chunk_size]
        plt.plot(
            df_chunk["processes"],
            df_chunk["cpu_percent"],
            marker="o",
            label=f"Chunk size {chunk_size}",
        )
    plt.xlabel("Number of Processes")
    plt.ylabel("CPU Usage (%)")
    plt.title("CPU Usage vs Number of Processes")
    plt.legend()
    plt.grid(True)
    plt.savefig("output/cpu_usage_plot.png")
    plt.close()

    # 4. Memory Usage
    plt.figure(figsize=(10, 6))
    for chunk_size in performance_df["chunk_size"].unique():
        df_chunk = performance_df[performance_df["chunk_size"] == chunk_size]
        plt.plot(
            df_chunk["processes"],
            df_chunk["memory_used"],
            marker="o",
            label=f"Chunk size {chunk_size}",
        )
    plt.xlabel("Number of Processes")
    plt.ylabel("Memory Usage (MB)")
    plt.title("Memory Usage vs Number of Processes")
    plt.legend()
    plt.grid(True)
    plt.savefig("output/memory_usage_plot.png")
    plt.close()

    # 5. Heatmap of Execution Time (processes vs chunk size) using matplotlib
    plt.figure(figsize=(10, 8))

    # Get unique values for processes and chunk sizes
    processes = sorted(performance_df["processes"].unique())
    chunk_sizes = sorted(performance_df["chunk_size"].unique())

    # Create a 2D array for the heatmap
    speedup_matrix = np.zeros((len(chunk_sizes), len(processes)))

    # Fill the matrix with speedup values
    for i, chunk_size in enumerate(chunk_sizes):
        for j, proc in enumerate(processes):
            # Get speedup for this configuration
            filtered_df = performance_df[
                (performance_df["chunk_size"] == chunk_size)
                & (performance_df["processes"] == proc)
            ]
            if not filtered_df.empty:
                speedup_matrix[i, j] = filtered_df["speedup"].values[0]

    # Create heatmap with color scale from 0 to max speedup
    heatmap = plt.imshow(
        speedup_matrix,
        cmap="RdYlGn",
        aspect="auto",
        vmin=1,
        vmax=np.max(speedup_matrix),
    )
    plt.colorbar(heatmap, label="Speedup")

    # Set x and y axis labels
    plt.xticks(range(len(processes)), processes)
    plt.yticks(range(len(chunk_sizes)), [f"{cs}" for cs in chunk_sizes])

    plt.xlabel("Number of Processes")
    plt.ylabel("Chunk Size")
    plt.title("Speedup Heatmap (Processes vs Chunk Size)")

    # Add text annotations with speedup values
    for i in range(len(chunk_sizes)):
        for j in range(len(processes)):
            plt.text(
                j,
                i,
                f"{speedup_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
            )

    plt.tight_layout()
    plt.savefig("output/speedup_heatmap.png")
    plt.close()


if __name__ == "__main__":
    # Set file path
    # file_path = "data/aisdk-small.csv"
    file_path = "data/aisdk-2025-02-09.csv"

    max_processes = mp.cpu_count()
    print(max_processes)
    # process_counts = np.linspace(2, max_processes, 5, dtype=int)

    # Analyze performance with different configurations
    print("\nAnalyzing performance...")
    performance_df = analyze_performance(
        file_path,
        chunk_sizes=[10000, 50000, 100000, 500000, 1000000, 5000000],
        process_counts=[1, 2, 4, 8, 12, 16, 24, 32, 48],
        # chunk_sizes=[10000, 100000, 1000000],
        # process_counts = [2, 8, 16]
    )

    # Plot and save results
    print("\nGenerating performance visualizations...")
    plot_performance_metrics(performance_df)

    # Save detailed results to CSV
    performance_df.to_csv("output/performance_metrics.csv", index=False)

    # Print summary
    print("\nPerformance analysis summary:")
    print("\nBest performing configuration:")
    best_config = performance_df.loc[performance_df["speedup"].idxmax()]
    print(f"Processes: {best_config['processes']}")
    print(f"Chunk size: {best_config['chunk_size']}")
    print(f"Speedup: {best_config['speedup']:.2f}x")
    print(f"Efficiency: {best_config['efficiency']:.2f}")
    print(f"\nDetailed results saved to 'performance_metrics.csv'")
    print(f"Visualizations saved to individual plots in the 'output' directory")
