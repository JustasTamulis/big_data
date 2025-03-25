import timeit_wrapper as tw
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import psutil
import multiprocessing as mp
from detection import load_data, detect_vessel_anomalies


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
            "result": result,
            "execution_time": execution_time,
            "cpu_percent": cpu_percent,
            "memory_used": mem_used,
        }

    return wrapper


@measure_time_and_resources
def process_sequential(df):
    """Process vessel data sequentially"""
    results = []
    for mmsi, group in df.groupby("MMSI"):
        results.append(detect_vessel_anomalies(group))
    return results


@measure_time_and_resources
def process_parallel(df, num_processes, chunk_size=10):
    """Process vessel data in parallel"""
    vessel_groups = (group for _, group in df.groupby("MMSI"))
    with mp.Pool(processes=num_processes) as pool:
        results = pool.imap_unordered(
            detect_vessel_anomalies, vessel_groups, chunksize=chunk_size
        )
        return list(results)


def analyze_performance(df, max_processes=None, chunk_sizes=[1, 5, 10, 20, 50]):
    """Comprehensive performance analysis across different configurations"""
    if max_processes is None:
        max_processes = mp.cpu_count()

    results = []

    # Run sequential first
    print("\nMeasuring sequential processing...")
    sequential_metrics = process_sequential(df)
    sequential_time = sequential_metrics["execution_time"]

    results.append(
        {
            "processes": 1,
            "chunk_size": "N/A",
            "time": sequential_time,
            "speedup": 1.0,
            "efficiency": 1.0,
            "cpu_percent": sequential_metrics["cpu_percent"],
            "memory_used": sequential_metrics["memory_used"],
        }
    )

    # Create 5 equally spaced process counts from 2 to max_processes
    process_counts = np.linspace(2, max_processes, 5, dtype=int)
    for num_processes in process_counts:
        for chunk_size in chunk_sizes:
            print(
                f"\nTesting with {num_processes} processes, chunk size {chunk_size}..."
            )

            metrics = process_parallel(df, num_processes, chunk_size)
            parallel_time = metrics["execution_time"]

            speedup = sequential_time / parallel_time
            efficiency = (
                speedup / num_processes
            )  # This shows how effectively we use each processor

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
    fig = plt.figure(figsize=(20, 15))

    # 1. Execution Time vs Processes for different chunk sizes
    ax1 = plt.subplot(2, 2, 1)
    for chunk_size in performance_df["chunk_size"].unique():
        if chunk_size != "N/A":
            df_chunk = performance_df[performance_df["chunk_size"] == chunk_size]
            ax1.plot(
                df_chunk["processes"],
                df_chunk["time"],
                marker="o",
                label=f"Chunk size {chunk_size}",
            )
    ax1.set_xlabel("Number of Processes")
    ax1.set_ylabel("Execution Time (s)")
    ax1.set_title("Execution Time vs Number of Processes")
    ax1.legend()
    ax1.grid(True)

    # 2. Speedup Analysis
    ax2 = plt.subplot(2, 2, 2)
    for chunk_size in performance_df["chunk_size"].unique():
        if chunk_size != "N/A":
            df_chunk = performance_df[performance_df["chunk_size"] == chunk_size]
            ax2.plot(
                df_chunk["processes"],
                df_chunk["speedup"],
                marker="o",
                label=f"Chunk size {chunk_size}",
            )
    # Add ideal speedup line
    max_procs = performance_df["processes"].max()
    ax2.plot([1, max_procs], [1, max_procs], "k--", label="Ideal Speedup")
    ax2.set_xlabel("Number of Processes")
    ax2.set_ylabel("Speedup")
    ax2.set_title("Speedup vs Number of Processes")
    ax2.legend()
    ax2.grid(True)

    # 3. CPU Usage
    ax3 = plt.subplot(2, 2, 3)
    for chunk_size in performance_df["chunk_size"].unique():
        if chunk_size != "N/A":
            df_chunk = performance_df[performance_df["chunk_size"] == chunk_size]
            ax3.plot(
                df_chunk["processes"],
                df_chunk["cpu_percent"],
                marker="o",
                label=f"Chunk size {chunk_size}",
            )
    ax3.set_xlabel("Number of Processes")
    ax3.set_ylabel("CPU Usage (%)")
    ax3.set_title("CPU Usage vs Number of Processes")
    ax3.legend()
    ax3.grid(True)

    # 4. Memory Usage
    ax4 = plt.subplot(2, 2, 4)
    for chunk_size in performance_df["chunk_size"].unique():
        if chunk_size != "N/A":
            df_chunk = performance_df[performance_df["chunk_size"] == chunk_size]
            ax4.plot(
                df_chunk["processes"],
                df_chunk["memory_used"],
                marker="o",
                label=f"Chunk size {chunk_size}",
            )
    ax4.set_xlabel("Number of Processes")
    ax4.set_ylabel("Memory Usage (MB)")
    ax4.set_title("Memory Usage vs Number of Processes")
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig("performance_analysis.png")
    plt.close()


if __name__ == "__main__":
    # Load and prepare data
    file_path = "data/aisdk-test.csv"
    print("Loading data...")
    df = load_data(file_path)
    print("Data length: ", len(df))

    # Analyze performance with different configurations
    print("\nAnalyzing performance...")
    performance_df = analyze_performance(df, chunk_sizes=[1, 5, 10, 20, 50])

    # Plot and save results
    print("\nGenerating performance visualizations...")
    plot_performance_metrics(performance_df)

    # Save detailed results to CSV
    performance_df.to_csv("performance_metrics.csv", index=False)

    # Print summary
    print("\nPerformance analysis summary:")
    print("\nBest performing configuration:")
    best_config = performance_df.loc[performance_df["speedup"].idxmax()]
    print(f"Processes: {best_config['processes']}")
    print(f"Chunk size: {best_config['chunk_size']}")
    print(f"Speedup: {best_config['speedup']:.2f}x")
    print(f"Efficiency: {best_config['efficiency']:.2f}")
    print(f"\nDetailed results saved to 'performance_metrics.csv'")
    print(f"Visualizations saved to 'performance_analysis.png'")
