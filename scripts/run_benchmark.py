import argparse
import matplotlib.pyplot as plt
from benchmarks.matrix_multiplication import matrix_multiplication  # Import function

def plot_results(sizes, cpu_times, gpu_times, speedups):
    """
    Plot CPU time, GPU time, and speedup as a function of matrix size.
    """
    plt.figure(figsize=(10, 6))

    # Plot CPU and GPU times
    plt.plot(sizes, cpu_times, label="CPU Time", color='b', marker='o')
    plt.plot(sizes, gpu_times, label="GPU Time", color='r', marker='x')

    # Plot Speedup
    plt.plot(sizes, speedups, label="Speedups", color='g', marker='^')

    # Labels, title, and legend
    plt.title('Performance Comparison: CPU vs GPU Matrix Multiplication')
    plt.xlabel("Matrix Size (N)")
    plt.ylabel("Time (Seconds)")
    plt.legend()
    plt.grid(True)

    # Display the plot
    plt.show()

def main():
    """
    Main function to handle user input, run matrix multiplication tests, and plot results.
    """
    parser = argparse.ArgumentParser(description="Matrix multiplication performance: CPU vs GPU.")
    parser.add_argument('--sizes', type=int, nargs='+', default=[100, 500, 1000, 1500, 2000],
                        help="List of matrix sizes to test.")
    args = parser.parse_args()  # Parse user arguments

    # Initialize lists to store results
    cpu_times = []
    gpu_times = []
    speedups = []

    for size in args.sizes:
        print(f"Testing matrix size: {size} x {size}")
        
        cpu_time, gpu_time, speedup = matrix_multiplication(size)

        # Store results
        cpu_times.append(cpu_time)
        gpu_times.append(gpu_time)
        speedups.append(speedup)

        # Print individual results
        print(f"CPU Time: {cpu_time:.4f} seconds")
        print(f"GPU Time: {gpu_time:.4f} seconds")
        print(f"Speedup: {speedup:.2f}x\n")

    # Plot results
    plot_results(args.sizes, cpu_times, gpu_times, speedups)

if __name__ == "__main__":
    main()
