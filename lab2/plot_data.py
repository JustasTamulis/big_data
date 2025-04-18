import numpy as np
import matplotlib.pyplot as plt
import argparse

def main(output_filename):
    """Generates a simple plot and saves it."""
    # Generate some data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    # Create a plot
    plt.figure(figsize=(8, 6))
    plt.plot(x, y)
    plt.title('Simple Sine Wave Plot')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)

    # Save the plot
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a simple plot.')
    parser.add_argument('--output', type=str, default='output_plot.png', 
                        help='Filename for the output plot (default: output_plot.png)')
    args = parser.parse_args()
    
    main(args.output)