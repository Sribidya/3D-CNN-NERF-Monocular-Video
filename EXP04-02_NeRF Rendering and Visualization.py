# =================================================================
# visualize_training.py
# Loads training history and plots the loss curve.
# Requires: matplotlib
# =================================================================
import pickle
import matplotlib.pyplot as plt
import os
import sys

def visualize_loss(history_file="training_history.pkl"):
    """
    Loads loss history from a file and plots the training curve.
    """
    if not os.path.exists(history_file):
        print(f"Error: Training history file not found at '{history_file}'")
        print("Please ensure simplenerf.py has been run successfully to generate this file.")
        sys.exit(1)

    try:
        with open(history_file, "rb") as f:
            data = pickle.load(f)
            loss_history = data.get("loss_history", [])
            
    except Exception as e:
        print(f"Error loading data from {history_file}: {e}")
        sys.exit(1)

    if not loss_history:
        print("No loss data found in the history file.")
        sys.exit(0)

    epochs = range(1, len(loss_history) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_history, marker='o', linestyle='-', color='skyblue', linewidth=2)
    
    plt.title("NeRF Training Loss (MSE) Over Epochs", fontsize=16, fontweight='bold')
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss (MSE)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(epochs) # Ensure x-axis ticks align with epoch numbers
    plt.tight_layout()
    
    # Save the plot
    plot_path = "training_loss_plot.png"
    plt.savefig(plot_path)
    print(f"Successfully generated and saved plot to {plot_path}")
    
    # Optional: Display the plot (comment out if running in headless environment)
    # plt.show()


if __name__ == "__main__":
    visualize_loss()