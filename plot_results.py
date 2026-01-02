import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.figsize': (10, 6), 'figure.dpi': 150})

RESULT_FILE = "experiments/results.csv"
OUTPUT_DIR = "experiments/plots"

def load_data():
    if not os.path.exists(RESULT_FILE):
        print(f"File {RESULT_FILE} not found.")
        return None
    df = pd.read_csv(RESULT_FILE)
    return df

def plot_params_vs_loss(df):
    """Scatter plot of Parameters vs Validation Loss"""
    plt.figure()
    
    # Filter only relevant columns to avoid clutter if multiple runs exists
    # We take the best (lowest val loss) for each experiment type if duplicates exist
    best_df = df.sort_values("Val Loss").groupby("Experiment").first().reset_index()
    
    sns.scatterplot(
        data=best_df, 
        x="Parameters", 
        y="Val Loss", 
        hue="Experiment", 
        style="Experiment",
        s=200,
        palette="viridis"
    )
    
    # Annotate points
    for i, row in best_df.iterrows():
        plt.text(
            row["Parameters"], 
            row["Val Loss"] + 0.002, 
            f"{row['Val Loss']:.3f}", 
            horizontalalignment='center'
        )

    plt.title("Parameter Efficiency: Val Loss vs Model Size")
    plt.xlabel("Total Parameters")
    plt.ylabel("Validation Loss (CrossEntropy)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/params_vs_loss.png")
    plt.close()
    print(f"Saved {OUTPUT_DIR}/params_vs_loss.png")

def plot_bar_metrics(df):
    """Bar charts for Memory and Speed"""
    best_df = df.sort_values("Val Loss").groupby("Experiment").first().reset_index()
    
    # 1. Memory Usage
    plt.figure()
    ax = sns.barplot(data=best_df, x="Experiment", y="Peak Memory (MB)", palette="rocket")
    ax.bar_label(ax.containers[0], fmt='%.0f')
    plt.title("Peak GPU Memory Usage (MB)")
    plt.ylabel("Memory (MB)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/memory_usage.png")
    plt.close()
    
    # 2. Training Speed
    plt.figure()
    ax = sns.barplot(data=best_df, x="Experiment", y="Samples/Sec", palette="mako")
    ax.bar_label(ax.containers[0], fmt='%.1f')
    plt.title("Training Throughput (Samples/Sec)")
    plt.ylabel("Samples / Second")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/throughput.png")
    plt.close()
    
    print(f"Saved metrics plots to {OUTPUT_DIR}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data()
    if df is not None:
        plot_params_vs_loss(df)
        plot_bar_metrics(df)
        print("✅ Plots generated successfully!")

if __name__ == "__main__":
    main()
