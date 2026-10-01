# step2_eda.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from step1_load_data import load_data

# turn off warnings
import warnings
warnings.filterwarnings("ignore")

def main():
    df = load_data()

    print("\n► Dataset Info:")
    print(df.info())

    print("\n► Missing values:")
    print(df.isnull().sum())

    print("\n► Class distribution:")
    print(df['Class'].value_counts())
    print("\n► Class distribution (%):")
    print(df['Class'].value_counts(normalize=True) * 100)

    # Create folder for plots
    ROOT = Path(__file__).resolve().parents[1]
    plots_dir = ROOT / "eda_plots"
    plots_dir.mkdir(exist_ok=True)

    # Plot class distribution
    plt.figure(figsize=(6,4))
    sns.countplot(x="Class", data=df)
    plt.title("Class Distribution")
    plt.savefig(plots_dir / "class_distribution.png")

    # Plot amount distribution
    plt.figure(figsize=(6,4))
    sns.histplot(df["Amount"], bins=50, log_scale=True)
    plt.title("Amount Distribution (log scale)")
    plt.savefig(plots_dir / "amount_hist.png")

    print(f"\n✔️ Plots saved in: {plots_dir}")

if __name__ == "__main__":
    main()
