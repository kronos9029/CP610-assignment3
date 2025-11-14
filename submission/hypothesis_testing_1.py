from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


## Read the exported CSV, drop any summary row, and keep only real customers."""
def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"Customer ID", "Customer Age Value", "Total Spent Sum", "Age Group"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {', '.join(sorted(missing))}")
    df = df.dropna(subset=["Age Group"]).copy()
    return df

## Print the headcount, average spend, and spread for each age group to compare to Excel.
def summarize_groups(df: pd.DataFrame) -> pd.DataFrame:
    # Calculate group-level summary stats
    summary = df.groupby("Age Group", observed=False)["Total Spent Sum"].agg( 
        ["count", "mean", "std", "var"]
    )
    print("Group summary (counts/means/std):")
    print(summary.to_string(float_format=lambda x: f"{x:,.2f}"))
    return summary

## Calculate the full ANOVA table (sums of squares, degrees of freedom, mean squares, F, p).
def run_anova(df: pd.DataFrame) -> dict:
    # Get group-level stats
    group_stats = df.groupby("Age Group", observed=False)["Total Spent Sum"].agg(
        ["count", "mean", "var"]
    ) 
    
    # Overall mean
    grand_mean = df["Total Spent Sum"].mean() 

    # Sum of squares between groups
    ss_between = float(
        (group_stats["count"] * (group_stats["mean"] - grand_mean) ** 2).sum()
    ) 
    # Sum of squares within groups
    ss_within = float(((group_stats["count"] - 1) * group_stats["var"]).sum()) 
    # Total sum of squares
    ss_total = ss_between + ss_within 

    # number of groups
    k = group_stats.shape[0] 
    # total number of observations
    n_total = int(group_stats["count"].sum()) 
    # degrees of freedom between groups
    df_between = k - 1 
    # degrees of freedom within groups
    df_within = n_total - k 
    # total degrees of freedom
    df_total = n_total - 1 

    # mean square between groups
    ms_between = ss_between / df_between 
    # mean square within groups
    ms_within = ss_within / df_within

    # F statistic and p-value
    f_stat = ms_between / ms_within
    p_val = stats.f.sf(f_stat, df_between, df_within)

    # Compile results into a dictionary
    results = {
        "SS Between": ss_between,
        "SS Within": ss_within,
        "SS Total": ss_total,
        "df Between": df_between,
        "df Within": df_within,
        "df Total": df_total,
        "MS Between": ms_between,
        "MS Within": ms_within,
        "F": f_stat,
        "p-value": p_val,
    }

    # Print the ANOVA table
    print("\nANOVA table (Python):")
    print(f"Between Groups: SS = {ss_between:,.2f}, df = {df_between}, MS = {ms_between:,.2f}")
    print(f"Within Groups:  SS = {ss_within:,.2f}, df = {df_within}, MS = {ms_within:,.2f}")
    print(f"Total:          SS = {ss_total:,.2f}, df = {df_total}")
    print(f"F = {f_stat:.4f}, p-value = {p_val:.4e}")

    return results


## Create a boxplot of total spend by age group and save it."""
def plot_boxplot(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4)) # Create boxplot
    df.boxplot(column="Total Spent Sum", by="Age Group", ax=ax, grid=False)
    ax.set_title("Total Spend by Age Group") # Set titles and labels
    ax.set_xlabel("Age Group") # Set x-axis label
    ax.set_ylabel("Total Spent ($)") # Set y-axis label
    fig.suptitle("") # Remove default title
    fig.tight_layout() # Adjust layout
    path = output_dir / "hyp1_boxplot.png" # Save figure
    fig.savefig(path, dpi=150) # Save figure
    plt.close(fig) # Close figure to free memory


## Draw QQ plots (normal probability plots) for each age group."""
def plot_qq(df: pd.DataFrame, output_dir: Path) -> None:
    groups = df["Age Group"].unique() # Get unique age groups
    n = len(groups) # Number of age groups
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4)) # Create subplots
    # Ensure axes is iterable
    if n == 1: 
        axes = [axes]
    for ax, group in zip(axes, groups): # Loop through each group
        subset = df[df["Age Group"] == group]["Total Spent Sum"].sort_values() # Subset and sort data
        n_obs = subset.size # Number of observations
        probs = (np.arange(1, n_obs + 1) - 0.5) / n_obs # Calculate plotting positions
        z_expected = stats.norm.ppf(probs) # Expected normal quantiles

        ax.plot(z_expected, subset, marker="o", markersize=2, linestyle="-", color="C0") # Plot actual vs expected
        # Reference line from min to max
        ax.plot([z_expected[0], z_expected[-1]], [subset.iloc[0], subset.iloc[-1]], color="orange")
        ax.set_title(f"{group} QQ Plot") # Set title
        ax.set_xlabel("Expected Normal Z") # Set x-axis label
        ax.set_ylabel("Actual Spend ($)") # Set y-axis label
    fig.tight_layout() # Adjust layout
    path = output_dir / "hyp1_qq.png" # Save figure
    fig.savefig(path, dpi=150) # Save figure
    plt.close(fig) # Close figure to free memory


## Load data, show summary stats, and run ANOVA.
def main() -> None:
    base_dir = Path(__file__).parent
    csv_path = base_dir / './datasets/hyp1_data.csv' # Path to CSV file
    df = load_data(csv_path) # Load data
    summarize_groups(df) # Summarize groups
    run_anova(df) # Calculate ANOVA
    output_dir = csv_path.parent # Output directory
    plot_boxplot(df, output_dir) # Create boxplot
    plot_qq(df, output_dir) # Create QQ plots


if __name__ == "__main__":
    main()
