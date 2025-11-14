from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def load_data(customer_path: Path, sales_path: Path) -> pd.DataFrame:
    """Load and prepare customer and sales data for gender analysis."""
    # Load customer data
    df_customers = pd.read_csv(customer_path)
    print(f"Customers loaded: {len(df_customers)} customers")
    print(f"Gender distribution: {df_customers['Gender'].value_counts().to_dict()}")

    # Load sales data
    df_sales = pd.read_csv(sales_path, sep=';')

    # Clean up column names
    df_sales.columns = df_sales.columns.str.strip()
    df_customers.columns = df_customers.columns.str.strip()

    # Convert 'Total Spent' from currency format to numeric
    df_sales['Total Spent'] = df_sales['Total Spent'].str.replace(',', '.').astype(float)

    print("\nCustomer Dataset Info:")
    print(f"Columns: {list(df_customers.columns)}")
    print(df_sales[['Customer ID', 'Total Spent']].head(3))

    # Data quality checks
    print("\n1. Missing Values in Sales Data:")
    missing_sales = df_sales[['Customer ID', 'Total Spent']].isnull().sum()
    print(missing_sales)

    print("\n2. Missing Values in Customer Data:")
    missing_customers = df_customers[['Customer ID', 'Gender']].isnull().sum()
    print(missing_customers)

    print("\n3. Gender Column Values:")
    print(df_customers['Gender'].value_counts())
    print(f"\nUnique genders: {df_customers['Gender'].unique()}")

    print("\n4. Customer ID Matching:")
    sales_customers = set(df_sales['Customer ID'].unique())
    demographic_customers = set(df_customers['Customer ID'].unique())
    print(f"Customers in sales data: {len(sales_customers)}")
    print(f"Customers in demographics data: {len(demographic_customers)}")
    print(f"Customers in both: {len(sales_customers & demographic_customers)}")

    # Merge to get gender for each transaction
    df_merged = df_sales.merge(df_customers[['Customer ID', 'Gender']],
                                on='Customer ID',
                                how='inner')

    # Group by Customer and Gender to get total spending per customer
    customer_spending = df_merged.groupby(['Customer ID', 'Gender'])['Total Spent'].sum().reset_index()
    customer_spending.columns = ['CustomerID', 'Gender', 'Total_Spent_Per_Customer']

    print(f"\nTotal unique customers: {customer_spending.shape[0]}")
    print(f"Gender distribution:")
    print(customer_spending['Gender'].value_counts())
    print(customer_spending.head(10))

    return customer_spending


def summarize_groups(df: pd.DataFrame) -> tuple:
    """Print descriptive statistics for each gender group."""
    # Extract spending data for each gender
    male_spending = df[df['Gender'] == 'Male']['Total_Spent_Per_Customer']
    female_spending = df[df['Gender'] == 'Female']['Total_Spent_Per_Customer']

    print("\nGENDER GROUP STATISTICS:")
    print(f"\nMale Customers:")
    print(f"Sample size: {len(male_spending)}")
    print(f"Mean spending: ${male_spending.mean():.2f}")
    print(f"Median spending: ${male_spending.median():.2f}")
    print(f"Std deviation: ${male_spending.std():.2f}")
    print(f"Min spending: ${male_spending.min():.2f}")
    print(f"Max spending: ${male_spending.max():.2f}")

    print(f"\nFemale Customers:")
    print(f"Sample size: {len(female_spending)}")
    print(f"Mean spending: ${female_spending.mean():.2f}")
    print(f"Median spending: ${female_spending.median():.2f}")
    print(f"Std deviation: ${female_spending.std():.2f}")
    print(f"Min spending: ${female_spending.min():.2f}")
    print(f"Max spending: ${female_spending.max():.2f}")

    print(f"\nObservation:")
    print(f"- Difference in means: ${abs(male_spending.mean() - female_spending.mean()):.2f}")
    print(f"- Ratio of standard deviations: {max(male_spending.std(), female_spending.std()) / min(male_spending.std(), female_spending.std()):.2f}")

    return male_spending, female_spending


def print_hypotheses() -> None:
    """Print the hypothesis statements for the test."""
    print("\n" + "=" * 80)
    print("STEP 1: FORMULATE THE HYPOTHESES")
    print("=" * 80)
    print("\nNull Hypothesis (H0):")
    print("  There is no difference in average spending between male and female customers.")
    print("  Mathematical notation: μ_male = μ_female")

    print("\nAlternative Hypothesis (H1):")
    print("  There is a difference in average spending between male and female customers.")
    print("  Mathematical notation: μ_male ≠ μ_female")

    print("\nType of Test: TWO-TAILED TEST")
    print("  WHY: We're testing for ANY difference, not a specific direction")


def check_independence(df: pd.DataFrame) -> None:
    """Check independence assumption by verifying unique customers."""
    print("STEP 2: CHECK THE ASSUMPTIONS")
    print("\n### ASSUMPTION 1: INDEPENDENCE ###")


    # Check: Verify no duplicate customers
    n_customers = len(df)
    n_unique_customers = df['CustomerID'].nunique()
    print(f"\nTotal customer records: {n_customers}")
    print(f"Unique customers: {n_unique_customers}")

    if n_customers == n_unique_customers:
        print("ASSUMPTION SATISFIED: All customers are unique")
    else:
        print("WARNING: Duplicate customers found - needs investigation")


def check_normality(male_spending: pd.Series, female_spending: pd.Series, output_dir: Path) -> bool:
    """Assess normality using visual and statistical methods."""

    # Method 1: Visual inspection with histograms and Q-Q plots
    print("\nMethod 1: Visual Inspection")
    print("Creating histograms and Q-Q plots for each gender group...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Histogram for Males
    axes[0, 0].hist(male_spending, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0, 0].set_title('Distribution of Spending - Male Customers', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Total Spent per Customer ($)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(male_spending.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: ${male_spending.mean():.2f}')
    axes[0, 0].axvline(male_spending.median(), color='green', linestyle='--', linewidth=2,
                       label=f'Median: ${male_spending.median():.2f}')
    axes[0, 0].legend()

    # Q-Q plot for Males
    stats.probplot(male_spending, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot - Male Customers', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # Histogram for Females
    axes[1, 0].hist(female_spending, bins=30, edgecolor='black', alpha=0.7, color='lightcoral')
    axes[1, 0].set_title('Distribution of Spending - Female Customers', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Total Spent per Customer ($)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].axvline(female_spending.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: ${female_spending.mean():.2f}')
    axes[1, 0].axvline(female_spending.median(), color='green', linestyle='--', linewidth=2,
                       label=f'Median: ${female_spending.median():.2f}')
    axes[1, 0].legend()

    # Q-Q plot for Females
    stats.probplot(female_spending, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot - Female Customers', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / "hyp2_normality_plots.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    print("INTERPRETATION OF Q-Q PLOTS:")
    print("- If points fall approximately on the diagonal line → data is normally distributed")
    print("- Deviations from the line indicate departures from normality")

    # Method 2: Shapiro-Wilk Test for Normality
    print("\nMethod 2: Shapiro-Wilk Test")

    # Test for males
    stat_male, p_male = stats.shapiro(male_spending)
    print(f"\nMale Customers:")
    print(f"  Test statistic (W): {stat_male:.4f}")
    print(f"  P-value: {p_male:.4f}")
    if p_male > 0.05:
        print(f"  RESULT: Fail to reject H0 (p = {p_male:.4f} > 0.05)")
        print(f"  CONCLUSION: Male spending data appears normally distributed")
    else:
        print(f"  RESULT: Reject H0 (p = {p_male:.4f} < 0.05)")
        print(f"  CONCLUSION: Male spending data deviates from normality")

    # Test for females
    stat_female, p_female = stats.shapiro(female_spending)
    print(f"\nFemale Customers:")
    print(f"  Test statistic (W): {stat_female:.4f}")
    print(f"  P-value: {p_female:.4f}")
    if p_female > 0.05:
        print(f"  RESULT: Fail to reject H0 (p = {p_female:.4f} > 0.05)")
        print(f"  CONCLUSION: Female spending data appears normally distributed")
    else:
        print(f"  RESULT: Reject H0 (p = {p_female:.4f} < 0.05)")
        print(f"  CONCLUSION: Female spending data deviates from normality")

    # Method 3: Central Limit Theorem (CLT)
    print("\nMethod 3: Central Limit Theorem (CLT)")
    print(f"\nSample sizes:")
    print(f"  Males: n₁ = {len(male_spending)}")
    print(f"  Females: n₂ = {len(female_spending)}")

    if len(male_spending) >= 30 and len(female_spending) >= 30:
        print(f"\n✓ BOTH samples have n ≥ 30")
        print(f"  CONCLUSION: Can invoke CLT - t-test is robust to normality violations")
        return True
    elif p_male > 0.05 and p_female > 0.05:
        print(f"\nShapiro-Wilk tests show normality")
        print(f"  CONCLUSION: Normality assumption is satisfied")
        return True
    else:
        print(f"\n✗ Consider non-parametric alternative (Mann-Whitney U test)")
        return False


def check_variance_homogeneity(male_spending: pd.Series, female_spending: pd.Series) -> bool:
    """Check homogeneity of variances using variance ratio and Levene's test."""
    print("\n### ASSUMPTION 3: HOMOGENEITY OF VARIANCES ###")
    print("\nMethod 1: Variance Ratio Rule of Thumb")

    var_male = male_spending.var()
    var_female = female_spending.var()
    variance_ratio = max(var_male, var_female) / min(var_male, var_female)

    print(f"Male variance: {var_male:,.2f}")
    print(f"Female variance: {var_female:,.2f}")
    print(f"Variance ratio: {variance_ratio:.2f}")

    if variance_ratio < 4:
        print(f"Ratio ({variance_ratio:.2f}) < 4: Variances can be considered equal")
        equal_var_rule = True
    else:
        print(f"Ratio ({variance_ratio:.2f}) ≥ 4: Variances may not be equal")
        equal_var_rule = False

    # Method 2: Levene's Test for Equality of Variances
    print("\nMethod 2: Levene's Test")

    # Perform Levene's test
    levene_stat, levene_p = stats.levene(male_spending, female_spending)

    print(f"Test statistic: {levene_stat:.4f}")
    print(f"P-value: {levene_p:.4f}")

    if levene_p > 0.05:
        print(f"RESULT: Fail to reject H0 (p = {levene_p:.4f} > 0.05)")
        print(f"CONCLUSION: Variances are approximately equal")
        equal_var_levene = True
    else:
        print(f"RESULT: Reject H0 (p = {levene_p:.4f} < 0.05)")
        print(f"CONCLUSION: Variances are significantly different")
        equal_var_levene = False

    # FINAL DECISION on variance equality
    print("\nFINAL DECISION ON EQUAL VARIANCES:")
    if equal_var_rule and equal_var_levene:
        print("  ✓ Both tests indicate equal variances")
        print("  DECISION: Use standard two-sample t-test with equal_var=True")
        return True
    elif equal_var_rule or equal_var_levene:
        print("Mixed results on variance equality")
        print("DECISION: Use Welch's t-test (equal_var=False) to be conservative")
        return False
    else:
        print("Both tests indicate unequal variances")
        print("DECISION: Use Welch's t-test (equal_var=False)")
        return False


def run_ttest(male_spending: pd.Series, female_spending: pd.Series, use_equal_var: bool) -> dict:
    """Conduct two-sample t-test and return results."""
    print("STEP 3: CONDUCT THE TEST")

    # Calculate descriptive statistics
    n1 = len(male_spending)
    n2 = len(female_spending)
    mean1 = male_spending.mean()
    mean2 = female_spending.mean()
    std1 = male_spending.std(ddof=1)
    std2 = female_spending.std(ddof=1)

    print("\nDESCRIPTIVE STATISTICS:")
    print(f"\nMale Customers (Group 1):")
    print(f"  Sample size (n₁): {n1}")
    print(f"  Sample mean (x̄₁): ${mean1:.2f}")
    print(f"  Sample std dev (s₁): ${std1:.2f}")
    print(f"  Sample variance (s₁²): {std1**2:,.2f}")

    print(f"\nFemale Customers (Group 2):")
    print(f"  Sample size (n₂): {n2}")
    print(f"  Sample mean (x̄₂): ${mean2:.2f}")
    print(f"  Sample std dev (s₂): ${std2:.2f}")
    print(f"  Sample variance (s₂²): {std2**2:,.2f}")

    print(f"\nDifference in means (x̄₁ - x̄₂): ${mean1 - mean2:.2f}")

    # Test parameters
    print("\nPERFORMING TWO-SAMPLE T-TEST:")
    print(f"\nTest parameters:")
    print(f"  Equal variances assumed: {use_equal_var}")
    if use_equal_var:
        print(f"  Using: Standard Student's t-test")
    else:
        print(f"  Using: Welch's t-test (does not assume equal variances)")
    print(f"  Significance level (α): 0.05")
    print(f"  Type of test: Two-tailed")

    # Perform the t-test
    t_statistic, p_value = stats.ttest_ind(male_spending, female_spending,
                                            equal_var=use_equal_var)

    print(f"\nTEST RESULTS:")
    print(f"  t-statistic: {t_statistic:.4f}")
    print(f"  p-value: {p_value:.4f}")

    # Calculate degrees of freedom
    if use_equal_var:
        df = n1 + n2 - 2
        print(f"  Degrees of freedom: {df}")
    else:
        numerator = (std1**2/n1 + std2**2/n2)**2
        denominator = (std1**2/n1)**2/(n1-1) + (std2**2/n2)**2/(n2-1)
        df = numerator / denominator
        print(f"  Degrees of freedom (Welch): {df:.2f}")

    # Get critical value for two-tailed test
    alpha = 0.05
    critical_value = stats.t.ppf(1 - alpha/2, df)
    print(f"  Critical t-value (±): ±{critical_value:.4f}")

    # Compile results
    results = {
        "n1": n1,
        "n2": n2,
        "mean1": mean1,
        "mean2": mean2,
        "std1": std1,
        "std2": std2,
        "t_statistic": t_statistic,
        "p_value": p_value,
        "df": df,
        "critical_value": critical_value,
        "alpha": alpha,
        "use_equal_var": use_equal_var,
    }

    return results


def interpret_results(results: dict) -> str:
    """Make statistical decision and interpret results."""
    print("STEP 4: DETERMINE P-VALUE AND CONCLUDE")

    alpha = results["alpha"]
    p_value = results["p_value"]
    t_statistic = results["t_statistic"]
    critical_value = results["critical_value"]
    df = results["df"]
    mean1 = results["mean1"]
    mean2 = results["mean2"]
    use_equal_var = results["use_equal_var"]

    # DECISION RULE
    print("\nDECISION RULE:")
    print(f"Significance level (α): {alpha}")
    print(f"Decision criteria:")
    print(f"    - If p-value < {alpha}: REJECT H0")
    print(f"    - If p-value >= {alpha}: FAIL TO REJECT H0")
    print(f"\n  Alternative decision criteria (using t-statistic):")
    print(f"    - If |t| > {critical_value:.4f}: REJECT H0")
    print(f"    - If |t| <= {critical_value:.4f}: FAIL TO REJECT H0")

    print("\nSTATISTICAL DECISION:")
    print(f"\nTest results summary:")
    print(f"    t-statistic: {t_statistic:.4f}")
    print(f"    p-value: {p_value:.4f}")
    print(f"    Critical value: ±{critical_value:.4f}")
    print(f"    Degrees of freedom: {df if use_equal_var else f'{df:.2f}'}")

    # Make decision
    if p_value < alpha:
        decision = "REJECT"
        print(f"\n  DECISION: REJECT H0")
        print(f"    WHY: p-value ({p_value:.4f}) < α ({alpha})")
        print(f"    AND: |t| = {abs(t_statistic):.4f} > {critical_value:.4f}")
    else:
        decision = "FAIL TO REJECT"
        print(f"\n  DECISION: FAIL TO REJECT H0")
        print(f"    WHY: p-value ({p_value:.4f}) ≥ α ({alpha})")
        print(f"    AND: |t| = {abs(t_statistic):.4f} ≤ {critical_value:.4f}")

    # STATISTICAL CONCLUSION
    print("\nSTATISTICAL CONCLUSION:")

    if use_equal_var:
        df_format = f"{int(df)}"
    else:
        df_format = f"{df:.2f}"

    print(f"\n  At the {alpha} significance level, we {decision} the null hypothesis")
    print(f"  (t({df_format}) = {t_statistic:.4f}, p = {p_value:.4f}).")

    print("\nINTERPRETATION:")

    if decision == "REJECT":
        print(f"\nThere IS statistically significant evidence to conclude that there is a difference in average spending between male and female customers.")
        print(f"\n  Specifically:")
        if mean1 > mean2:
            print(f"    - Male customers spend significantly MORE than female customers")
            print(f"    - Male mean: ${mean1:.2f} vs Female mean: ${mean2:.2f}")
            print(f"    - Difference: ${mean1 - mean2:.2f}")
        else:
            print(f"    - Female customers spend significantly MORE than male customers")
            print(f"    - Female mean: ${mean2:.2f} vs Male mean: ${mean1:.2f}")
            print(f"    - Difference: ${mean2 - mean1:.2f}")
    else:
        print(f"\n  There is NOT sufficient statistical evidence to conclude that there is a difference in average spending between male and female customers.")
        print(f"\n  Specifically:")
        print(f"    - Male mean: ${mean1:.2f}")
        print(f"    - Female mean: ${mean2:.2f}")
        print(f"    - Observed difference: ${abs(mean1 - mean2):.2f}")

    return decision


def calculate_confidence_interval(results: dict, output_dir: Path) -> None:
    """Calculate and visualize confidence interval for the difference in means."""
    print("\n" + "=" * 80)
    print("STEP 5: REPORT WITH CONFIDENCE INTERVAL")
    print("=" * 80)

    confidence_level = 0.95
    alpha_ci = 1 - confidence_level

    mean1 = results["mean1"]
    mean2 = results["mean2"]
    std1 = results["std1"]
    std2 = results["std2"]
    n1 = results["n1"]
    n2 = results["n2"]
    df = results["df"]
    use_equal_var = results["use_equal_var"]


    # Calculate the difference in means
    mean_diff = mean1 - mean2
    print(f"\nObserved difference (x̄₁ - x̄₂): ${mean_diff:.2f}")

    # Calculate standard error
    if use_equal_var:
        pooled_variance = ((n1-1)*std1**2 + (n2-1)*std2**2) / (n1 + n2 - 2)
        standard_error = np.sqrt(pooled_variance * (1/n1 + 1/n2))
    else:
        standard_error = np.sqrt(std1**2/n1 + std2**2/n2)

    t_critical_ci = stats.t.ppf(1 - alpha_ci/2, df)
    print(f"Critical t-value (for {confidence_level*100:.0f}% CI): {t_critical_ci:.4f}")

    margin_of_error = t_critical_ci * standard_error
    print(f"Standard error: {standard_error:.4f}")
    print(f"Margin of error: {margin_of_error:.2f}")

    ci_lower = mean_diff - margin_of_error
    ci_upper = mean_diff + margin_of_error

    print(f"\n95% CONFIDENCE INTERVAL:")
    print(f"(${ci_lower:.2f}, ${ci_upper:.2f})")

    print(f"\nCalculation breakdown:")
    print(f"- Lower bound = {mean_diff:.2f} - {margin_of_error:.2f} = ${ci_lower:.2f}")
    print(f"- Upper bound = {mean_diff:.2f} + {margin_of_error:.2f} = ${ci_upper:.2f}")

    print("\nINTERPRETATION OF CONFIDENCE INTERVAL:")
    print(f"\n  We are {confidence_level*100:.0f}% confident that the true mean difference in spending")
    print(f"  between male and female customers lies between ${ci_lower:.2f} and ${ci_upper:.2f}.")

    if ci_lower <= 0 <= ci_upper:
        print(f"We cannot be confident there is a true difference in the population.")
    else:
        print(f"We CAN be confident there is a true difference in the population.")

        if ci_lower > 0:
            print(f"\nSince the entire interval is POSITIVE, we're confident that male customers spend MORE than female customers.")
        else:
            print(f"\nSince the entire interval is NEGATIVE, we're confident that female customers spend MORE than male customers.")

    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the confidence interval
    ax.errorbar(x=0, y=mean_diff, yerr=margin_of_error,
                fmt='o', markersize=12, capsize=10, capthick=2,
                color='steelblue', ecolor='steelblue', linewidth=2,
                label=f'{confidence_level*100:.0f}% Confidence Interval')

    # Add horizontal line at zero
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2,
               label='No Difference (H0: μ₁ - μ₂ = 0)')

    # Add labels and formatting
    ax.set_ylabel('Difference in Mean Spending (Male - Female) ($)', fontsize=12, fontweight='bold')
    ax.set_title(f'{confidence_level*100:.0f}% Confidence Interval for Difference in Means\n' +
                 f'Male vs Female Customer Spending', fontsize=14, fontweight='bold')
    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([])
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    # Add text annotations
    ax.text(0.05, mean_diff, f'  Difference: ${mean_diff:.2f}',
            verticalalignment='center', fontsize=11, fontweight='bold')
    ax.text(0.05, ci_upper + 0.05*abs(ci_upper) if ci_upper != 0 else ci_upper + 100,
            f'  Upper: ${ci_upper:.2f}',
            verticalalignment='bottom', fontsize=10, color='darkblue')
    ax.text(0.05, ci_lower - 0.05*abs(ci_lower) if ci_lower != 0 else ci_lower - 100,
            f'  Lower: ${ci_lower:.2f}',
            verticalalignment='top', fontsize=10, color='darkblue')

    # Add interpretation box
    interpretation = "CI includes 0 → No significant difference" if (ci_lower <= 0 <= ci_upper) else \
                    "CI excludes 0 → Significant difference"
    box_color = 'lightyellow' if (ci_lower <= 0 <= ci_upper) else 'lightgreen'
    ax.text(0.95, 0.95, interpretation, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.8),
            fontsize=11, fontweight='bold')

    plt.tight_layout()
    path = output_dir / "hyp2_confidence_interval.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_summary(results: dict, decision: str) -> None:
    """Print final summary report."""
    print("COMPLETE HYPOTHESIS TEST SUMMARY REPORT")

    alpha = results["alpha"]
    p_value = results["p_value"]
    t_statistic = results["t_statistic"]
    df = results["df"]
    mean1 = results["mean1"]
    mean2 = results["mean2"]
    std1 = results["std1"]
    std2 = results["std2"]
    n1 = results["n1"]
    n2 = results["n2"]
    use_equal_var = results["use_equal_var"]

    if use_equal_var:
        test_name = "Independent Samples T-Test"
        df_format = f"{int(df)}"
    else:
        test_name = "Welch's T-Test"
        df_format = f"{df:.2f}"

    print("\nRESEARCH QUESTION:")
    print("  Is there a significant difference in average spending between male and female customers?")

    print(f"Statistical Test: {test_name}")
    print(f"Significance Level: α = {alpha}")
    print(f"Sample Sizes: n₁ = {n1} (Male), n₂ = {n2} (Female)")

    print(f"\n  2. Test Results:")
    print(f"       t({df_format}) = {t_statistic:.4f}, p = {p_value:.4f}")

    print(f"\n  3. Statistical Decision:")
    print(f"       {decision} the null hypothesis at α = {alpha}")

    print("\nCONCLUSION:")

    if decision == "REJECT":
        print(f"\nThere IS statistically significant evidence (p = {p_value:.4f}) that")
        print(f"  male and female customers differ in their average spending.")
        if mean1 > mean2:
            print(f"  Male customers spend significantly more than female customers.")
        else:
            print(f"  Female customers spend significantly more than male customers.")
    else:
        print(f"\n  There is NO statistically significant evidence (p = {p_value:.4f}) that")
        print(f"  male and female customers differ in their average spending.")
        print(f"  The observed difference of ${abs(mean1 - mean2):.2f} can be attributed")
        print(f"  to random sampling variation.")


def main() -> None:
    """Main function to run the complete analysis."""
    # Define paths
    base_dir = Path(__file__).parent
    customer_path = base_dir / './datasets/Customers_v4.csv'
    sales_path = base_dir / './datasets/Sales_Cleaned.csv'
    output_dir = base_dir

    # Load and prepare data
    customer_spending = load_data(customer_path, sales_path)

    # Summarize groups and get spending data
    male_spending, female_spending = summarize_groups(customer_spending)

    # Print hypotheses
    print_hypotheses()

    # Check assumptions
    check_independence(customer_spending)
    check_normality(male_spending, female_spending, output_dir)
    use_equal_var = check_variance_homogeneity(male_spending, female_spending)

    # Run t-test
    results = run_ttest(male_spending, female_spending, use_equal_var)

    # Interpret results
    decision = interpret_results(results)

    # Calculate confidence interval
    calculate_confidence_interval(results, output_dir)

    # Print summary
    print_summary(results, decision)


if __name__ == "__main__":
    main()