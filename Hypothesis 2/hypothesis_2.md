# Hypothesis 2: Gender Differences in Spending

## Research Question
Is there a significant difference in average spending between male and female customers?

---

## Variables
- **Independent Variable:** Gender (categorical: Male, Female)
- **Dependent Variable:** Mean Total Spent per customer (continuous/numerical)

---

## Statistical Test
**Two-Sample t-Test (Independent Samples t-Test)**

This test is appropriate because:
- We have TWO independent groups (male and female customers)
- The groups are mutually exclusive (each customer belongs to only one group)
- We are comparing means of a continuous variable between two groups
- Observations are independent (one customer's spending doesn't affect another's)

---

## Step 1: Formulate the Hypotheses

### Null Hypothesis (H₀)
There is no difference in average spending between male and female customers.

**Mathematical notation:** μ_male = μ_female
or equivalently: μ_male - μ_female = 0

### Alternative Hypothesis (Hₐ)
There IS a difference in average spending between male and female customers.

**Mathematical notation:** μ_male ≠ μ_female
or equivalently: μ_male - μ_female ≠ 0

### Type of Test
**Two-tailed test** - We are testing for ANY difference between the groups, not specifically testing if one gender spends more than the other.

### Significance Level
**α = 0.05** (5% significance level)

---

## Step 2: Check the Assumptions

Before conducting the two-sample t-test, we must verify the following assumptions:

### Assumption 1: Independence
**Requirement:** The observations within each group and between groups must be independent.

**Verification:**
- Each customer's spending is recorded independently
- One customer's spending does not influence another customer's spending
- The samples are randomly selected or representative of the population

**Conclusion:** ✓ Assumption satisfied - observations are independent.

### Assumption 2: Normality
**Requirement:** The dependent variable (spending) should be approximately normally distributed within each group.

**Verification Methods:**
1. **Visual Check:** Create Q-Q plots for each group (male and female)
2. **Statistical Test:** Conduct Shapiro-Wilk test for normality
3. **Central Limit Theorem:** With sample size ≥ 30 in each group, can assume approximate normality

**Results:**
- Sample size for males: n₁ = 489
- Sample size for females: n₂ = 511
- Shapiro-Wilk test for males: W = 0.806629, p = 0.000000
- Shapiro-Wilk test for females: W = 0.809940, p = 0.000000

**Conclusion:** ✓ Assumption SATISFIED
- Although the Shapiro-Wilk test indicates non-normality (p < 0.05 for both groups), both samples are large (n₁ = 489, n₂ = 511, both ≥ 30)
- By the **Central Limit Theorem (CLT)**, we can assume approximate normality of the sampling distribution of means
- The t-test is robust to violations of normality with large sample sizes

### Assumption 3: Homogeneity of Variances (Equal Variances)
**Requirement:** The variances of the two groups should be approximately equal.

**Verification Methods:**
1. **Rule of Thumb:** Variance ratio should be < 4
2. **Statistical Test:** Levene's Test for equality of variances

**Results:**
- Variance for males: s₁² = 103,210,411.56
- Variance for females: s₂² = 100,492,791.42
- Variance ratio: [larger variance / smaller variance] = 1.0270
- Levene's Test: F = 0.0708, p = 0.7902

**Decision:**
✓ Use **standard two-sample t-test**
- The variance ratio (1.0270) is much less than 4 ✓
- Levene's test p-value (0.7902) ≥ 0.05, so we fail to reject the null hypothesis of equal variances
- Variances are approximately equal

**Conclusion:** ✓ Assumption SATISFIED - Variances are homogeneous; use standard two-sample t-test

---

## Step 3: Conduct the Test

### Descriptive Statistics

#### Male Customers
- Sample size: n₁ = 489
- Sample mean: x̄₁ = $12,528.31
- Sample standard deviation: s₁ = $10,159.25
- Sample variance: s₁² = $103,210,411.56

#### Female Customers
- Sample size: n₂ = 511
- Sample mean: x̄₂ = $12,542.60
- Sample standard deviation: s₂ = $10,024.61
- Sample variance: s₂² = $100,492,791.42

### Calculate Pooled Standard Deviation

The pooled standard deviation combines the variability from both groups:

**Formula:**
```
s_p = √( ((n₁-1)s₁² + (n₂-1)s₂²) / (n₁+n₂-2) )
```

**Calculation:**
```
s_p = √( ((489-1)×103,210,411.56 + (511-1)×100,492,791.42) / (489+511-2) )
s_p = √( (50,366,680,839.47 + 51,251,323,625.56) / 998 )
s_p = √( 101,618,004,465.03 / 998 )
s_p = √101,821,647.76
s_p = $10,090.67
```

### Calculate Test Statistic

**Formula:**
```
t = (x̄₁ - x̄₂) / (s_p × √(1/n₁ + 1/n₂))
```

**Calculation:**
```
t = (12,528.31 - 12,542.60) / (10,090.67 × √(1/489 + 1/511))
t = -14.29 / (10,090.67 × √0.002045 + 0.001957)
t = -14.29 / (10,090.67 × √0.004002)
t = -14.29 / (10,090.67 × 0.06326)
t = -14.29 / 638.34
t = -0.0224
```

### Degrees of Freedom
```
df = n₁ + n₂ - 2
df = 489 + 511 - 2
df = 998
```

### Critical Value
For a two-tailed test with α = 0.05 and df = 998:
- **Critical t-value:** t_critical = ±1.9623 (from t-table)

---

## Step 4: Determine the p-value and Conclude

### P-Value
**Calculated p-value:** p = 0.9821

### Decision Rule
- If p-value < α (0.05): **Reject H₀**
- If p-value ≥ α (0.05): **Fail to reject H₀**

### Statistical Decision
Since p-value = 0.9821 is **greater than** α = 0.05, we **FAIL TO REJECT** the null hypothesis.

**Additional verification:**
- The test statistic |t| = |-0.0224| = 0.0224 is much less than the critical value of 1.9623
- This confirms our decision to fail to reject H₀

### Statistical Conclusion
At the 0.05 significance level, we **fail to reject** the null hypothesis (t(998) = -0.0224, p = 0.9821).

### Interpretation in Context
There is **insufficient** evidence to conclude that there is a statistically significant difference in average spending between male and female customers.

The data does not provide sufficient evidence to conclude that gender affects customer spending behavior. The observed difference in means ($14.29) is very small relative to the variability in the data and could easily be due to random chance.

---

## Step 5: Report with Confidence Interval

### Calculate 95% Confidence Interval for Difference in Means

**Formula:**
```
CI = (x̄₁ - x̄₂) ± (t_critical × s_p × √(1/n₁ + 1/n₂))
```

**Calculation:**
```
Difference in means: x̄₁ - x̄₂ = 12,528.31 - 12,542.60 = -14.29

Standard Error: SE = s_p × √(1/n₁ + 1/n₂)
                SE = 10,090.67 × √(1/489 + 1/511)
                SE = 10,090.67 × 0.06326
                SE = 638.34

Margin of error: ME = t_critical × SE
                 ME = 1.9623 × 638.34
                 ME = 1,252.65

Lower bound: -14.29 - 1,252.65 = -1,266.95
Upper bound: -14.29 + 1,252.65 = 1,238.36
```

### 95% Confidence Interval
**CI: ($-1,266.95, $1,238.36)**

### Interpretation
We are 95% confident that the true mean difference in spending between male and female customers lies between **$-1,266.95 and $1,238.36**.

**Interpretation Notes:**
Since the confidence interval **includes zero**, this is consistent with failing to reject the null hypothesis. The difference in spending could be zero (no difference).

This means:
- Male customers could spend up to $1,266.95 **less** than female customers
- Male customers could spend up to $1,238.36 **more** than female customers
- The true difference could be **zero** (no difference at all)

The wide interval and inclusion of zero indicates high uncertainty about the true difference and strong evidence that gender does not have a meaningful effect on spending.

---

## Summary and Practical Significance

### Statistical Findings
- **Test conducted:** Standard two-sample t-test (independent samples)
- **Sample sizes:** n₁ = 489 males, n₂ = 511 females
- **Result:** **NOT significant** - No statistically significant difference found
- **Test statistic:** t(998) = -0.0224
- **P-value:** p = 0.9821 (extremely high, indicating strong evidence for H₀)
- **Effect size:** The mean difference is only $14.29 (male average is slightly lower)
  - Males: $12,528.31 average total spending
  - Females: $12,542.60 average total spending
  - Difference represents only 0.11% of the average spending

### Practical Significance

**Key Finding:** Gender does NOT appear to be a meaningful factor in customer spending behavior.

**Business Implications:**
1. **Very Small Difference:** The observed difference ($14.29) represents less than 0.11% of average spending - this is negligible from a business perspective.

2. **Statistical Support:** With an extremely high p-value (0.9821), the evidence strongly suggests no real difference exists between male and female customer spending.

3. **Similar Variability:** Both groups have similar standard deviations (~$10,000), indicating comparable spending patterns and variability.

4. **Equal Treatment Justified:** The lack of difference justifies treating male and female customers similarly in terms of marketing budgets, promotional strategies, and inventory planning.

### Recommendations

Based on these findings, we recommend:

1. **Unified Marketing Strategy:**
   - Do NOT create gender-specific marketing campaigns based solely on spending differences
   - Focus marketing differentiation on other factors (age, membership level, region, product preferences)

2. **Resource Allocation:**
   - Allocate marketing resources equally between genders
   - No need for gender-based budget adjustments

3. **Further Analysis:**
   - Investigate other demographic factors that may better explain spending differences:
     - **Age groups** - might show more significant differences
     - **Membership levels** (Standard, Gold, Platinum) - likely more predictive
     - **Geographic region** - could reveal location-based patterns
     - **Product categories** - while total spending is similar, product preferences might differ

4. **Product Preference Analysis:**
   - Although total spending is equal, conduct analysis of **which products** each gender purchases
   - This could reveal opportunities for targeted product recommendations without changing spending targets

5. **Validation:**
   - Monitor this relationship over time to ensure it remains stable
   - Consider testing if gender interacts with other variables (e.g., does gender matter within certain age groups?)

---

## Implementation Notes

### Python Implementation

See the complete Python implementation in `hypothesis_testing_2.ipynb` and the analysis script `calculate_hypothesis2_stats.py`.

**Data Sources:**
- Customer demographics: `datasets/Customers_v4.csv`
- Sales transactions: `datasets/Sales_Cleaned.csv`

**Key Steps:**
```python
# 1. Load customer data with gender information
df_customers = pd.read_csv('datasets/Customers_v4.csv')

# 2. Load sales data (handle European decimal format with comma separator)
df_sales = pd.read_csv('datasets/Sales_Cleaned.csv', sep=';')
df_sales['Total Spent'] = df_sales['Total Spent'].str.replace(',', '.').astype(float)

# 3. Merge to get gender for each transaction
df_merged = df_sales.merge(df_customers[['Customer ID', 'Gender']], on='Customer ID')

# 4. Aggregate by customer (sum total spending per customer)
customer_spending = df_merged.groupby(['Customer ID', 'Gender'])['Total Spent'].sum()

# 5. Separate by gender
male_spending = customer_spending[customer_spending['Gender'] == 'Male']
female_spending = customer_spending[customer_spending['Gender'] == 'Female']

# 6. Check assumptions (Shapiro-Wilk, Levene's test)
shapiro_male = stats.shapiro(male_spending)
shapiro_female = stats.shapiro(female_spending)
levene_stat, levene_p = stats.levene(male_spending, female_spending)

# 7. Conduct t-test
t_stat, p_value = stats.ttest_ind(male_spending, female_spending, equal_var=True)

# 8. Calculate confidence interval
# [See complete implementation in calculate_hypothesis2_stats.py]
```

### Excel Implementation
**Steps:**
1. **Data Preparation:**
   - Import both 'Customers_v4.csv' and 'Sales_Cleaned.csv' into Excel
   - Use VLOOKUP or Power Query to merge Gender from Customers into Sales data
   - Create a pivot table or use SUMIF to aggregate Total Spent by Customer ID and Gender

2. **Create Separate Columns:**
   - Column A: Male customer total spending (489 values)
   - Column B: Female customer total spending (511 values)

3. **Descriptive Statistics:**
   - Use Data → Data Analysis → Descriptive Statistics for each column
   - Calculate n, mean, standard deviation, variance for both groups

4. **Assumption Tests:**
   - Conduct F-test for variance equality or manually calculate variance ratio
   - Visually inspect distributions with histograms

5. **t-Test:**
   - Data Analysis Toolpak → t-Test: Two-Sample Assuming Equal Variances
   - Set α = 0.05
   - Input ranges for male and female spending

6. **Confidence Interval:**
   - Calculate using formula: =mean_diff ± t_critical * standard_error
   - Use T.INV.2T(0.05, 998) for critical value

7. **Document Results:**
   - Create a summary table with all statistics
   - Write interpretation of results

---

## References
- Lecture: Two-Sample t-Test
- Deliverable: CP610 Project Deliverable #3
- Dataset: Sales_Cleaned.csv (25,000 transactions) merged with Customers_v4.csv (1,000 customers)
- Analysis Date: 2025-11-06
