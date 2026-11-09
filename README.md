Domain: E-commerce

# Flipkart E-commerce Decision Intelligence — IIT Mandi Capstone

**Status:** Part 1 complete and verified. Parts 2–4 are not yet implemented. **Do not submit this repository as the finished capstone until all four parts and required output files are present.**

## Scope and data provenance

This project uses **synthetic data only**; it does not contain real Flipkart customer data, and the computed values do not describe Flipkart's actual operations. The business questions concern order-value planning, customer churn, shopper segmentation, and potential follow-on models.

The data generator is copied verbatim from the supplied **Flipkart E-commerce Decision Intelligence** brief. It produces 800 rows, 10 columns, and 170 churn-positive rows. The expected dataset MD5 is `c6f249ba693f1ebeed318d6d5e18de23`.

The seven predictors are X1 `customer_tenure_months`, X2 `complaint_count_90d`, X3 `discount_order_share`, X4 `avg_order_value_inr_hundreds`, X5 `loyalty_score`, X6 `cod_flag`, and X7 `plus_member_flag`. The outcome is `churned` and the value analyzed in Part 1 is `order_value_inr_hundreds` (one unit equals INR 100).

## Files and setup

- `generate_data.py`: supplied data generator, unchanged.
- `data/records.csv`: generator's committed output.
- `part1.py`: from-scratch computations for Tasks 1.1–1.7.
- `outputs/part1_results.txt`: 47 computed result keys, rounded to four decimals where applicable.
- `part2.py`, `part3.py`, `part4.py`: reserved scripts for future implementation; currently they fail explicitly rather than inventing output.
- `requirements.txt`: exact pinned NumPy version used for this run.

### Reproduce the generated data and Part 1 output

Use Python 3.10 or later from the repository root:

```bash
python -m pip install -r requirements.txt
python generate_data.py
python part1.py
```

The generator should print:

```text
800 10 170 c6f249ba693f1ebeed318d6d5e18de23
```

The script overwrites `outputs/part1_results.txt` with computed values. Only Python standard library and NumPy are needed; no network, dataset download, API key, or account-gated service is required to compute the results. No model-fitting library is used to calculate graded results. Every result comes from the scripts, never from a typed checkpoint.

## Part 1 findings

### R1.1 — Planning assumption and Type I error

The null hypothesis is that the true mean order value is INR 800 (8.0 hundreds of INR), tested against a two-sided alternative at the 5% significance level. We **reject H0**, since `z_stat = 2.8930` is above the critical magnitude `1.96`. The estimated mean order value is `8.9382` hundreds of INR, with 95% CI `[8.3026, 9.5738]` hundreds of INR. The difference from the assumption is approximately **0.9382 hundreds of INR**, or **INR 93.82** (about 11.7% above INR 800). This is potentially meaningful for category revenue and budget planning, but it would be necessary to compare it with actual operational margins before deciding that the difference is economically important.

A **Type I error** here would mean concluding that Flipkart's true mean order value differs from INR 800 when in fact the planning assumption is correct.

### R1.2 — Sample requirement

The computed target is `n_required = 1293`, compared with **800** available records. Thus the existing sample is **493 records short** of the size required to attain a 95% interval with half-width at most 0.5 hundreds of INR (INR 50), under the assumed standard deviation and sampling formula. The current data can estimate the mean, but not at the team's requested precision.

### R1.3 — Gradient descent convergence

The two original-scale slope estimates agree to printed precision: `gd_slope_gap = 0.0000`. Batch gradient descent converged to the closed-form ordinary least-squares line because the squared-error objective for this one-predictor linear model is convex and the specified step size and 500 updates allow it to approach the unique optimum closely.

### R1.4 — Heteroskedastic residuals and log transform

The raw fitted-value halves have a residual standard-deviation ratio of `resid_sd_ratio_raw = 3.7525`, meaning the upper half has substantially wider errors: a marked **funnel**. The log-log model reduces the corresponding ratio to `resid_sd_ratio_log = 1.0396`, approximately one, so the transformation largely removes the uneven residual spread under this diagnostic. The raw and log regression R² values are measured on different response scales and should not be compared directly.

### R1.5 — Multiple regression and extra predictors

Using the mechanical criterion **absolute coefficient greater than 0.05**, the qualifying nonconstant predictors in Task 1.7 are:

- `ln(X4)` (logged historical average order value): coefficient `1.1576`.
- `X2` (complaint count over 90 days): coefficient `0.0940`.
- `X3` (share of discounted orders): coefficient `0.1067`.
- `X6` (cash-on-delivery flag): coefficient `0.1518`.

The full model has `mr_adj_r2 = 0.7403`, compared with `loglog_adj_r2 = 0.6561` for the log-log baseline. The additional predictors therefore improve adjusted fit on the **same logged target**, enough to justify their inclusion as an explanatory baseline. Because the predictors have different units, the 0.05 threshold is not a meaningful ranking of real-world importance or statistical significance.

## Part 2 findings

Not yet implemented. The next step is a 560/240 stratified train/test split, logistic regression from scratch, threshold assessment, KNN cross-validation, and cost-sensitive decision analysis. The test rows must not affect training, standardization, K selection, or the alert-threshold choice.

## Part 3 findings

Not yet implemented. Planned work covers K-means, cluster profiles, three hierarchical linkage methods, and PCA, following the exact prescribed conventions.

## Part 4 findings

Not yet implemented. Planned work covers the model comparisons, ensembles/boosting, and neural-network architecture calculations specified in the brief.

## Submission checklist

- Domain selection in Question 1 must be exactly **`E-commerce`** (with its original spelling and hyphen).
- The README's first line must be `Domain: E-commerce`.
- Finish and verify Parts 2–4 and all 11 required output files before submitting.
- All work must be present in **one public GitHub repository**, on `main`, and the single submission link must resolve to `main`.
- Keep only text/code/CSV deliverables in the submitted repository. No image, screenshot, PDF, video, or external artifact is graded.
- Before submission, confirm that running the scripts on a clean clone regenerates the committed output files, and be prepared to explain the implementation.
- **Deadline:** 16 November 2026, 11:59 PM (per course instructions; check the platform's closing time).
