"""Part 1: reproducible inference and regression, implemented from formulas.

Run from the repository root after generate_data.py. Only NumPy's array
arithmetic and linear solve are used; no fitting or statistics packages.
"""
from pathlib import Path
import csv
import math
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "records.csv"
OUTPUT = ROOT / "outputs" / "part1_results.txt"


def sample_sd(a):
    """Sample standard deviation (n-1 divisor)."""
    a = np.asarray(a, dtype=float)
    return math.sqrt(float(np.sum((a - np.mean(a)) ** 2)) / (len(a) - 1))


def fit_line(x, y):
    """Closed-form OLS: covariance / variance, both using n-1."""
    xbar, ybar = float(np.mean(x)), float(np.mean(y))
    centered_x = x - xbar
    denominator = len(x) - 1
    cov = float(np.sum(centered_x * (y - ybar)) / denominator)
    var = float(np.sum(centered_x ** 2) / denominator)
    slope = cov / var
    intercept = ybar - slope * xbar
    pred = intercept + slope * x
    return intercept, slope, pred


def goodness(y, pred):
    tss = float(np.sum((y - np.mean(y)) ** 2))
    ess = float(np.sum((pred - np.mean(y)) ** 2))
    rss = float(np.sum((y - pred) ** 2))
    return tss, ess, rss, 1.0 - rss / tss, math.sqrt(rss / len(y)), float(np.mean(np.abs(y - pred)))


def residual_halves(pred, y):
    """Ascending fitted values; preserve source row order for ties."""
    order = np.argsort(pred, kind="stable")
    mid = len(order) // 2
    residuals = y - pred
    lower_sd = sample_sd(residuals[order[:mid]])
    upper_sd = sample_sd(residuals[order[mid:]])
    return lower_sd, upper_sd, upper_sd / lower_sd


def run():
    with DATA.open(newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    assert len(records) == 800, "Unexpected dataset; run generate_data.py"
    names = [
        "customer_tenure_months", "complaint_count_90d", "discount_order_share",
        "avg_order_value_inr_hundreds", "loyalty_score", "cod_flag", "plus_member_flag"
    ]
    X = np.array([[float(row[name]) for name in names] for row in records], dtype=float)
    y = np.array([float(row["order_value_inr_hundreds"]) for row in records])
    x4 = X[:, 3]
    n = len(y)
    result = {}

    # 1.1: 95% confidence interval, using the normal critical value.
    mean_y = float(np.mean(y))
    sd_y = sample_sd(y)
    se = sd_y / math.sqrt(n)
    result.update(n=n, mean_value=mean_y, sd_value=sd_y, se_value=se,
                  ci95_low=mean_y-1.96*se, ci95_high=mean_y+1.96*se)

    # 1.2: two-sided planning assumption test (8 hundreds of INR).
    z_stat = (mean_y - 8.0) / se
    result.update(z_stat=z_stat, reject_h0=abs(z_stat) > 1.96)

    # 1.3: required N for a half-unit confidence-interval margin.
    result["n_required"] = math.ceil((1.96 * sd_y / 0.5) ** 2)

    # 1.4: simple linear regression from the covariance formula.
    intercept, slope, pred = fit_line(x4, y)
    tss, ess, rss, r2, rmse, mae = goodness(y, pred)
    result.update(reg_slope=slope, reg_intercept=intercept,
                  tss=tss, ess=ess, rss=rss,
                  tss_equals_ess_plus_rss=abs(tss - ess - rss) < 1e-6*tss,
                  r2=r2, rmse=rmse, mae=mae)

    # 1.5: batch GD, 500 updates, keeping unrounded state throughout.
    xmean, xsd = float(np.mean(x4)), sample_sd(x4)
    z = (x4 - xmean) / xsd
    m, c = 0.0, 0.0
    def cost(m, c):
        return float(np.mean((y - (m*z + c)) ** 2))
    result["gd_J0"] = cost(m, c)
    for epoch in range(1, 501):
        e = y - (m*z + c)
        d_m = -2.0 * float(np.sum(z * e)) / n
        d_c = -2.0 * float(np.sum(e)) / n
        m -= 0.1 * d_m
        c -= 0.1 * d_c
        if epoch in (1, 10, 100, 500):
            result[f"gd_J{epoch}"] = cost(m, c)
    gd_slope = m / xsd
    gd_intercept = c - m*xmean/xsd
    result.update(gd_slope=gd_slope, gd_intercept=gd_intercept,
                  gd_slope_gap=abs(gd_slope - slope))

    # 1.6: check for heteroskedasticity, then use a log-log model.
    sdlo, sdhi, ratio = residual_halves(pred, y)
    result.update(resid_sd_lower_raw=sdlo, resid_sd_upper_raw=sdhi,
                  resid_sd_ratio_raw=ratio)
    log_x4 = np.log(x4)
    log_y = np.log(y)
    log_intercept, log_slope, log_pred = fit_line(log_x4, log_y)
    _, _, _, logr2, logrmse, _ = goodness(log_y, log_pred)
    lsdlo, lsdhi, lratio = residual_halves(log_pred, log_y)
    result.update(loglog_intercept=log_intercept, loglog_slope=log_slope,
                  loglog_r2=logr2, loglog_rmse=logrmse,
                  resid_sd_lower_log=lsdlo, resid_sd_upper_log=lsdhi,
                  resid_sd_ratio_log=lratio)

    # 1.7: multiple regression via normal equations (columns pinned).
    D = np.column_stack((np.ones(n), log_x4, X[:, 0], X[:, 1],
                         X[:, 2], X[:, 4], X[:, 5], X[:, 6]))
    beta = np.linalg.solve(D.T @ D, D.T @ log_y)
    fitted = D @ beta
    _, _, _, mr_r2, _, _ = goodness(log_y, fitted)
    for i, b in enumerate(beta):
        result[f"mr_beta_{i}"] = float(b)
    result.update(mr_r2=mr_r2,
                  mr_adj_r2=1.0-(1.0-mr_r2)*(n-1)/(n-7-1),
                  loglog_adj_r2=1.0-(1.0-logr2)*(n-1)/(n-1-1))

    # Exact expected keys: errors here indicate a missing task quantity.
    keys = (
        "n mean_value sd_value se_value ci95_low ci95_high z_stat reject_h0 n_required "
        "reg_slope reg_intercept tss ess rss tss_equals_ess_plus_rss r2 rmse mae "
        "gd_J0 gd_J1 gd_J10 gd_J100 gd_J500 gd_slope gd_intercept gd_slope_gap "
        "resid_sd_lower_raw resid_sd_upper_raw resid_sd_ratio_raw "
        "loglog_intercept loglog_slope loglog_r2 loglog_rmse "
        "resid_sd_lower_log resid_sd_upper_log resid_sd_ratio_log "
        "mr_beta_0 mr_beta_1 mr_beta_2 mr_beta_3 mr_beta_4 mr_beta_5 mr_beta_6 mr_beta_7 "
        "mr_r2 mr_adj_r2 loglog_adj_r2"
    ).split()
    assert set(keys) == set(result), "Missing or extra keys in Part 1 results"
    OUTPUT.parent.mkdir(exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as f:
        for key in keys:
            value = result[key]
            formatted = ("yes" if value else "no") if isinstance(value, (bool, np.bool_)) else (
                str(value) if isinstance(value, (int, np.integer)) else f"{value:.4f}")
            f.write(f"{key} = {formatted}\n")
    print(f"Wrote {OUTPUT.relative_to(ROOT)} ({len(keys)} computed keys)")
    return result


if __name__ == "__main__":
    run()
