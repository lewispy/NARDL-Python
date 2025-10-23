# NARDL: Nonlinear ARDL helper for Python

A compact helper for estimating **Nonlinear ARDL (NARDL)** models in Python, with automatic lag selection, Pesaran–Shin–Smith style bounds tests, HAC defaults, long‑run deltas, asymmetry tests, dynamic multipliers with bootstrap bands, and convenient Excel export. Built on `statsmodels` and `pandas`. fileciteturn0file0

> Core file: `nardl.py`

---

## Features

- **UECM/NARDL estimation** with optional constant or linear trend, and IC‑based lag selection
- **Per‑exogenous lag selection** using a greedy search, or a common `q` for all exogenous levels
- **Bounds testing** that supports PSS cases I to V, with embedded asymptotic critical values for small `k` and an optional **bootstrap** alternative
- **Long‑run coefficients** via the delta method, including standard errors and p‑values
- **Short‑run ECM view** with the error‑correction term reported directly
- **Asymmetry tests** for short‑run and long‑run effects
- **Dynamic multipliers** for positive and negative shocks, with **bootstrap confidence bands**
- **Plot helpers** for single or grid plots
- **One‑shot Excel export** of all key tables, including multipliers per variable
- Sensible **HAC defaults** using a Newey–West maxlags rule of thumb

---

## Installation

This is a single‑file module. You can vendor the file or install it in your environment.

```bash
pip install pandas numpy statsmodels xlsxwriter openpyxl matplotlib
```

Then add the repo to your project or install it as a local package as you prefer.

---

## Quick start

```python
import pandas as pd
from nardl import NARDL

# Example toy data
# y is the dependent variable
# x1 and x2 are the candidate asymmetric regressors
# any other columns will be treated as controls in levels and first differences
df = pd.DataFrame({
    "y":  [1.0, 1.1, 1.15, 1.2, 1.22, 1.3, 1.28, 1.35, 1.37, 1.45],
    "x1": [1.0, 1.0, 1.1,  1.1, 1.2,  1.2,  1.3,  1.25, 1.35, 1.4 ],
    "x2": [0.9, 1.0, 1.0,  0.95, 1.05, 1.0,  1.1,  1.15, 1.1,  1.2 ]
})

# Build and fit a NARDL with trend option = "c" (constant only)
model = NARDL(
    data=df,
    dep="y",
    asym_vars=["x1", "x2"],
    max_lag_endog=3,
    max_lag_exog=3,
    trend="c",
    ic="aic",
    cov_type="HAC",
    hac_maxlags="auto",          # uses a Newey–West rule internally
    per_exog_lags=True           # greedy per‑exogenous q_k selection
).fit()

# Short‑run ECM summary and long‑run table
ecm = model.ecm_summary()
lr  = model.longrun_summary()
print(ecm)
print(lr)

# Bounds test: choose case based on deterministic terms
bt = model.bounds_test(case=None, alpha=0.05)  # case inferred from trend
print(bt)

# Bootstrap version of the bounds test
bt_star = model.bounds_bootstrap(B=499, case=None, alpha=0.05)
print(bt_star)

# Asymmetry tests
asy = model.asymmetry_tests()
print(asy)

# Diagnostics
diag = model.diagnostics(bg_lags=None, arch_lags=4)
print(diag)

# Dynamic multipliers and confidence bands
tab = model.dynamic_multipliers_table("x1", horizon=40)
bands_pos = model.multiplier_bands("x1", horizon=40, shock="pos", nboot=999, level=0.95)

# Plot helpers (matplotlib UI)
model.plot_dynamic_multiplier("x1", horizon=40, bands=True, nboot=499)
model.plot_all_multipliers(horizon=40, bands=True, nboot=499)
```

---

## API overview

```python
NARDL(
  data: pd.DataFrame,
  dep: str,
  asym_vars: list[str],
  max_lag_endog: int = 3,
  max_lag_exog: int = 3,
  trend: str = "c",          # "n" (none), "c" (const), "ct" (const + trend)
  ic: str = "aic",           # or "bic"
  cov_type: str = "HAC",     # any statsmodels OLS cov_type works
  cov_kwds: dict | None = None,
  lag_select_grid: tuple[list[int], list[int]] | None = None,
  per_exog_lags: bool = True,
  exog_lags: dict[str, int] | None = None,   # override per‑var q_k
  dropna: bool = True,
  seed: int | None = None,
  hac_maxlags: int | str | None = "auto"
)
```

Main methods:

- `fit(p: int | None = None, q: int | None = None)`  
  Fit the UECM. If `per_exog_lags=True`, the model runs a greedy search for `q_k` per exogenous variable. If `exog_lags` is provided, it is respected. Otherwise a common‑lags grid search runs over `p` and `q` using AIC or BIC.

- `ecm_summary() -> pd.DataFrame`  
  Returns the short‑run ECM, including the error‑correction term at lag 1.

- `longrun_summary() -> pd.DataFrame`  
  Long‑run coefficients computed with the delta method.

- `bounds_test(case: str | int | None = None, alpha=0.05, use_pesaran=True)`  
  F‑test for joint significance of lagged levels. Supports PSS cases I to V. Case defaults follow the trend: `n → I`, `c → III`, `ct → V`.

- `bounds_bootstrap(B=999, case=None, alpha=0.05)`  
  Bootstrap distribution of the F‑statistic under the null, with empirical quantiles and p‑value.

- `asymmetry_tests()`  
  Wald tests for long‑run and short‑run asymmetry for each variable in `asym_vars`.

- `diagnostics(bg_lags=None, arch_lags=4)`  
  Breusch–Godfrey LM test for serial correlation and ARCH LM test for conditional heteroskedasticity, with R² and adjusted R².

- `dynamic_multiplier(var, horizon=40, shock="pos")` and `dynamic_multipliers_table(var, horizon=40)`  
  Cumulated responses for +1 and −1 unit changes in the positive and negative partial‑sum processes.

- `multiplier_bands(var, horizon=40, shock="pos", nboot=999, level=0.95)`  
  Bootstrap bands for the multiplier path using the estimated parameter covariance.

- `plot_dynamic_multiplier(...)` and `plot_all_multipliers(...)`  
  Convenience plots using matplotlib.

- `to_excel(path, case=None, alpha=0.05)` and `export_all(path, horizon=40, B=499, case=None, alpha=0.05)`  
  Write ECM, long‑run, bounds, bootstrap, asymmetry, diagnostics, and spec tables to a single Excel workbook. `export_all` also writes a sheet per variable with the dynamic multipliers.

---

## Bounds test cases at a glance

- **Case I**: no deterministic terms in model or test  
- **Case II**: constant in both model and test  
- **Case III**: constant in model, not in test  
- **Case IV**: constant and trend in model, only trend tested  
- **Case V**: constant and trend in model, neither tested  

The module includes a small table of PSS‑style asymptotic critical values for `k ≤ 5` and `α ∈ {10%, 5%, 1%}`. When outside this range, the fields are returned as missing. Use the bootstrap alternative if you need data‑driven criticals for your exact design. fileciteturn0file0

---

## Data requirements and conventions

- Provide a tidy `DataFrame`. Pass the dependent variable name through `dep`, and list all asymmetric variables in `asym_vars`.
- The class builds **partial‑sum processes** of first differences for each asymmetric variable. Columns with suffix `_pos` and `_neg` are created internally.
- Additional columns not in `asym_vars` are treated as **controls** and enter in levels and differences.
- The design uses Unicode delta in generated column names, for example `Δy_lag1`. Most notebooks and IDEs render these names without problems. If you export tables to CSV, ensure your tool preserves UTF‑8 encoding.

---

## Lag selection strategy

- **Common lags**: grid search over `p ∈ {1..max_lag_endog}`, `q ∈ {0..max_lag_exog}` using AIC or BIC.
- **Per‑exogenous lags**: greedy forward search over `q_k` for each exogenous level term. This keeps the design compact when only some regressors need additional short‑run dynamics.
- You can override with a fixed dictionary `exog_lags={"x1_pos":2, "x1_neg":1, ...}` when you want full control.

---

## HAC defaults

If `cov_type="HAC"` and no `maxlags` is provided, the class applies a Newey–West style rule based on sample size. You can set an explicit integer through `hac_maxlags` or pass a `cov_kwds={"maxlags": ...}` to take full control. fileciteturn0file0

---

## Excel output

`to_excel` and `export_all` write multiple named sheets: `ECM`, `LongRun`, `Bounds`, `BoundsBootstrap`, `Asymmetry`, `Diagnostics`, `Spec`, plus `DM_<var>` sheets when you export dynamic multipliers. The writer tries `xlsxwriter` first, then falls back to `openpyxl`. fileciteturn0file0

---

## Minimal reproducible example

```python
import numpy as np, pandas as pd
from nardl import NARDL

np.random.seed(7)
T = 200
x = np.cumsum(np.random.normal(size=T))
y = 0.6*np.roll(y := np.zeros(T), 1) + 0.2*x + np.random.normal(size=T)  # quick ARDL‑like DGP
df = pd.DataFrame({"y": y, "x": x})

res = NARDL(df, dep="y", asym_vars=["x"], trend="c").fit()
print(res.ecm_summary().head())
print(res.longrun_summary())
print(res.bounds_test())
```

---

## Roadmap and contributions

Planned items include:
- Optional seasonal dummies helper
- Alternative bootstrap schemes and wild bootstrap for heteroskedastic settings
- Convenience wrappers for common publication tables

Contributions and issues are welcome. Please open a descriptive issue with a small dataset and code to reproduce.

---

## License

This project is licensed under the MIT License © 2025 Patrick Onodje. See the [LICENSE](LICENSE) file for details.

---

## Citation

If you use this helper in academic work, please cite the repository and the original NARDL literature as appropriate. Also cite the authors of `statsmodels` and key references for the bounds test.

---

## Acknowledgements

This module relies on `statsmodels` for estimation and testing, and on `pandas` and `numpy` for data handling. The Excel writers `xlsxwriter` and `openpyxl` are used for output. The plotting helpers rely on `matplotlib`.

