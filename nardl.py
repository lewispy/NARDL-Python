# nardl.py
# NARDL helper with per-exog lags, bounds cases (PSS I–V), Pesaran CVs, HAC defaults,
# dynamic multipliers with bootstrap bands, and grid subplots (rows x 2: pos|neg).
# Author: Patrick Onodje

from __future__ import annotations

import numpy as np
import pandas as pd
import itertools
import math
from typing import List, Optional, Tuple, Dict, Union
from pathlib import Path

import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResults
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_arch


def _nw_maxlags_rule(T: int) -> int:
    """Newey–West maxlags rule of thumb."""
    return max(1, int(round(1.2 * (T ** (1.0 / 3.0)))))


def _parse_bounds_case(case: Optional[Union[str, int]], trend: str) -> str:
    """
    Normalize bounds test 'case' into one of: 'I','II','III','IV','V'.
    Defaults by model trend:
      - trend='n'  -> 'I'   (no deterministic terms)
      - trend='c'  -> 'III' (const in model, not in test)
      - trend='ct' -> 'V'   (const+trend in model, neither included in test)
    Accepts descriptive strings too.
    """
    if case is None or str(case).lower() in {"auto","default"}:
        return {"n":"I","c":"III","ct":"V"}.get(trend, "III")
    s = str(case).strip().lower()
    mapping = {
        "i": "I",
        "none": "I",
        "no deterministic terms": "I",
        "ii": "II",
        "const in model & test": "II",
        "constant included in both the model and the test": "II",
        "iii": "III",
        "const in model only": "III",
        "constant included in the model but not in the test": "III",
        "iv": "IV",
        "trend only in test": "IV",
        "constant and trend included in the model, only trend included in the test": "IV",
        "v": "V",
        "neither in test": "V",
        "constant and trend included in the model, neither included in the test": "V",
    }
    return mapping.get(s, s.upper() if s in {"I","II","III","IV","V"} else "III")


def _deterministic_restrictions(case_code: str) -> Dict[str, bool]:
    """
    Which deterministic terms are restricted (included in the F-test) by case.
    Keys: 'const', 'trend' -> True if restricted (zero under H0), False otherwise.
    """
    if case_code == "II":
        return {"const": True, "trend": False}
    if case_code == "IV":
        return {"const": False, "trend": True}
    return {"const": False, "trend": False}


# Minimal embedded asymptotic Pesaran-Shin-Smith style CVs (k<=5) for convenience.
# PSS_CV[case][k][alpha] = (I0, I1), alpha in {0.10, 0.05, 0.01}
PSS_CV: Dict[str, Dict[int, Dict[float, Tuple[float, float]]]] = {
    "I": {
        1: {0.10: (2.79, 3.67), 0.05: (3.41, 4.36), 0.01: (4.66, 5.77)},
        2: {0.10: (3.15, 4.11), 0.05: (3.79, 4.85), 0.01: (5.15, 6.36)},
        3: {0.10: (3.47, 4.52), 0.05: (4.18, 5.32), 0.01: (5.61, 6.84)},
        4: {0.10: (3.79, 4.89), 0.05: (4.55, 5.75), 0.01: (6.03, 7.24)},
        5: {0.10: (4.06, 5.23), 0.05: (4.89, 6.12), 0.01: (6.41, 7.61)},
    },
    "II": {
        1: {0.10: (3.17, 4.14), 0.05: (3.79, 4.85), 0.01: (5.15, 6.36)},
        2: {0.10: (3.62, 4.67), 0.05: (4.32, 5.29), 0.01: (5.73, 6.84)},
        3: {0.10: (4.01, 5.07), 0.05: (4.89, 5.98), 0.01: (6.41, 7.56)},
        4: {0.10: (4.37, 5.43), 0.05: (5.23, 6.36), 0.01: (6.84, 7.99)},
        5: {0.10: (4.66, 5.74), 0.05: (5.57, 6.73), 0.01: (7.23, 8.40)},
    },
    "III": {
        1: {0.10: (4.04, 4.78), 0.05: (4.94, 5.73), 0.01: (6.84, 7.84)},
        2: {0.10: (4.78, 5.73), 0.05: (5.73, 6.68), 0.01: (7.56, 8.73)},
        3: {0.10: (5.15, 6.07), 0.05: (6.41, 7.37), 0.01: (8.29, 9.50)},
        4: {0.10: (5.52, 6.49), 0.05: (7.01, 7.98), 0.01: (8.86,10.04)},
        5: {0.10: (5.86, 6.84), 0.05: (7.56, 8.56), 0.01: (9.42,10.63)},
    },
    "IV": {
        1: {0.10: (4.45, 5.52), 0.05: (5.15, 6.36), 0.01: (6.73, 8.10)},
        2: {0.10: (5.03, 6.14), 0.05: (5.86, 7.19), 0.01: (7.56, 8.93)},
        3: {0.10: (5.47, 6.59), 0.05: (6.41, 7.76), 0.01: (8.18, 9.59)},
        4: {0.10: (5.86, 7.01), 0.05: (6.84, 8.29), 0.01: (8.73,10.13)},
        5: {0.10: (6.24, 7.37), 0.05: (7.23, 8.73), 0.01: (9.24,10.68)},
    },
    "V": {
        1: {0.10: (4.83, 5.77), 0.05: (5.59, 6.79), 0.01: (7.19, 8.45)},
        2: {0.10: (5.27, 6.31), 0.05: (6.22, 7.42), 0.01: (7.90, 9.17)},
        3: {0.10: (5.73, 6.84), 0.05: (6.73, 7.98), 0.01: (8.62, 9.95)},
    },
}
def get_pesaran_bounds(case_code: str, k: int, alpha: float = 0.05) -> Optional[Tuple[float, float]]:
    try:
        return PSS_CV[case_code][int(k)][float(alpha)]
    except Exception:
        return None


class NARDL:
    """
    Core API:
      fit()                       -> estimate UECM with lag selection
      ecm_summary()               -> short-run form with ECT(-1)
      longrun_summary()           -> long-run coefficients via delta method
      bounds_test(case=...)       -> F-test with PSS-style cases I–V (with Pesaran CVs if available)
      bounds_bootstrap(case=...)  -> bootstrap p-value/criticals under the same case
      asymmetry_tests()           -> Wald tests (short/long) for each asym var
      diagnostics()               -> BG & ARCH tests
      export_all(path, ...)       -> Excel export + dynamic multipliers tables
      plot_dynamic_multiplier(var, ...)        -> single var, pos/neg with bands
      plot_all_multipliers(...)                -> grid: rows=len(asym_vars), cols=2 (pos|neg)
    """
    def __init__(
        self,
        data: pd.DataFrame,
        dep: str,
        asym_vars: List[str],
        max_lag_endog: int = 3,
        max_lag_exog: int = 3,
        trend: str = "c",              # "n", "c", "ct"
        ic: str = "aic",
        cov_type: str = "HAC",
        cov_kwds: Optional[dict] = None,
        lag_select_grid: Optional[Tuple[List[int], List[int]]] = None,
        per_exog_lags: bool = True,    # independent q_k by default
        exog_lags: Optional[Dict[str, int]] = None,
        dropna: bool = True,
        seed: Optional[int] = None,
        hac_maxlags: Optional[Union[int, str]] = "auto",
    ):
        self.data = data.copy()
        self.dep = dep
        self.asym_vars = list(asym_vars)
        self.trend = trend.lower()
        assert self.trend in {"n","c","ct"}, "trend must be 'n', 'c', or 'ct'"
        self.ic = ic.lower()
        self.max_lag_endog = int(max_lag_endog)
        self.max_lag_exog = int(max_lag_exog)
        self.cov_type = cov_type
        self.cov_kwds = cov_kwds.copy() if cov_kwds else {}
        self.lag_select_grid = lag_select_grid
        self.per_exog_lags = per_exog_lags
        self.exog_lags = exog_lags.copy() if exog_lags else None
        self.dropna = dropna
        self.seed = seed
        self.hac_maxlags = hac_maxlags

        # Learned
        self.controls_: List[str] = []
        self.exog_level_cols_: List[str] = []
        self.posneg_map_: Dict[str, Tuple[str, str]] = {}
        self.p_: Optional[int] = None
        self.q_: Optional[int] = None
        self.qk_: Optional[Dict[str, int]] = None
        self.uecm_res_: Optional[RegressionResults] = None
        self.uecm_design_cols_: List[str] = []
        self.longrun_: Optional[pd.DataFrame] = None
        self.sample_index_: Optional[pd.Index] = None
        self._df_prepared: Optional[pd.DataFrame] = None

    # ---- Variable prep ----

    @staticmethod
    def _partial_sums(x: pd.Series) -> Tuple[pd.Series, pd.Series]:
        dx = x.diff()
        pos = dx.clip(lower=0).fillna(0).cumsum()
        neg = dx.clip(upper=0).fillna(0).cumsum()
        return pos.rename(x.name + "_pos"), neg.rename(x.name + "_neg")

    def _prepare_variables(self) -> pd.DataFrame:
        cols = [c for c in self.data.columns if c != self.dep]
        self.controls_ = [c for c in cols if c not in self.asym_vars]

        df = self.data.copy()
        for v in self.asym_vars:
            pos, neg = self._partial_sums(df[v])
            df[pos.name] = pos
            df[neg.name] = neg
            self.posneg_map_[v] = (pos.name, neg.name)

        self.exog_level_cols_ = []
        for v in self.asym_vars:
            self.exog_level_cols_.extend(list(self.posneg_map_[v]))
        self.exog_level_cols_.extend(self.controls_)
        self._df_prepared = df
        return df

    # ---- Design ----

    def _build_uecm_design(self, df: pd.DataFrame, p: int,
                           q: Optional[int] = None, qk: Optional[Dict[str, int]] = None
                           ) -> Tuple[pd.Series, pd.DataFrame]:
        y = df[self.dep].astype(float)
        Δy = y.diff()

        X = pd.DataFrame(index=df.index)

        if self.trend in {"c","ct"}:
            X["const"] = 1.0
        if self.trend == "ct":
            X["trend"] = np.arange(len(df), dtype=float)

        X["y_lag1"] = y.shift(1)
        for col in self.exog_level_cols_:
            X[f"{col}_lag1"] = df[col].shift(1)

        for i in range(1, max(p - 0, 1)):
            X[f"Δy_lag{i}"] = Δy.shift(i)

        if qk is None:
            q_use = 0 if q is None else int(q)
            for col in self.exog_level_cols_:
                dx = df[col].diff()
                for j in range(0, q_use):
                    X[f"Δ{col}_lag{j}"] = dx.shift(j)
        else:
            for col in self.exog_level_cols_:
                q_use = int(qk.get(col, 0))
                dx = df[col].diff()
                for j in range(0, q_use):
                    X[f"Δ{col}_lag{j}"] = dx.shift(j)

        Z = pd.concat([Δy.rename("Δy"), X], axis=1)
        if self.dropna:
            Z = Z.dropna(axis=0)

        target = Z["Δy"]
        X = Z.drop(columns=["Δy"])
        return target, X

    def _apply_hac_defaults(self, nobs: int, base_kwds: Optional[dict] = None) -> dict:
        cov_kwds = (base_kwds or {}).copy()
        if "maxlags" in cov_kwds:
            return cov_kwds
        if (self.hac_maxlags is not None) and (self.hac_maxlags != "auto"):
            try:
                cov_kwds["maxlags"] = int(self.hac_maxlags)
                return cov_kwds
            except Exception:
                pass
        if str(self.hac_maxlags).lower() == "auto":
            cov_kwds["maxlags"] = _nw_maxlags_rule(int(nobs))
            return cov_kwds
        return cov_kwds

    def _fit_uecm(self, y: pd.Series, X: pd.DataFrame) -> RegressionResults:
        cov_type = self.cov_type
        cov_kwds = self.cov_kwds.copy()
        if str(cov_type).upper() == "HAC":
            cov_kwds = self._apply_hac_defaults(len(y), cov_kwds)
        model = sm.OLS(y, X, missing="drop")
        res = model.fit(cov_type=cov_type, cov_kwds=cov_kwds)
        return res

    def _ic_value(self, res: RegressionResults) -> float:
        return res.aic if self.ic == "aic" else res.bic

    # ---- Lag selection ----

    def _select_lags_common(self, df: pd.DataFrame) -> Tuple[int, int, RegressionResults, pd.Series, pd.DataFrame]:
        p_grid = self.lag_select_grid[0] if (self.lag_select_grid and len(self.lag_select_grid) == 2) else list(range(1, self.max_lag_endog + 1))
        q_grid = self.lag_select_grid[1] if (self.lag_select_grid and len(self.lag_select_grid) == 2) else list(range(0, self.max_lag_exog + 1))

        best_ic = np.inf
        best = None

        for pp, qq in itertools.product(p_grid, q_grid):
            y_t, X_t = self._build_uecm_design(df, pp, q=qq, qk=None)
            if len(X_t) < (X_t.shape[1] + 5):
                continue
            res_t = self._fit_uecm(y_t, X_t)
            ic_val = self._ic_value(res_t)
            if ic_val < best_ic:
                best_ic = ic_val
                best = (pp, qq, res_t, y_t, X_t)

        if best is None:
            raise ValueError("Unable to fit any UECM specification with common lags. Adjust lag limits.")
        return best

    def _select_lags_greeΔy_qk(self, df: pd.DataFrame) -> Tuple[int, Dict[str, int], RegressionResults, pd.Series, pd.DataFrame]:
        pp_best, _, res0, y0, X0 = self._select_lags_common(df)
        qk = {col: 0 for col in self.exog_level_cols_}

        improved = True
        best_ic = self._ic_value(res0)
        best_pack = (pp_best, qk.copy(), res0, y0, X0)

        while improved:
            improved = False
            for col in self.exog_level_cols_:
                if qk[col] >= self.max_lag_exog:
                    continue
                trial_qk = qk.copy()
                trial_qk[col] += 1
                y_t, X_t = self._build_uecm_design(df, pp_best, q=None, qk=trial_qk)
                if len(X_t) < (X_t.shape[1] + 5):
                    continue
                res_t = self._fit_uecm(y_t, X_t)
                ic_val = self._ic_value(res_t)
                if ic_val + 1e-8 < best_ic:
                    best_ic = ic_val
                    best_pack = (pp_best, trial_qk.copy(), res_t, y_t, X_t)
                    qk = trial_qk
                    improved = True
                    break

        return best_pack

    # ---- Fit ----

    def fit(self, p: Optional[int] = None, q: Optional[int] = None) -> "NARDL":
        df = self._prepare_variables()

        if self.exog_lags is not None:
            pp_best, _, res0, y0, X0 = self._select_lags_common(df)
            y_t, X_t = self._build_uecm_design(df, p or pp_best, q=None, qk=self.exog_lags)
            res = self._fit_uecm(y_t, X_t)
            self.p_, self.q_, self.qk_ = (p or pp_best), None, self.exog_lags.copy()
            self.uecm_res_, self.uecm_design_cols_ = res, list(X_t.columns)
            self.sample_index_ = y_t.index
        elif self.per_exog_lags:
            pp_best, qk_best, res, y_t, X_t = self._select_lags_greeΔy_qk(df)
            self.p_, self.q_, self.qk_ = pp_best, None, qk_best
            self.uecm_res_, self.uecm_design_cols_ = res, list(X_t.columns)
            self.sample_index_ = y_t.index
        else:
            if (p is not None) or (q is not None):
                y_t, X_t = self._build_uecm_design(df, p or 1, q=q or 0, qk=None)
                res = self._fit_uecm(y_t, X_t)
                self.p_, self.q_, self.qk_ = (p or 1), (q or 0), None
                self.uecm_res_, self.uecm_design_cols_ = res, list(X_t.columns)
                self.sample_index_ = y_t.index
            else:
                pp_best, qq_best, res, y_t, X_t = self._select_lags_common(df)
                self.p_, self.q_, self.qk_ = pp_best, qq_best, None
                self.uecm_res_, self.uecm_design_cols_ = res, list(X_t.columns)
                self.sample_index_ = y_t.index

        self.longrun_ = self._compute_longrun_table()
        return self

    # ---- Summaries ----

    def ecm_summary(self) -> pd.DataFrame:
        self._check_fitted()
        res = self.uecm_res_
        params = res.params.copy()
        bse = res.bse
        tvals = res.tvalues
        pvals = res.pvalues

        rows = []
        if "const" in params.index:
            rows.append(("const", params["const"], bse["const"], tvals["const"], pvals["const"]))
        if "trend" in params.index:
            rows.append(("trend", params["trend"], bse["trend"], tvals["trend"], pvals["trend"]))

        if "y_lag1" not in params.index:
            raise RuntimeError("y_lag1 not found in UECM results.")
        rows.append(("ECT(-1)", params["y_lag1"], bse["y_lag1"], tvals["y_lag1"], pvals["y_lag1"]))

        for col in self.uecm_design_cols_:
            if col.startswith("Δy_lag") or col.startswith("Δ"):
                rows.append((col, params.get(col, np.nan), bse.get(col, np.nan), tvals.get(col, np.nan), pvals.get(col, np.nan)))

        out = pd.DataFrame(rows, columns=["term", "coef", "std_err", "t", "p"])
        return out.set_index("term")

    def longrun_summary(self) -> pd.DataFrame:
        self._check_fitted()
        return self.longrun_.copy()

    # ---- Bounds ----

    def _bounds_R_matrix(self, res: RegressionResults, case: Optional[Union[str,int]]) -> np.ndarray:
        case_code = _parse_bounds_case(case, self.trend)
        det_flags = _deterministic_restrictions(case_code)

        level_params = ["y_lag1"] + [f"{col}_lag1" for col in self.exog_level_cols_]
        if det_flags.get("const", False) and "const" in res.params.index:
            level_params = ["const"] + level_params
        if det_flags.get("trend", False) and "trend" in res.params.index:
            level_params = ["trend"] + level_params

        level_params = [p for p in level_params if p in res.params.index]

        R = np.zeros((len(level_params), len(res.params)))
        param_index = {name: i for i, name in enumerate(res.params.index)}
        for i, name in enumerate(level_params):
            R[i, param_index[name]] = 1.0
        return R

    def bounds_test(self, case: Optional[Union[str,int]] = None, alpha: float = 0.05,
                    use_pesaran: bool = True) -> pd.DataFrame:
        self._check_fitted()
        res = self.uecm_res_
        case_code = _parse_bounds_case(case, self.trend)
        R = self._bounds_R_matrix(res, case_code)
        ftest = res.f_test(R)
        F = float(np.squeeze(ftest.fvalue))
        df_num = int(ftest.df_num) if hasattr(ftest, "df_num") else R.shape[0]
        df_den = int(ftest.df_den) if hasattr(ftest, "df_den") else res.df_resid
        k_levels = len([c for c in self.exog_level_cols_ if f"{c}_lag1" in res.params.index])

        row = {"case": case_code, "F-stat": F, "df_num": df_num, "df_den": df_den, "k_levels": k_levels}
        if use_pesaran:
            cv = get_pesaran_bounds(case_code, k_levels, alpha)
            if cv is not None:
                I0, I1 = cv
                decision = "cointegration" if F > I1 else ("no cointegration" if F < I0 else "inconclusive")
                row.update({f"I0@{alpha:.2f}": I0, f"I1@{alpha:.2f}": I1, f"decision@{alpha:.2f}": decision})
            else:
                row.update({f"I0@{alpha:.2f}": np.nan, f"I1@{alpha:.2f}": np.nan, f"decision@{alpha:.2f}": "CV unavailable"})
        return pd.DataFrame([row])

    def bounds_bootstrap(self, B: int = 999, case: Optional[Union[str,int]] = None,
                         alpha: float = 0.05, random_state: Optional[int] = None) -> pd.DataFrame:
        self._check_fitted()
        rng = np.random.default_rng(self.seed if random_state is None else random_state)

        res = self.uecm_res_
        y = res.model.endog.copy()
        X = pd.DataFrame(res.model.exog, columns=res.model.exog_names, index=self.sample_index_)

        case_code = _parse_bounds_case(case, self.trend)
        det_flags = _deterministic_restrictions(case_code)

        keep_cols = [c for c in X.columns if c.startswith("Δ") or c.startswith("Δy_lag")]
        if not det_flags.get("const", False) and "const" in X.columns:
            keep_cols.append("const")
        if not det_flags.get("trend", False) and "trend" in X.columns:
            keep_cols.append("trend")
        X_null = X[keep_cols]

        cov_type = self.cov_type
        cov_kwds_null = self.cov_kwds.copy()
        cov_kwds_star = self.cov_kwds.copy()
        if str(cov_type).upper() == "HAC":
            cov_kwds_null = self._apply_hac_defaults(len(y), cov_kwds_null)
            cov_kwds_star = self._apply_hac_defaults(len(y), cov_kwds_star)

        model_null = sm.OLS(y, X_null, missing="drop")
        res_null = model_null.fit(cov_type=cov_type, cov_kwds=cov_kwds_null)
        uhat = res_null.resid.values

        R = self._bounds_R_matrix(res, case_code)
        obs = float(np.squeeze(res.f_test(R).fvalue))

        stats = np.empty(B, dtype=float)
        for b in range(B):
            u_star = rng.choice(uhat, size=len(uhat), replace=True)
            y_star = X_null.values @ res_null.params.values + u_star

            model_star = sm.OLS(y_star, X, missing="drop")
            res_star = model_star.fit(cov_type=cov_type, cov_kwds=cov_kwds_star)

            R_star = self._bounds_R_matrix(res_star, case_code)
            f_star = res_star.f_test(R_star)
            stats[b] = float(np.squeeze(f_star.fvalue))

        pval = (np.sum(stats >= obs) + 1.0) / (B + 1.0)
        q90, q95, q99 = np.quantile(stats, [0.90, 0.95, 0.99])
        decision = "cointegration" if obs > q95 else ("no cointegration" if obs < q90 else "inconclusive")
        out = pd.DataFrame({"case":[case_code], "F-stat":[obs], "p_boot":[pval], "q90":[q90], "q95":[q95], "q99":[q99], "decision@%":[decision]})
        out.rename(columns={"decision@%": "decision@5%"}, inplace=True)
        return out

    # ---- Asymmetry ----

    def asymmetry_tests(self) -> pd.DataFrame:
        self._check_fitted()
        res = self.uecm_res_
        param_names = list(res.params.index)
        idx = {n: i for i, n in enumerate(param_names)}

        records = []
        for v in self.asym_vars:
            pos_name, neg_name = self.posneg_map_[v]
            lr_left = f"{pos_name}_lag1"
            lr_right = f"{neg_name}_lag1"
            if (lr_left in idx) and (lr_right in idx):
                R = np.zeros((1, len(param_names)))
                R[0, idx[lr_left]] = 1.0
                R[0, idx[lr_right]] = -1.0
                w = self.uecm_res_.wald_test(R)
                records.append([v, "long_run", float(np.squeeze(w.statistic)), float(np.squeeze(w.pvalue))])
            else:
                records.append([v, "long_run", np.nan, np.nan])

            pos_diffs = [name for name in param_names if name.startswith(f"Δ{pos_name}_lag")]
            neg_diffs = [name for name in param_names if name.startswith(f"Δ{neg_name}_lag")]
            if pos_diffs and neg_diffs:
                R = np.zeros((1, len(param_names)))
                for n in pos_diffs: R[0, idx[n]] += 1.0
                for n in neg_diffs: R[0, idx[n]] -= 1.0
                w = self.uecm_res_.wald_test(R)
                records.append([v, "short_run", float(np.squeeze(w.statistic)), float(np.squeeze(w.pvalue))])
            else:
                records.append([v, "short_run", np.nan, np.nan])

        return pd.DataFrame(records, columns=["variable", "test", "wald_stat", "p_value"])

    # ---- Diagnostics ----

    def diagnostics(self, bg_lags: Optional[int] = None, arch_lags: int = 4) -> pd.DataFrame:
        self._check_fitted()
        res = self.uecm_res_
        if bg_lags is None:
            bg_lags = max(1, (self.p_ or 1))
        bg_lm, bg_p, _, _ = acorr_breusch_godfrey(res, nlags=bg_lags)
        arch_lm, arch_p, _, _ = het_arch(res.resid, nlags=arch_lags)
        return pd.DataFrame(
            {"BG_lags":[bg_lags], "BG_LM":[bg_lm], "BG_p":[bg_p], "ARCH_lags":[arch_lags],
             "ARCH_LM":[arch_lm], "ARCH_p":[arch_p], "R2":[res.rsquared], "Adj_R2":[res.rsquared_adj]}
        )

    # ---- Excel export ----

    def to_excel(self, path: Union[str, Path], case: Optional[Union[str,int]] = None, alpha: float = 0.05) -> Path:
        self._check_fitted()
        path = Path(path)
        ecm = self.ecm_summary()
        lr = self.longrun_summary()
        bt = self.bounds_test(case=case, alpha=alpha, use_pesaran=True)
        try:
            bb = self.bounds_bootstrap(B=499, case=case, alpha=alpha)
        except Exception as e:
            bb = pd.DataFrame({"note": [f"Bootstrap failed: {e}"]})
        asy = self.asymmetry_tests()
        diag = self.diagnostics()
        spec = pd.DataFrame([self.model_spec()])

        try:
            with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
                ecm.to_excel(writer, sheet_name="ECM")
                lr.to_excel(writer, sheet_name="LongRun")
                bt.to_excel(writer, sheet_name="Bounds", index=False)
                bb.to_excel(writer, sheet_name="BoundsBootstrap", index=False)
                asy.to_excel(writer, sheet_name="Asymmetry", index=False)
                diag.to_excel(writer, sheet_name="Diagnostics", index=False)
                spec.to_excel(writer, sheet_name="Spec", index=False)
        except Exception:
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                ecm.to_excel(writer, sheet_name="ECM")
                lr.to_excel(writer, sheet_name="LongRun")
                bt.to_excel(writer, sheet_name="Bounds", index=False)
                bb.to_excel(writer, sheet_name="BoundsBootstrap", index=False)
                asy.to_excel(writer, sheet_name="Asymmetry", index=False)
                diag.to_excel(writer, sheet_name="Diagnostics", index=False)
                spec.to_excel(writer, sheet_name="Spec", index=False)
        return path

    def export_all(self, path: Union[str, Path], horizon: int = 40, B: int = 499,
                   case: Optional[Union[str,int]] = None, alpha: float = 0.05) -> Path:
        self._check_fitted()
        path = Path(path)
        ecm = self.ecm_summary()
        lr = self.longrun_summary()
        bt = self.bounds_test(case=case, alpha=alpha, use_pesaran=True)
        try:
            bb = self.bounds_bootstrap(B=B, case=case, alpha=alpha)
        except Exception as e:
            bb = pd.DataFrame({"note": [f"Bootstrap failed: {e}"]})
        asy = self.asymmetry_tests()
        diag = self.diagnostics()
        spec = pd.DataFrame([self.model_spec()])

        dm_tabs = {}
        for v in self.asym_vars:
            tab = self.dynamic_multipliers_table(v, horizon=horizon)
            dm_tabs[v] = tab

        try:
            with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
                ecm.to_excel(writer, sheet_name="ECM")
                lr.to_excel(writer, sheet_name="LongRun")
                bt.to_excel(writer, sheet_name="Bounds", index=False)
                bb.to_excel(writer, sheet_name="BoundsBootstrap", index=False)
                asy.to_excel(writer, sheet_name="Asymmetry", index=False)
                diag.to_excel(writer, sheet_name="Diagnostics", index=False)
                spec.to_excel(writer, sheet_name="Spec", index=False)
                for v, tab in dm_tabs.items():
                    tab.to_excel(writer, sheet_name=f"DM_{v}", index=False)
        except Exception:
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                ecm.to_excel(writer, sheet_name="ECM")
                lr.to_excel(writer, sheet_name="LongRun")
                bt.to_excel(writer, sheet_name="Bounds", index=False)
                bb.to_excel(writer, sheet_name="BoundsBootstrap", index=False)
                asy.to_excel(writer, sheet_name="Asymmetry", index=False)
                diag.to_excel(writer, sheet_name="Diagnostics", index=False)
                spec.to_excel(writer, sheet_name="Spec", index=False)
                for v, tab in dm_tabs.items():
                    tab.to_excel(writer, sheet_name=f"DM_{v}", index=False)
        return path

    # ---- dynamic multipliers & bands ----

    def dynamic_multiplier(self, var: str, horizon: int = 40, shock: str = "pos") -> pd.DataFrame:
        self._check_fitted()
        assert var in self.asym_vars, f"{var} is not in asym_vars."
        pos_name, neg_name = self.posneg_map_[var]
        use_neg = (shock.lower().startswith("n"))
        shock_col = neg_name if use_neg else pos_name

        res = self.uecm_res_
        phi = res.params.get("y_lag1", 0.0)
        alpha = res.params.get("const", 0.0)
        trend_coef = res.params.get("trend", 0.0)

        theta = {col: res.params.get(f"{col}_lag1", 0.0) for col in self.exog_level_cols_}
        psi = [res.params.get(name, 0.0) for name in self.uecm_design_cols_ if name.startswith("Δy_lag")]

        delta = {col: [] for col in self.exog_level_cols_}
        for name in self.uecm_design_cols_:
            if name.startswith("Δ"):
                for col in self.exog_level_cols_:
                    prefix = f"Δ{col}_lag"
                    if name.startswith(prefix):
                        delta[col].append(res.params.get(name, 0.0))
                        break

        qk_lengths = {col: len(delta[col]) for col in self.exog_level_cols_}
        p_Δy = len(psi)

        H = int(horizon)
        Δy_base = np.zeros(H+1)
        Δy_shock = np.zeros(H+1)
        y_base = np.zeros(H+1+1)
        y_shock = np.zeros(H+1+1)

        xlvl_base = {col: 0.0 for col in self.exog_level_cols_}
        xlvl_shock = {col: 0.0 for col in self.exog_level_cols_}

        Δy_hist_base = [0.0]*p_Δy
        Δy_hist_shock = [0.0]*p_Δy
        dX_hist_base = {col: [0.0]*qk_lengths[col] for col in self.exog_level_cols_}
        dX_hist_shock = {col: [0.0]*qk_lengths[col] for col in self.exog_level_cols_}

        s = -1.0 if use_neg else 1.0
        if qk_lengths[shock_col] > 0:
            dX_hist_shock[shock_col][0] = s
        xlvl_shock[shock_col] += s

        for t in range(0, H+1):
            sr_base = 0.0
            if "trend" in res.params.index:
                sr_base += trend_coef * (t)
            sr_base += phi * (y_base[t])
            for col in self.exog_level_cols_:
                sr_base += theta[col] * (xlvl_base[col])
            for i, coef in enumerate(psi):
                sr_base += coef * (Δy_hist_base[i] if i < len(Δy_hist_base) else 0.0)
            for col in self.exog_level_cols_:
                for j, coef in enumerate(delta[col]):
                    sr_base += coef * (dX_hist_base[col][j])
            Δy_base[t] = alpha + sr_base
            y_base[t+1] = y_base[t] + Δy_base[t]

            sr_shock = 0.0
            if "trend" in res.params.index:
                sr_shock += trend_coef * (t)
            sr_shock += phi * (y_shock[t])
            for col in self.exog_level_cols_:
                sr_shock += theta[col] * (xlvl_shock[col])
            for i, coef in enumerate(psi):
                sr_shock += coef * (Δy_hist_shock[i] if i < len(Δy_hist_shock) else 0.0)
            for col in self.exog_level_cols_:
                for j, coef in enumerate(delta[col]):
                    sr_shock += coef * (dX_hist_shock[col][j])
            Δy_shock[t] = alpha + sr_shock
            y_shock[t+1] = y_shock[t] + Δy_shock[t]

            if p_Δy > 0:
                Δy_hist_base = [Δy_base[t]] + Δy_hist_base[:-1]
                Δy_hist_shock = [Δy_shock[t]] + Δy_hist_shock[:-1]
            for col in self.exog_level_cols_:
                if qk_lengths[col] > 0:
                    dX_hist_base[col] = [0.0] + dX_hist_base[col][:-1]
                    dX_hist_shock[col] = [0.0] + dX_hist_shock[col][:-1]

        dm = (y_shock[1:] - y_base[1:])
        return pd.DataFrame({"h": np.arange(0, H+1), "multiplier": dm})

    def dynamic_multipliers_table(self, var: str, horizon: int = 40) -> pd.DataFrame:
        dm_pos = self.dynamic_multiplier(var, horizon=horizon, shock="pos")["multiplier"].values
        dm_neg = self.dynamic_multiplier(var, horizon=horizon, shock="neg")["multiplier"].values
        return pd.DataFrame({"h": np.arange(0, horizon+1), "pos": dm_pos, "neg": dm_neg})

    def _dm_from_params(self, params: pd.Series, var: str, horizon: int, shock: str) -> np.ndarray:
        res = self.uecm_res_
        phi = params.get("y_lag1", 0.0)
        alpha = params.get("const", 0.0)
        trend_coef = params.get("trend", 0.0)

        theta = {col: params.get(f"{col}_lag1", 0.0) for col in self.exog_level_cols_}
        psi = [params.get(name, 0.0) for name in self.uecm_design_cols_ if name.startswith("Δy_lag")]

        delta = {col: [] for col in self.exog_level_cols_}
        for name in self.uecm_design_cols_:
            if name.startswith("Δ"):
                for col in self.exog_level_cols_:
                    prefix = f"Δ{col}_lag"
                    if name.startswith(prefix):
                        delta[col].append(params.get(name, 0.0))
                        break

        qk_lengths = {col: len(delta[col]) for col in self.exog_level_cols_}
        p_Δy = len(psi)

        H = int(horizon)
        Δy_base = np.zeros(H+1)
        Δy_shock = np.zeros(H+1)
        y_base = np.zeros(H+1+1)
        y_shock = np.zeros(H+1+1)

        xlvl_base = {col: 0.0 for col in self.exog_level_cols_}
        xlvl_shock = {col: 0.0 for col in self.exog_level_cols_}

        Δy_hist_base = [0.0]*p_Δy
        Δy_hist_shock = [0.0]*p_Δy
        dX_hist_base = {col: [0.0]*qk_lengths[col] for col in self.exog_level_cols_}
        dX_hist_shock = {col: [0.0]*qk_lengths[col] for col in self.exog_level_cols_}

        pos_name, neg_name = self.posneg_map_[var]
        use_neg = (shock.lower().startswith("n"))
        shock_col = neg_name if use_neg else pos_name
        s = -1.0 if use_neg else 1.0
        if qk_lengths[shock_col] > 0:
            dX_hist_shock[shock_col][0] = s
        xlvl_shock[shock_col] += s

        for t in range(0, H+1):
            sr_base = 0.0
            if "trend" in res.params.index:
                sr_base += trend_coef * (t)
            sr_base += phi * (y_base[t])
            for col in self.exog_level_cols_:
                sr_base += theta[col] * (xlvl_base[col])
            for i, coef in enumerate(psi):
                sr_base += coef * (Δy_hist_base[i] if i < len(Δy_hist_base) else 0.0)
            for col in self.exog_level_cols_:
                for j, coef in enumerate(delta[col]):
                    sr_base += coef * (dX_hist_base[col][j])
            Δy_base[t] = alpha + sr_base
            y_base[t+1] = y_base[t] + Δy_base[t]

            sr_shock = 0.0
            if "trend" in res.params.index:
                sr_shock += trend_coef * (t)
            sr_shock += phi * (y_shock[t])
            for col in self.exog_level_cols_:
                sr_shock += theta[col] * (xlvl_shock[col])
            for i, coef in enumerate(psi):
                sr_shock += coef * (Δy_hist_shock[i] if i < len(Δy_hist_shock) else 0.0)
            for col in self.exog_level_cols_:
                for j, coef in enumerate(delta[col]):
                    sr_shock += coef * (dX_hist_shock[col][j])
            Δy_shock[t] = alpha + sr_shock
            y_shock[t+1] = y_shock[t] + Δy_shock[t]

            if p_Δy > 0:
                Δy_hist_base = [Δy_base[t]] + Δy_hist_base[:-1]
                Δy_hist_shock = [Δy_shock[t]] + Δy_hist_shock[:-1]
            for col in self.exog_level_cols_:
                if qk_lengths[col] > 0:
                    dX_hist_base[col] = [0.0] + dX_hist_base[col][:-1]
                    dX_hist_shock[col] = [0.0] + dX_hist_shock[col][:-1]

        dm = (y_shock[1:] - y_base[1:])
        return dm

    def multiplier_bands(self, var: str, horizon: int = 40, shock: str = "pos",
                          nboot: int = 999, level: float = 0.95, random_state: Optional[int] = None) -> pd.DataFrame:
        self._check_fitted()
        rng = np.random.default_rng(self.seed if random_state is None else random_state)
        res = self.uecm_res_
        params = res.params
        cov = res.cov_params()

        H = int(horizon)
        dms = np.empty((nboot, H+1), dtype=float)

        cov_use = cov.values.copy()
        try:
            _ = np.linalg.cholesky(cov_use)
        except np.linalg.LinAlgError:
            ridge = 1e-8 * np.eye(cov_use.shape[0])
            cov_use = cov_use + ridge

        names = list(params.index)
        mean = params.values
        for b in range(nboot):
            draw = rng.multivariate_normal(mean, cov_use)
            draw_s = pd.Series(draw, index=names)
            dms[b, :] = self._dm_from_params(draw_s, var=var, horizon=H, shock=shock)

        alpha2 = (1.0 - level) / 2.0
        lo = np.quantile(dms, alpha2, axis=0)
        hi = np.quantile(dms, 1.0 - alpha2, axis=0)
        return pd.DataFrame({"h": np.arange(0, H+1), "lo": lo, "hi": hi})

    def plot_dynamic_multiplier(self, var: str, horizon: int = 40, bands: bool = True,
                                nboot: int = 499, level: float = 0.95):
        import matplotlib.pyplot as plt
        tab = self.dynamic_multipliers_table(var, horizon=horizon)
        plt.figure()
        plt.plot(tab["h"].values, tab["pos"].values, color="#476EAE", label=f"{var} (+)")
        if bands:
            band_pos = self.multiplier_bands(var, horizon=horizon, shock="pos", nboot=nboot, level=level)
            plt.fill_between(band_pos["h"].values, band_pos["lo"].values, band_pos["hi"].values, color="#476EAE", alpha=0.25)
        plt.plot(tab["h"].values, tab["neg"].values, color="#F75270", label=f"{var} (-)")
        if bands:
            band_neg = self.multiplier_bands(var, horizon=horizon, shock="neg", nboot=nboot, level=level)
            plt.fill_between(band_neg["h"].values, band_neg["lo"].values, band_neg["hi"].values, color="#F75270", alpha=0.25)
        plt.axhline(0.0, color="black", linewidth=0.8)
        plt.title(f"dynamic multipliers: {var}")
        plt.xlabel("Horizon")
        plt.ylabel("Δy response")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_all_multipliers(self, horizon: int = 40, bands: bool = True,
                             nboot: int = 499, level: float = 0.95,
                             sharex: bool = True, sharey: bool = False,
                             figsize: Optional[Tuple[float,float]] = None):
        """
        Grid of subplots: rows = number of asym_vars, columns = 2 [pos | neg].
        Pos curves are blue; neg curves are red. Optional filled bootstrap bands.
        """
        import matplotlib.pyplot as plt
        n = len(self.asym_vars)
        if n == 0:
            raise ValueError("No asymmetric variables to plot.")
        if figsize is None:
            figsize = (10, 3*n)
        fig, axes = plt.subplots(nrows=n, ncols=2, figsize=figsize, sharex=sharex, sharey=sharey)
        if n == 1:
            axes = np.array([axes])

        for i, v in enumerate(self.asym_vars):
            tab = self.dynamic_multipliers_table(v, horizon=horizon)

            # POS (left)
            ax_pos = axes[i, 0]
            ax_pos.plot(tab["h"].values, tab["pos"].values, color="#476EAE", label=f"{v} (+)")
            if bands:
                band_pos = self.multiplier_bands(v, horizon=horizon, shock="pos", nboot=nboot, level=level)
                ax_pos.fill_between(band_pos["h"].values, band_pos["lo"].values, band_pos["hi"].values, color="#476EAE", alpha=0.25)
            ax_pos.axhline(0.0, color="black", linewidth=0.8)
            ax_pos.set_title(f"{v} (+)")
            ax_pos.set_ylabel("Δy")

            # NEG (right)
            ax_neg = axes[i, 1]
            ax_neg.plot(tab["h"].values, tab["neg"].values, color="#F75270", label=f"{v} (-)")
            if bands:
                band_neg = self.multiplier_bands(v, horizon=horizon, shock="neg", nboot=nboot, level=level)
                ax_neg.fill_between(band_neg["h"].values, band_neg["lo"].values, band_neg["hi"].values, color="#F75270", alpha=0.25)
            ax_neg.axhline(0.0, color="black", linewidth=0.8)
            ax_neg.set_title(f"{v} (-)")

        for j in range(2):
            axes[-1, j].set_xlabel("Horizon")

        plt.tight_layout()
        plt.show()

    # ---- Internals ----

    def _compute_longrun_table(self) -> pd.DataFrame:
        res = self.uecm_res_
        params = res.params
        cov = res.cov_params()

        if "y_lag1" not in params.index:
            raise RuntimeError("y_lag1 not found in UECM results.")
        phi = params["y_lag1"]

        rows = []

        if "const" in params.index:
            alpha = params["const"]
            d = np.zeros(len(params))
            i_alpha = list(params.index).index("const")
            i_phi = list(params.index).index("y_lag1")
            d[i_alpha] = -1.0 / phi
            d[i_phi] = alpha / (phi ** 2)
            var = d @ cov.values @ d.T
            se = np.sqrt(var)
            z = (-alpha / phi) / se
            p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z)/math.sqrt(2.0))))
            rows.append(("longrun_const", -alpha / phi, se, z, p))

        if "trend" in params.index:
            gamma = params["trend"]
            d = np.zeros(len(params))
            i_gamma = list(params.index).index("trend")
            i_phi = list(params.index).index("y_lag1")
            d[i_gamma] = -1.0 / phi
            d[i_phi] = gamma / (phi ** 2)
            var = d @ cov.values @ d.T
            se = np.sqrt(var)
            z = (-gamma / phi) / se
            p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z)/math.sqrt(2.0))))
            rows.append(("longrun_trend", -gamma / phi, se, z, p))

        for col in self.exog_level_cols_:
            name = f"{col}_lag1"
            if name not in params.index:
                continue
            theta = params[name]
            d = np.zeros(len(params))
            i_theta = list(params.index).index(name)
            i_phi = list(params.index).index("y_lag1")
            d[i_theta] = -1.0 / phi
            d[i_phi] = theta / (phi ** 2)
            var = d @ cov.values @ d.T
            se = np.sqrt(var)
            z = (-theta / phi) / se
            p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z)/math.sqrt(2.0))))
            rows.append((col.replace("_pos", " (pos)").replace("_neg", " (neg)"), -theta / phi, se, z, p))

        out = pd.DataFrame(rows, columns=["term", "coef", "std_err", "z", "p"])
        return out.set_index("term")

    def _check_fitted(self):
        if self.uecm_res_ is None:
            raise RuntimeError("Call .fit() before requesting results.")

    def model_spec(self) -> dict:
        self._check_fitted()
        return {
            "p": self.p_,
            "q": self.q_,
            "qk": self.qk_,
            "levels": ["y_lag1"] + [f"{c}_lag1" for c in self.exog_level_cols_],
            "diffs_y": [c for c in self.uecm_design_cols_ if c.startswith("Δy_lag")],
            "diffs_x": [c for c in self.uecm_design_cols_ if c.startswith("Δ") and not c.startswith("Δy_")],
            "cov_type": self.cov_type,
            "cov_kwds": self.cov_kwds,
            "trend": self.trend,
            "hac_maxlags": self.hac_maxlags,
        }


if __name__ == "__main__":
    pass
