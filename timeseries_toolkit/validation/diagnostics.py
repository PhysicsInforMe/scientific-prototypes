"""
Forecast Diagnostics Module.

This module provides comprehensive diagnostic tools for evaluating
time series forecasting models through statistical testing.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.stats as stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from statsmodels.stats.diagnostic import acorr_ljungbox


class ForensicEnsembleAnalyzer:
    """
    Comprehensive diagnostic suite for forecast model evaluation.

    Performs 7 statistical tests to validate whether forecast errors
    behave like white noise (indicating a good model) or contain
    exploitable patterns (indicating model deficiencies).

    The 7 tests are:
    1. Baseline Check: Does the model beat naive benchmarks?
    2. Ljung-Box: Are residuals white noise (no autocorrelation)?
    3. Shapiro-Wilk: Are residuals normally distributed?
    4. Spectral Analysis: Is there hidden periodicity in errors?
    5. Hurst Exponent: Is there long-range dependence?
    6. Entropy Ratio: Is error complexity appropriate?
    7. Feature Leakage: Can features predict residuals?

    Attributes:
        n_samples: Number of samples in test set.
        y_true: True target values.
        dates: Date labels for samples.

    Example:
        >>> analyzer = ForensicEnsembleAnalyzer(
        ...     df=results_df,
        ...     target_col='actual',
        ...     model_cols=['model_A', 'model_B'],
        ...     date_col='date'
        ... )
        >>> report = analyzer.run_full_analysis()
        >>> print(report)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str,
        model_cols: List[str],
        date_col: Optional[str] = None,
        horizon_col: Optional[str] = None,
        analysis_horizon: Optional[int] = None,
        feature_cols: Optional[List[str]] = None
    ):
        """
        Initialize the ForensicEnsembleAnalyzer.

        Args:
            df: DataFrame with forecast results.
            target_col: Name of column containing actual values.
            model_cols: List of column names containing model predictions.
            date_col: Name of date column for sorting. If None, uses index.
            horizon_col: Name of column containing forecast horizon.
            analysis_horizon: Specific horizon to analyze (filters df).
            feature_cols: Feature columns for leakage test. If None, skips test 7.
        """
        self.target_col = target_col
        self.model_cols = model_cols
        self.feature_cols = feature_cols if feature_cols else []

        # Filter by horizon if specified
        if horizon_col is not None and analysis_horizon is not None:
            self.df_test = df[df[horizon_col] == analysis_horizon].copy()
        else:
            self.df_test = df.copy()

        # Sort by date if available
        if date_col is not None:
            self._sort_by_date(date_col)
            self.dates = self.df_test[date_col].values
        else:
            self.dates = np.arange(len(self.df_test))

        # Remove NaN in target
        self.df_test = self.df_test.dropna(subset=[target_col])

        # Extract vectors
        self.y_true = self.df_test[target_col].values
        self.n_samples = len(self.y_true)

        # Generate internal benchmarks
        self._generate_benchmarks()

    def _sort_by_date(self, date_col: str) -> None:
        """Sort DataFrame by date, handling various date formats."""
        try:
            # Try to handle 'YYYYQx' format
            self.df_test['_dt_sort'] = self.df_test[date_col].astype(str).apply(
                lambda x: pd.to_datetime(
                    x.replace('Q1', '-03-31').replace('Q2', '-06-30')
                     .replace('Q3', '-09-30').replace('Q4', '-12-31')
                )
            )
            self.df_test = self.df_test.sort_values(by='_dt_sort')
        except Exception:
            # Fallback to standard sorting
            self.df_test = self.df_test.sort_values(by=date_col)

    def _generate_benchmarks(self) -> None:
        """Generate naive benchmark forecasts."""
        # Naive: previous value (t-1)
        self.benchmark_naive = pd.Series(self.y_true).shift(1).values
        # Seasonal naive: value from 4 periods ago (t-4, for quarterly)
        self.benchmark_snaive = pd.Series(self.y_true).shift(4).values

    def _test_baseline_check(self, mae_model: float) -> Tuple[bool, float, float]:
        """
        Test 1: Check if model beats naive benchmarks.

        Args:
            mae_model: MAE of the model.

        Returns:
            Tuple of (passed, mae_naive, mae_seasonal_naive).
        """
        # Naive benchmark
        mask_n = ~np.isnan(self.benchmark_naive) & ~np.isnan(self.y_true)
        if np.sum(mask_n) > 0:
            mae_naive = np.mean(np.abs(
                self.y_true[mask_n] - self.benchmark_naive[mask_n]
            ))
        else:
            mae_naive = np.inf

        # Seasonal naive benchmark
        mask_s = ~np.isnan(self.benchmark_snaive) & ~np.isnan(self.y_true)
        if np.sum(mask_s) > 0:
            mae_snaive = np.mean(np.abs(
                self.y_true[mask_s] - self.benchmark_snaive[mask_s]
            ))
        else:
            mae_snaive = np.inf

        passed = (mae_model < mae_naive) and (mae_model < mae_snaive)
        return passed, mae_naive, mae_snaive

    def _test_white_noise_ljungbox(
        self,
        residuals: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Test 2: Ljung-Box test for white noise.

        Args:
            residuals: Model residuals.

        Returns:
            Tuple of (passed, p_value).
        """
        if len(residuals) < 5:
            return False, 0.0

        try:
            lags = min(10, int(len(residuals) / 5))
            lb = acorr_ljungbox(residuals, lags=[lags], return_df=True)
            p_val = lb['lb_pvalue'].iloc[0]
            return p_val > 0.05, p_val
        except Exception:
            return False, 0.0

    def _test_normality_shapiro(
        self,
        residuals: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Test 3: Shapiro-Wilk test for normality.

        Args:
            residuals: Model residuals.

        Returns:
            Tuple of (passed, p_value).
        """
        if len(residuals) < 3:
            return False, 0.0

        stat, p_val = stats.shapiro(residuals)
        return p_val > 0.05, p_val

    def _test_spectral_analysis(
        self,
        residuals: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Test 4: Check for hidden periodicity using Welch's method.

        Args:
            residuals: Model residuals.

        Returns:
            Tuple of (passed, coefficient_of_variation).
        """
        if len(residuals) < 5:
            return True, 0.0

        # Set nperseg to avoid warning when signal is shorter than default (256)
        nperseg = min(256, len(residuals))
        freqs, psd = signal.welch(residuals, nperseg=nperseg)
        cv_psd = np.std(psd) / (np.mean(psd) + 1e-9)

        # High CV indicates periodic patterns
        return cv_psd < 1.2, cv_psd

    def _test_hurst_exponent(
        self,
        residuals: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Test 5: Hurst exponent for long-range dependence.

        Args:
            residuals: Model residuals.

        Returns:
            Tuple of (passed, hurst_exponent).
            H ~ 0.5 indicates random walk (good).
            H > 0.5 indicates persistence (bad).
            H < 0.5 indicates mean reversion.
        """
        try:
            lags = range(2, min(20, len(residuals) // 2))
            if len(list(lags)) < 2:
                return True, 0.5

            tau = []
            for lag in lags:
                diff = residuals[lag:] - residuals[:-lag]
                tau.append(np.sqrt(np.std(diff)))

            poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
            H = poly[0]

            # H should be close to 0.5 for white noise
            return (0.4 <= H <= 0.6), H
        except Exception:
            return False, 0.5

    def _test_entropy_ratio(
        self,
        residuals: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Test 6: Compare entropy of residuals vs target.

        Args:
            residuals: Model residuals.

        Returns:
            Tuple of (passed, entropy_ratio).
        """
        def calc_entropy(x: np.ndarray, bins: int = 20) -> float:
            hist, _ = np.histogram(x, bins=bins, density=True)
            hist = hist[hist > 0]
            return -np.sum(hist * np.log(hist))

        e_res = calc_entropy(residuals)
        e_tgt = calc_entropy(self.y_true)

        if e_tgt == 0:
            return True, 1.0

        ratio = e_res / e_tgt
        # Residuals should retain most entropy (not be too predictable)
        return ratio > 0.75, ratio

    def _test_feature_leakage(
        self,
        residuals: np.ndarray
    ) -> Tuple[Optional[bool], float]:
        """
        Test 7: Check if features can predict residuals (leakage).

        Args:
            residuals: Model residuals.

        Returns:
            Tuple of (passed, adjusted_r2). None for passed if skipped.
        """
        if not self.feature_cols:
            return None, 0.0

        missing_cols = [c for c in self.feature_cols if c not in self.df_test.columns]
        if missing_cols:
            return None, 0.0

        X = self.df_test[self.feature_cols].fillna(0)
        y = residuals

        if len(y) < 10:
            return None, 0.0

        model = RandomForestRegressor(
            n_estimators=50,
            max_depth=3,
            random_state=42
        )

        try:
            r2_scores = cross_val_score(model, X, y, cv=3, scoring='r2')
            mean_r2 = np.mean(r2_scores)

            # Compute adjusted R2
            n = len(y)
            k = X.shape[1]
            denom = n - k - 1
            if denom <= 0:
                denom = 1
            adj_r2 = 1 - (1 - mean_r2) * (n - 1) / denom

            # Features should NOT be able to predict residuals
            return adj_r2 <= 0.05, adj_r2
        except Exception:
            return None, 0.0

    def run_full_analysis(self) -> pd.DataFrame:
        """
        Run all 7 tests on each model and generate a summary report.

        Returns:
            DataFrame with test results for each model, sorted by
            forensic score (descending) and MAE (ascending).
        """
        results = []
        max_score = 7 if self.feature_cols else 6

        for model_name in self.model_cols:
            if model_name not in self.df_test.columns:
                continue

            y_pred = self.df_test[model_name].values
            mask = ~np.isnan(y_pred) & ~np.isnan(self.y_true)
            residuals = self.y_true[mask] - y_pred[mask]

            if len(residuals) < 5:
                continue

            # Compute error metrics
            mae = np.mean(np.abs(residuals))
            rmse = np.sqrt(np.mean(residuals ** 2))

            # Run all tests
            t1, mae_naive, mae_snaive = self._test_baseline_check(mae)
            t2, p_lb = self._test_white_noise_ljungbox(residuals)
            t3, p_sw = self._test_normality_shapiro(residuals)
            t4, cv_psd = self._test_spectral_analysis(residuals)
            t5, hurst = self._test_hurst_exponent(residuals)
            t6, entropy_ratio = self._test_entropy_ratio(residuals)
            t7, adj_r2 = self._test_feature_leakage(residuals)

            # Compute score
            score_elements = [t1, t2, t3, t4, t5, t6]
            if t7 is not None:
                score_elements.append(t7)

            total_score = sum([int(x) for x in score_elements if x is not None])

            result = {
                'Model': model_name,
                f'Forensic_Score (/{max_score})': total_score,
                'MAE': round(mae, 4),
                'RMSE': round(rmse, 4),
                '1_Baseline_Beat': t1,
                '2_WhiteNoise_Pass': t2,
                '3_Normality_Pass': t3,
                '4_Spectral_Pass': t4,
                '5_Hurst_Pass': t5,
                '6_Entropy_Pass': t6,
            }

            if t7 is not None:
                result['7_Leakage_Pass'] = t7
            else:
                result['7_Leakage_Pass'] = 'Skipped'

            results.append(result)

        if not results:
            return pd.DataFrame()

        result_df = pd.DataFrame(results)
        score_col = f'Forensic_Score (/{max_score})'
        result_df = result_df.sort_values(
            by=[score_col, 'MAE'],
            ascending=[False, True]
        )

        return result_df

    def get_detailed_results(
        self,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Get detailed test results for a specific model.

        Args:
            model_name: Name of model column.

        Returns:
            Dictionary with all test statistics and diagnostics.
        """
        if model_name not in self.df_test.columns:
            return {}

        y_pred = self.df_test[model_name].values
        mask = ~np.isnan(y_pred) & ~np.isnan(self.y_true)
        residuals = self.y_true[mask] - y_pred[mask]

        if len(residuals) < 5:
            return {}

        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals ** 2))

        t1, mae_naive, mae_snaive = self._test_baseline_check(mae)
        t2, p_lb = self._test_white_noise_ljungbox(residuals)
        t3, p_sw = self._test_normality_shapiro(residuals)
        t4, cv_psd = self._test_spectral_analysis(residuals)
        t5, hurst = self._test_hurst_exponent(residuals)
        t6, entropy_ratio = self._test_entropy_ratio(residuals)
        t7, adj_r2 = self._test_feature_leakage(residuals)

        return {
            'model': model_name,
            'n_samples': len(residuals),
            'mae': mae,
            'rmse': rmse,
            'test_1_baseline': {
                'passed': t1,
                'mae_model': mae,
                'mae_naive': mae_naive,
                'mae_seasonal_naive': mae_snaive,
            },
            'test_2_ljung_box': {
                'passed': t2,
                'p_value': p_lb,
            },
            'test_3_shapiro_wilk': {
                'passed': t3,
                'p_value': p_sw,
            },
            'test_4_spectral': {
                'passed': t4,
                'cv_psd': cv_psd,
            },
            'test_5_hurst': {
                'passed': t5,
                'hurst_exponent': hurst,
            },
            'test_6_entropy': {
                'passed': t6,
                'entropy_ratio': entropy_ratio,
            },
            'test_7_leakage': {
                'passed': t7,
                'adjusted_r2': adj_r2,
            },
            'residuals': residuals,
        }

    def get_residuals(self, model_name: str) -> np.ndarray:
        """
        Get residuals for a specific model.

        Args:
            model_name: Name of model column.

        Returns:
            Array of residuals (actual - predicted).
        """
        if model_name not in self.df_test.columns:
            return np.array([])

        y_pred = self.df_test[model_name].values
        mask = ~np.isnan(y_pred) & ~np.isnan(self.y_true)

        return self.y_true[mask] - y_pred[mask]
