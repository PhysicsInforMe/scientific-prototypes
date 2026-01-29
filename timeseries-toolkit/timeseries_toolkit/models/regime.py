"""
Regime Detection Module.

This module provides tools for identifying market regimes using
Hidden Markov Models with Gaussian Mixture emissions (GMMHMM).
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

try:
    from hmmlearn.hmm import GaussianHMM, GMMHMM
    HAS_HMMLEARN = True
except ImportError:
    HAS_HMMLEARN = False


class RegimeDetector:
    """
    Market regime detector using Gaussian Mixture Hidden Markov Models.

    This class identifies distinct market regimes (e.g., bull/bear markets,
    high/low volatility periods) from time series data. It uses GMMHMM from
    hmmlearn with automatic state selection via cross-validation.

    Attributes:
        is_fitted: Whether the model has been fitted.
        optimal_states: Number of states selected via cross-validation.
        model: The fitted GMMHMM model.

    Example:
        >>> detector = RegimeDetector(max_states=4)
        >>> detector.fit(spread_series)
        >>> regimes = detector.predict_regimes()
        >>> probs = detector.get_regime_probabilities()
    """

    def __init__(
        self,
        max_states: int = 5,
        n_cv_splits: int = 5,
        covariance_type: str = 'full',
        n_iter: int = 200,
        random_state: int = 42
    ):
        """
        Initialize the RegimeDetector.

        Args:
            max_states: Maximum number of states to test during selection.
            n_cv_splits: Number of cross-validation folds for state selection.
            covariance_type: Type of covariance parameters ('full', 'diag', 'tied').
            n_iter: Maximum number of EM iterations.
            random_state: Random seed for reproducibility.

        Raises:
            ImportError: If hmmlearn is not installed.
        """
        if not HAS_HMMLEARN:
            raise ImportError(
                "hmmlearn is required for RegimeDetector. "
                "Install it with: pip install hmmlearn"
            )

        self.max_states = max_states
        self.n_cv_splits = n_cv_splits
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state

        self.is_fitted: bool = False
        self.optimal_states: Optional[int] = None
        self.model: Optional[GMMHMM] = None
        self._series: Optional[pd.Series] = None
        self._regime_data: Optional[pd.DataFrame] = None
        self._cv_scores: Optional[Dict[int, float]] = None

    def _select_optimal_states(self, X: np.ndarray) -> int:
        """
        Select optimal number of states using time series cross-validation.

        Args:
            X: Data array of shape (n_samples, n_features).

        Returns:
            Optimal number of states.
        """
        possible_states = range(2, self.max_states + 1)
        cv_scores = {}

        tscv = TimeSeriesSplit(n_splits=self.n_cv_splits)

        for n_components in possible_states:
            fold_scores = []

            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]

                # Skip if training set too small
                if len(X_train) < n_components * 10:
                    continue

                try:
                    model = GMMHMM(
                        n_components=n_components,
                        covariance_type=self.covariance_type,
                        n_iter=100,  # Fewer iterations for CV
                        random_state=self.random_state,
                        tol=1e-3
                    )
                    model.fit(X_train)
                    score = model.score(X_test)
                    fold_scores.append(score)
                except (ValueError, np.linalg.LinAlgError):
                    continue

            if fold_scores:
                cv_scores[n_components] = np.nanmean(fold_scores)

        self._cv_scores = cv_scores

        if not cv_scores:
            return 2  # Default to 2 states

        return max(cv_scores, key=cv_scores.get)

    def fit(
        self,
        series: pd.Series,
        n_states: Optional[int] = None,
        auto_select: bool = True
    ) -> 'RegimeDetector':
        """
        Fit the regime model to time series data.

        Args:
            series: Time series data with DatetimeIndex.
            n_states: Number of states (overrides auto-selection).
            auto_select: Whether to automatically select optimal states.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If series is empty or too short.
        """
        if series.empty:
            raise ValueError("Input series cannot be empty")

        self._series = series.copy()
        X = series.values.reshape(-1, 1)

        # Select number of states
        if n_states is not None:
            self.optimal_states = n_states
        elif auto_select:
            self.optimal_states = self._select_optimal_states(X)
        else:
            self.optimal_states = 2

        # Fit final model
        self.model = GMMHMM(
            n_components=self.optimal_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state
        )
        self.model.fit(X)

        # Create regime DataFrame
        self._regime_data = pd.DataFrame(index=series.index)
        self._regime_data['value'] = series.values
        self._regime_data['regime'] = self.model.predict(X)

        # Order regimes by volatility (ascending)
        self._order_regimes_by_volatility()

        self.is_fitted = True
        return self

    def _order_regimes_by_volatility(self) -> None:
        """Reorder regime labels so that 0 = lowest volatility."""
        regime_vols = self._regime_data.groupby('regime')['value'].std().sort_values()
        mapping = {old: new for new, old in enumerate(regime_vols.index)}
        self._regime_data['regime'] = self._regime_data['regime'].map(mapping)

    def predict_regimes(self) -> pd.Series:
        """
        Get the Viterbi path (most likely regime sequence).

        Returns:
            Series with regime labels (0 = lowest volatility, etc.).

        Raises:
            ValueError: If model has not been fitted.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting")

        return self._regime_data['regime'].copy()

    def get_regime_probabilities(self) -> pd.DataFrame:
        """
        Get smoothed regime probabilities for each time point.

        Returns:
            DataFrame with columns 'regime_0', 'regime_1', etc.
            containing probabilities that sum to 1 at each time point.

        Raises:
            ValueError: If model has not been fitted.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting probabilities")

        X = self._series.values.reshape(-1, 1)

        # Get posterior probabilities
        _, posteriors = self.model.score_samples(X)

        # Reorder columns to match regime ordering
        regime_vols = self._regime_data.groupby('regime')['value'].std().sort_values()
        original_order = list(regime_vols.index)

        # Create DataFrame with proper column ordering
        prob_df = pd.DataFrame(
            posteriors,
            index=self._series.index,
            columns=[f'regime_{i}' for i in range(self.optimal_states)]
        )

        # Reorder columns based on volatility
        reordered_cols = [f'regime_{original_order.index(i)}' for i in range(self.optimal_states)]
        result = pd.DataFrame(index=self._series.index)
        for new_idx, old_col in enumerate(reordered_cols):
            result[f'regime_{new_idx}'] = prob_df[old_col]

        return result

    def aggregate_to_frequency(
        self,
        target_freq: str = 'QE',
        method: str = 'mean'
    ) -> pd.DataFrame:
        """
        Aggregate regime probabilities to a lower frequency.

        Useful for mixed-frequency analysis where regimes are detected at
        high frequency (e.g., daily) but need to be used with low-frequency
        data (e.g., quarterly GDP).

        Args:
            target_freq: Target frequency string (e.g., 'QE', 'ME', 'W').
            method: Aggregation method ('mean' for proportions, 'last', 'first').

        Returns:
            DataFrame with regime probabilities at target frequency.

        Raises:
            ValueError: If model has not been fitted.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before aggregating")

        probs = self.get_regime_probabilities()

        if method == 'mean':
            return probs.resample(target_freq).mean()
        elif method == 'last':
            return probs.resample(target_freq).last()
        elif method == 'first':
            return probs.resample(target_freq).first()
        else:
            raise ValueError(f"Unknown method: {method}")

    def get_regime_statistics(self) -> pd.DataFrame:
        """
        Get summary statistics for each regime.

        Returns:
            DataFrame with mean, std, min, max, and count for each regime.

        Raises:
            ValueError: If model has not been fitted.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before computing statistics")

        stats = self._regime_data.groupby('regime')['value'].agg([
            'mean', 'std', 'min', 'max', 'count'
        ])
        stats['proportion'] = stats['count'] / len(self._regime_data)

        return stats

    def get_transition_matrix(self) -> pd.DataFrame:
        """
        Get the estimated transition probability matrix.

        Returns:
            DataFrame with transition probabilities where element (i, j)
            is the probability of transitioning from regime i to regime j.

        Raises:
            ValueError: If model has not been fitted.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting transition matrix")

        trans_mat = self.model.transmat_

        return pd.DataFrame(
            trans_mat,
            index=[f'from_regime_{i}' for i in range(self.optimal_states)],
            columns=[f'to_regime_{i}' for i in range(self.optimal_states)]
        )

    def get_cv_scores(self) -> Optional[Dict[int, float]]:
        """
        Get cross-validation scores from state selection.

        Returns:
            Dictionary mapping number of states to CV log-likelihood,
            or None if auto-selection was not used.
        """
        return self._cv_scores

    def predict_proba_new(self, new_series: pd.Series) -> pd.DataFrame:
        """
        Predict regime probabilities for new data.

        Args:
            new_series: New time series data.

        Returns:
            DataFrame with regime probabilities.

        Raises:
            ValueError: If model has not been fitted.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting")

        X = new_series.values.reshape(-1, 1)
        _, posteriors = self.model.score_samples(X)

        return pd.DataFrame(
            posteriors,
            index=new_series.index,
            columns=[f'regime_{i}' for i in range(self.optimal_states)]
        )
