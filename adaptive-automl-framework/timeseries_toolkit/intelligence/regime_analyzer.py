"""
Layer 1: Regime Detection.

Wraps the HMM-based RegimeDetector to classify market states into
four interpretable regimes: bull, bear, crisis, sideways.

The classification uses return statistics and volatility from the
fitted HMM states. States are labelled by mapping mean return and
variance to the four canonical regimes.

Design decisions:
    - 4 regimes cover the main market environments a pipeline must adapt to.
    - Confidence is taken from the smoothed HMM posterior probability.
    - Transition risk is the probability of leaving the current regime
      within *window* days, derived from the transition matrix.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from timeseries_toolkit.models.regime import RegimeDetector


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class RegimeResult:
    """Container for regime detection output."""

    current_regime: str = "unknown"
    confidence: float = 0.0
    regime_probabilities: Dict[str, float] = field(default_factory=dict)
    transition_matrix: Optional[pd.DataFrame] = None
    days_in_regime: int = 0
    transition_risk: float = 0.0
    regime_history: Optional[pd.Series] = None


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class RegimeAnalyzer:
    """
    Detects market regime using HMM and supplementary indicators.

    Regimes
    -------
    - **bull**: Positive mean return, low volatility.
    - **bear**: Negative mean return, elevated volatility.
    - **crisis**: Negative mean return, very high volatility / large drawdowns.
    - **sideways**: Near-zero mean return, low-to-moderate volatility.

    The mapping from raw HMM states to named regimes is performed
    automatically after fitting, based on the mean and variance of
    returns in each state.
    """

    # Canonical regime names in semantic order.
    REGIME_NAMES = ["bull", "bear", "crisis", "sideways"]

    def __init__(self, n_regimes: int = 4):
        """
        Args:
            n_regimes: Number of HMM states.  Default 4 maps cleanly
                       to bull / bear / crisis / sideways.
        """
        self.n_regimes = n_regimes
        # RegimeDetector uses max_states for BIC selection; fix to n_regimes
        # by passing n_states directly at fit time.
        self.regime_detector = RegimeDetector(max_states=n_regimes)
        # Mapping from HMM integer state -> human label, set after fit.
        self._state_map: Dict[int, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        prices: pd.DataFrame,
        volatility: Optional[pd.Series] = None,
        macro_indicators: Optional[pd.DataFrame] = None,
    ) -> RegimeResult:
        """
        Detect the current regime and transition probabilities.

        Args:
            prices: DataFrame with at least one column of price data
                    (Close prices). Index must be DatetimeIndex.
            volatility: Optional external volatility series (e.g. VIX).
                        Currently used only for supplementary validation.
            macro_indicators: Reserved for future use.

        Returns:
            RegimeResult with current regime, confidence, and history.
        """
        # --- Step 1: Compute log returns from prices ----------------------
        # Log returns are better behaved for HMM fitting than raw prices.
        if isinstance(prices, pd.DataFrame):
            price_series = prices.iloc[:, 0]
        else:
            price_series = prices

        returns = np.log(price_series / price_series.shift(1)).dropna()

        if len(returns) < 60:
            # Not enough data for reliable regime detection.
            warnings.warn(
                "Less than 60 observations; regime detection may be unreliable."
            )

        # --- Step 2: Fit the HMM ------------------------------------------
        # Fix n_states so that BIC selection is skipped; we always want
        # exactly self.n_regimes states.
        self.regime_detector.fit(
            returns, n_states=self.n_regimes, auto_select=False
        )

        # --- Step 3: Map HMM states to semantic labels --------------------
        self._map_states_to_regimes(returns)

        # --- Step 4: Extract results --------------------------------------
        raw_regimes = self.regime_detector.predict_regimes()
        try:
            raw_probs = self.regime_detector.get_regime_probabilities()
        except (ValueError, IndexError):
            # HMM may use fewer states than requested; build uniform probs.
            n_obs = len(raw_regimes)
            unique_states = sorted(raw_regimes.unique())
            prob_data = {}
            for s in unique_states:
                prob_data[f"regime_{s}"] = (raw_regimes == s).astype(float).values
            raw_probs = pd.DataFrame(prob_data, index=raw_regimes.index)
        trans_matrix = self.regime_detector.get_transition_matrix()

        # Translate integer states to named regimes.
        named_regimes = raw_regimes.map(self._state_map)
        named_probs = raw_probs.rename(
            columns={c: self._state_map.get(i, f"state_{i}")
                     for i, c in enumerate(raw_probs.columns)}
        )

        # Current regime is determined from the posterior probabilities at
        # t_last (not from the Viterbi path).  This avoids the pathological
        # case where Viterbi assigns a state whose marginal posterior is
        # near-zero, producing a contradictory "regime X at 0% confidence".
        last_probs = named_probs.iloc[-1]
        current_regime = str(last_probs.idxmax())

        # Confidence = posterior probability of the selected regime at t_last.
        confidence = float(last_probs.max())

        # Days in current regime: count consecutive days at end.
        days_in_regime = self._count_days_in_regime(named_regimes)

        # Transition risk: probability of leaving current regime in 14 days.
        current_state = raw_regimes.iloc[-1]
        transition_risk = self._transition_risk(trans_matrix, current_state, window=14)

        # Regime probability summary (latest).
        regime_prob_dict = {col: float(named_probs[col].iloc[-1])
                           for col in named_probs.columns}

        # Build labelled transition matrix.
        label_list = [self._state_map.get(i, f"state_{i}")
                      for i in range(len(trans_matrix))]
        labelled_trans = pd.DataFrame(
            trans_matrix.values, index=label_list, columns=label_list
        )

        return RegimeResult(
            current_regime=current_regime,
            confidence=confidence,
            regime_probabilities=regime_prob_dict,
            transition_matrix=labelled_trans,
            days_in_regime=days_in_regime,
            transition_risk=transition_risk,
            regime_history=named_regimes,
        )

    def get_regime_history(
        self,
        prices: pd.DataFrame,
        resample_freq: str = "W",
    ) -> pd.DataFrame:
        """
        Get historical regime classification resampled to *resample_freq*.

        Useful for backtesting: returns a DataFrame with one column per regime
        containing the proportion of days in that regime during each period.
        """
        result = self.detect(prices)
        if result.regime_history is None:
            return pd.DataFrame()

        # One-hot encode regimes and resample to target frequency.
        dummies = pd.get_dummies(result.regime_history)
        return dummies.resample(resample_freq).mean()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _map_states_to_regimes(self, returns: pd.Series) -> None:
        """
        Assign semantic labels to HMM states based on return statistics.

        Strategy:
        1. Compute mean return and variance per state.
        2. Rank by variance (ascending).
        3. Among low-variance states: positive mean → bull, near-zero → sideways.
        4. Among high-variance states: negative mean → crisis, else → bear.
        """
        regimes = self.regime_detector.predict_regimes()
        stats = []
        for state in sorted(regimes.unique()):
            mask = regimes == state
            state_returns = returns.loc[mask.values[:len(returns)]]
            stats.append({
                "state": state,
                "mean": state_returns.mean() if len(state_returns) > 0 else 0.0,
                "var": state_returns.var() if len(state_returns) > 0 else 0.0,
            })

        # Sort by variance ascending.
        stats.sort(key=lambda s: s["var"])

        # Assign labels.  With exactly 4 states:
        #   lowest var  + positive mean → bull
        #   lowest var  + ~zero/neg mean → sideways
        #   highest var + negative mean → crisis
        #   remaining                   → bear
        assigned: Dict[int, str] = {}
        used_labels = set()

        if len(stats) >= 4:
            # Lowest variance state.
            if stats[0]["mean"] > 0:
                assigned[stats[0]["state"]] = "bull"
                assigned[stats[1]["state"]] = "sideways"
            else:
                assigned[stats[0]["state"]] = "sideways"
                assigned[stats[1]["state"]] = "bull" if stats[1]["mean"] > 0 else "bear"

            # Highest variance state.
            if stats[-1]["mean"] < 0:
                assigned[stats[-1]["state"]] = "crisis"
            else:
                assigned[stats[-1]["state"]] = "bear"

            # Fill remaining.
            for s in stats:
                if s["state"] not in assigned:
                    for lbl in ["bear", "sideways", "bull", "crisis"]:
                        if lbl not in assigned.values():
                            assigned[s["state"]] = lbl
                            break
        else:
            # Fewer than 4 states: simple heuristic.
            labels_pool = list(self.REGIME_NAMES)
            for i, s in enumerate(stats):
                assigned[s["state"]] = labels_pool[i % len(labels_pool)]

        self._state_map = assigned

    @staticmethod
    def _count_days_in_regime(regime_series: pd.Series) -> int:
        """Count consecutive days the last regime has persisted."""
        if len(regime_series) == 0:
            return 0
        current = regime_series.iloc[-1]
        count = 0
        for val in reversed(regime_series.values):
            if val == current:
                count += 1
            else:
                break
        return count

    @staticmethod
    def _transition_risk(
        trans_matrix: pd.DataFrame,
        current_state: int,
        window: int = 14,
    ) -> float:
        """
        Probability of leaving *current_state* within *window* days.

        Uses the transition matrix raised to the *window*-th power:
            risk = 1 - P^window[current, current]
        """
        P = trans_matrix.values.astype(float)
        try:
            P_w = np.linalg.matrix_power(P, window)
            stay_prob = P_w[current_state, current_state]
            return float(1.0 - stay_prob)
        except Exception:
            return 0.5  # conservative fallback
