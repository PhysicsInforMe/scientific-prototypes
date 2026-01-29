"""
Layer 3: Explainable Output.

Generates human-readable reports from analysis results.  Supports
three output styles:
    - **executive**: High-level summary for decision makers.
    - **technical**: Detailed statistics and diagnostics.
    - **brief**: One-paragraph overview.

The ``IntelligenceReport`` class is the central container that
holds all results and provides export methods (dict, DataFrame,
Markdown, file).

Design notes:
    - All monetary values are formatted with locale-appropriate separators.
    - Confidence is mapped to qualitative labels (HIGH / MEDIUM / LOW).
    - Warnings are generated automatically from diagnostics and regime risk.
    - The Markdown export follows a fixed template so that downstream
      renderers can parse sections predictably.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from timeseries_toolkit.intelligence.pipelines import DiagnosticsReport
from timeseries_toolkit.intelligence.regime_analyzer import RegimeResult


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class CausalityResult:
    """Container for causal analysis output."""
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    driver_status: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ComparisonReport:
    """Container for multi-asset comparison."""
    assets: List[str] = field(default_factory=list)
    correlations: Optional[pd.DataFrame] = None
    relative_strength: Optional[pd.DataFrame] = None
    individual_reports: Dict[str, "IntelligenceReport"] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# IntelligenceReport
# ---------------------------------------------------------------------------

@dataclass
class IntelligenceReport:
    """
    Container for all analysis results.

    Provides export to dict, DataFrame, Markdown string, and file.
    """

    # Metadata
    timestamp: Optional[datetime] = None
    assets: List[str] = field(default_factory=list)
    horizon: str = ""

    # Core results
    forecast: Optional[pd.DataFrame] = None  # columns: forecast, lower, upper
    regime: Optional[RegimeResult] = None
    drivers: Optional[CausalityResult] = None

    # Pipeline info
    pipeline_used: str = ""
    pipeline_reason: str = ""
    diagnostics: Optional[DiagnosticsReport] = None
    quality_score: float = 0.0  # 0-1

    # Explanations
    summary: str = ""
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Data context
    current_prices: Dict[str, float] = field(default_factory=dict)
    data_range: str = ""
    observations_used: int = 0

    # ------------------------------------------------------------------
    # Export methods
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Export report as a nested dictionary."""
        return {
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "assets": self.assets,
            "horizon": self.horizon,
            "quality_score": self.quality_score,
            "pipeline_used": self.pipeline_used,
            "pipeline_reason": self.pipeline_reason,
            "summary": self.summary,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "regime": {
                "current": self.regime.current_regime if self.regime else None,
                "confidence": self.regime.confidence if self.regime else None,
                "days_in_regime": self.regime.days_in_regime if self.regime else None,
                "transition_risk": self.regime.transition_risk if self.regime else None,
                "probabilities": self.regime.regime_probabilities if self.regime else {},
            },
            "forecast": self.forecast.to_dict() if self.forecast is not None else None,
            "diagnostics": {
                "pass_count": self.diagnostics.pass_count if self.diagnostics else 0,
                "total_tests": self.diagnostics.total_tests if self.diagnostics else 0,
                "details": self.diagnostics.details if self.diagnostics else {},
            },
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Export forecast as DataFrame."""
        if self.forecast is not None:
            return self.forecast.copy()
        return pd.DataFrame()

    def to_markdown(self, include_charts: bool = False) -> str:
        """
        Export report as formatted Markdown.

        Args:
            include_charts: If True, include base64-encoded chart images
                            (requires matplotlib).

        Returns:
            Complete Markdown document as string.
        """
        lines: List[str] = []

        # --- Header -------------------------------------------------------
        lines.append("# Market Intelligence Report")
        lines.append("")
        ts_str = self.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC") if self.timestamp else "N/A"
        lines.append(f"**Generated**: {ts_str}  ")
        lines.append(f"**Assets**: {', '.join(self.assets)}  ")
        lines.append(f"**Horizon**: {self.horizon}  ")
        qs = _quality_label(self.quality_score)
        lines.append(f"**Quality Score**: {self.quality_score:.2f} ({qs})")
        lines.append("")
        lines.append("---")
        lines.append("")

        # --- Executive Summary --------------------------------------------
        lines.append("## Executive Summary")
        lines.append("")
        if self.summary:
            lines.append(self.summary)
        lines.append("")
        lines.append("---")
        lines.append("")

        # --- Regime Analysis ----------------------------------------------
        if self.regime:
            lines.append("## Current Regime Analysis")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| Current Regime | {self.regime.current_regime.title()} |")
            lines.append(f"| Confidence | {self.regime.confidence:.0%} |")
            lines.append(f"| Days in Regime | {self.regime.days_in_regime} |")
            lines.append(f"| Transition Risk (14d) | {self.regime.transition_risk:.0%} |")
            lines.append("")

            # Regime probabilities table.
            if self.regime.regime_probabilities:
                lines.append("### Regime Probabilities")
                lines.append("")
                lines.append("| Regime | Probability |")
                lines.append("|--------|-------------|")
                for r, p in sorted(self.regime.regime_probabilities.items(),
                                   key=lambda x: -x[1]):
                    lines.append(f"| {r.title()} | {p:.0%} |")
                lines.append("")

            lines.append("---")
            lines.append("")

        # --- Forecast Details ---------------------------------------------
        if self.forecast is not None and len(self.forecast) > 0:
            lines.append("## Forecast Details")
            lines.append("")
            for asset in self.assets:
                lines.append(f"### {asset}")
                lines.append("")
                lines.append("| Date | Forecast | Lower (95%) | Upper (95%) |")
                lines.append("|------|----------|-------------|-------------|")
                for idx, row in self.forecast.iterrows():
                    date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)
                    lines.append(
                        f"| {date_str} | {row['forecast']:.2f} "
                        f"| {row['lower']:.2f} | {row['upper']:.2f} |"
                    )
                lines.append("")
            lines.append("---")
            lines.append("")

        # --- Key Drivers --------------------------------------------------
        if self.drivers and self.drivers.relationships:
            lines.append("## Key Drivers")
            lines.append("")
            lines.append("### Causal Relationships Detected")
            lines.append("")
            for i, rel in enumerate(self.drivers.relationships, 1):
                src = rel.get("source", "?")
                tgt = rel.get("target", "?")
                strength = rel.get("strength", 0)
                lines.append(f"{i}. **{src} → {tgt}** (Strength: {strength:.2f})")
            lines.append("")
            lines.append("---")
            lines.append("")

        # --- Pipeline & Diagnostics ---------------------------------------
        lines.append("## Pipeline & Diagnostics")
        lines.append("")
        lines.append("### Selected Pipeline")
        lines.append("")
        lines.append(f"**Name**: {self.pipeline_used}  ")
        lines.append(f"**Reason**: {self.pipeline_reason}")
        lines.append("")

        if self.diagnostics:
            lines.append("### Diagnostic Results")
            lines.append("")
            lines.append("| Test | Result |")
            lines.append("|------|--------|")
            for test_name, result in self.diagnostics.details.items():
                icon = "PASS" if result == "PASS" else ("SKIP" if result == "SKIP" else "FAIL")
                lines.append(f"| {test_name} | {icon} |")
            lines.append("")

        lines.append("---")
        lines.append("")

        # --- Warnings & Recommendations ----------------------------------
        if self.warnings or self.recommendations:
            lines.append("## Warnings & Recommendations")
            lines.append("")
            if self.warnings:
                lines.append("### Warnings")
                lines.append("")
                for w in self.warnings:
                    lines.append(f"- {w}")
                lines.append("")
            if self.recommendations:
                lines.append("### Recommendations")
                lines.append("")
                for r in self.recommendations:
                    lines.append(f"- {r}")
                lines.append("")
            lines.append("---")
            lines.append("")

        # --- Technical Details --------------------------------------------
        lines.append("## Technical Details")
        lines.append("")
        lines.append(f"- **Data Range**: {self.data_range}")
        lines.append(f"- **Observations Used**: {self.observations_used}")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("*Report generated by MarketIntelligence v1.0*  ")
        lines.append("*This is not financial advice. Past performance does not guarantee future results.*")

        return "\n".join(lines)

    def save_markdown(self, filepath: str, include_charts: bool = False) -> None:
        """Save report as a Markdown file."""
        md = self.to_markdown(include_charts=include_charts)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(md)


# ---------------------------------------------------------------------------
# Explainer
# ---------------------------------------------------------------------------

class Explainer:
    """
    Generates human-readable explanations of analysis results.

    Combines regime detection, forecast, drivers, and diagnostics
    into a coherent narrative.
    """

    def generate_report(
        self,
        forecast: pd.DataFrame,
        regime: Optional[RegimeResult],
        drivers: Optional[CausalityResult],
        diagnostics: Optional[DiagnosticsReport],
        pipeline_name: str,
        pipeline_reason: str,
        assets: List[str],
        horizon: str,
        current_prices: Optional[Dict[str, float]] = None,
        data_range: str = "",
        observations_used: int = 0,
    ) -> IntelligenceReport:
        """
        Compile all results into a comprehensive report.
        """
        report = IntelligenceReport(
            timestamp=datetime.now(timezone.utc),
            assets=assets,
            horizon=horizon,
            forecast=forecast,
            regime=regime,
            drivers=drivers,
            pipeline_used=pipeline_name,
            pipeline_reason=pipeline_reason,
            diagnostics=diagnostics,
            current_prices=current_prices or {},
            data_range=data_range,
            observations_used=observations_used,
        )

        # --- Quality score: fraction of diagnostics tests passed ----------
        if diagnostics and diagnostics.total_tests > 0:
            report.quality_score = diagnostics.pass_count / diagnostics.total_tests
        else:
            report.quality_score = 0.5  # unknown → medium

        # --- Warnings -----------------------------------------------------
        report.warnings = self._generate_warnings(regime, diagnostics, forecast)

        # --- Recommendations ----------------------------------------------
        report.recommendations = self._generate_recommendations(regime, drivers)

        # --- Summary text -------------------------------------------------
        report.summary = self.generate_summary(report, style="executive")

        return report

    def generate_summary(
        self,
        report: IntelligenceReport,
        style: str = "executive",
    ) -> str:
        """
        Generate plain-language summary.

        Styles:
            - executive: multi-line overview with key metrics.
            - technical: includes diagnostic details.
            - brief: single paragraph.
        """
        parts: List[str] = []

        # Regime line.
        if report.regime:
            r = report.regime
            parts.append(
                f"MARKET REGIME: {r.current_regime.title()} "
                f"({r.confidence:.0%} confidence)"
            )
            parts.append("")

        # Forecast lines.
        if report.forecast is not None and len(report.forecast) > 0:
            last_fc = report.forecast.iloc[-1]
            fc_val = last_fc["forecast"]
            for asset in report.assets:
                current = report.current_prices.get(asset)
                if current and current > 0:
                    pct = (fc_val - current) / current * 100
                    direction = "+" if pct >= 0 else ""
                    parts.append(
                        f"{asset} {report.horizon} Outlook: "
                        f"${current:,.2f} -> ${fc_val:,.2f} ({direction}{pct:.1f}%)"
                    )
                else:
                    parts.append(
                        f"{asset} {report.horizon} Outlook: {fc_val:,.2f}"
                    )

            # Confidence label.
            ql = _quality_label(report.quality_score)
            parts.append(f"Confidence: {ql}")
            parts.append("")

        # Warnings.
        if report.warnings:
            parts.append("Key Factors:")
            for w in report.warnings[:3]:
                parts.append(f"  - {w}")
            parts.append("")

        # Quality line.
        if report.diagnostics:
            d = report.diagnostics
            parts.append(
                f"Quality: {_quality_label(report.quality_score)} "
                f"({d.pass_count}/{d.total_tests} diagnostics pass)"
            )

        # Transition risk.
        if report.regime and report.regime.transition_risk > 0.15:
            parts.append(
                f"Risk: {report.regime.transition_risk:.0%} chance of regime shift "
                f"within 14 days"
            )

        if style == "brief":
            return " ".join(p.strip() for p in parts if p.strip())
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_warnings(
        regime: Optional[RegimeResult],
        diagnostics: Optional[DiagnosticsReport],
        forecast: Optional[pd.DataFrame],
    ) -> List[str]:
        """Auto-generate warnings from analysis results."""
        warnings_list: List[str] = []

        if regime:
            if regime.transition_risk > 0.2:
                warnings_list.append(
                    f"Regime transition risk elevated ({regime.transition_risk:.0%})"
                )
            if regime.current_regime in ("crisis", "bear"):
                warnings_list.append(
                    f"Currently in {regime.current_regime} regime; "
                    "forecast uncertainty is higher than usual"
                )

        if diagnostics:
            if diagnostics.pass_count < diagnostics.total_tests:
                failed = diagnostics.total_tests - diagnostics.pass_count
                warnings_list.append(
                    f"{failed} of {diagnostics.total_tests} diagnostic tests failed"
                )

        if forecast is not None and len(forecast) > 0:
            band_width = (forecast["upper"] - forecast["lower"]).mean()
            fc_mean = forecast["forecast"].mean()
            if fc_mean != 0 and band_width / abs(fc_mean) > 0.2:
                warnings_list.append(
                    "Wide confidence bands indicate elevated uncertainty"
                )

        return warnings_list

    @staticmethod
    def _generate_recommendations(
        regime: Optional[RegimeResult],
        drivers: Optional[CausalityResult],
    ) -> List[str]:
        """Auto-generate recommendations."""
        recs: List[str] = []

        if regime:
            if regime.current_regime == "bear":
                recs.append("Consider reducing position sizes given bear regime")
            elif regime.current_regime == "crisis":
                recs.append("Consider defensive positioning; crisis regime active")
                recs.append("Monitor volatility indicators for stabilisation signals")
            elif regime.current_regime == "bull":
                recs.append("Trend-following strategies may be effective in bull regime")
            elif regime.current_regime == "sideways":
                recs.append("Range-bound market; mean-reversion strategies may outperform")

        return recs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quality_label(score: float) -> str:
    """Map numeric quality score to a label."""
    if score >= 0.8:
        return "HIGH"
    elif score >= 0.6:
        return "MEDIUM"
    elif score >= 0.4:
        return "LOW"
    return "UNRELIABLE"
