"""
Visualization Module
Creates interactive Plotly charts for SOC analysis
"""

from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.settings import CHART_HEIGHT, CHART_TEMPLATE, COLORS


class PlotlyVisualizer:
    """
    Creates interactive Plotly visualizations for SOC analysis:
    1. Volatility Clustering Heatmap (Price with magnitude coloring)
    2. Power Law Proof (Log-Log Fat Tail analysis)
    3. Criticality & Trend (Traffic Light system)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        symbol: str = "BTCUSDT",
        vol_low_threshold: Optional[float] = None,
        vol_high_threshold: Optional[float] = None,
    ) -> None:
        """
        Initialize visualizer

        Args:
            df: DataFrame with calculated SOC metrics
            symbol: Trading symbol for chart titles
            vol_low_threshold: Lower volatility threshold
            vol_high_threshold: Upper volatility threshold
        """
        self.df = df
        self.symbol = symbol
        self.vol_low_threshold = vol_low_threshold
        self.vol_high_threshold = vol_high_threshold

    def create_interactive_dashboard(self) -> go.Figure:
        """
        Create a Tabbed Dashboard with 3 views.
        Only one chart is visible at a time, selectable via buttons.
        """
        print("Generating interactive dashboard...")

        # Create a single figure with secondary Y-axis capability
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # --- Add Traces for All Charts ---
        # We add all traces to the same figure but toggle their visibility
        
        # Track trace indices for buttons
        # Chart 1 Traces
        self._add_volatility_clustering_traces(fig)
        
        # Chart 2 Traces
        self._add_power_law_traces(fig)
        
        # Chart 3 Traces
        self._add_criticality_trend_traces(fig)

        # --- Define Explanatory Texts ---
        text_1 = (
            "<b>1. Volatility Clustering (The 'Sandpile'):</b><br>"
            "In SOC systems, extreme events do not occur in isolation. They come in waves (clusters).<br>"
            "The color coding shows phases where the system is 'working' (Blue=Stable, Red=Active)."
        )
        
        text_2 = (
            "<b>2. The Power Curve (Log/Log Proof):</b><br>"
            "This is the mathematical 'fingerprint'. Red points = Real Bitcoin data. Blue line = Normal distribution.<br>"
            "The straight line of red points proves the 'Fat Tails' (extremely high probability for Black Swans)."
        )
        
        text_3 = (
            "<b>3. System Criticality & Trading Signals:</b><br>"
            "Combine SOC (Volatility) with Trend (SMA 200).<br>"
            "⚠ <b>Rule:</b> Red phase + Price < SMA 200 = Crash Risk.<br>"
            "✓ <b>Rule:</b> Green phase + Price > SMA 200 = Accumulation."
        )

        # --- Create Layout & Buttons ---
        
        # Default Layout (Chart 1 visible)
        fig.update_layout(
            template=CHART_TEMPLATE,
            height=700,  # Good height for a single view
            title_text=f"Financial SOC Analysis - {self.symbol}",
            title_x=0.5,
            showlegend=True,
            hovermode="x unified",
            # Margins for text box at top
            margin=dict(t=160, b=50, l=50, r=50),
            legend=dict(
                orientation="h",  # Horizontal legend at bottom
                yanchor="top",
                y=-0.1,
                xanchor="center",
                x=0.5
            )
        )

        # Create Menu Buttons
        updatemenus = [
            dict(
                type="buttons",
                direction="right",
                x=0.5,
                y=1.25,  # Above title
                xanchor="center",
                yanchor="top",
                pad={"r": 10, "t": 10},
                buttons=[
                    # Button 1: Volatility Clustering
                    dict(
                        label="1. Volatility Clustering",
                        method="update",
                        args=[
                            # Visibility: [Chart 1 (1 trace), Chart 2 (2 traces), Chart 3 (4 traces)]
                            {"visible": [True] + [False]*2 + [False]*4},
                            {
                                "title": f"Chart 1: Volatility Clustering - {self.symbol}",
                                "annotations": [self._get_annotation_dict(text_1)],
                                "xaxis": {"title": "Date", "type": "date"},
                                "yaxis": {"title": "Price (USD)", "type": "linear"},
                            }
                        ],
                    ),
                    # Button 2: Power Law
                    dict(
                        label="2. Power Law Proof",
                        method="update",
                        args=[
                            # Visibility
                            {"visible": [False] + [True]*2 + [False]*4},
                            {
                                "title": "Chart 2: Power Law Proof - Fat Tail Analysis",
                                "annotations": [self._get_annotation_dict(text_2)],
                                "xaxis": {"title": "Absolute Returns (Log Scale)", "type": "log"},
                                "yaxis": {"title": "Frequency (Log Scale)", "type": "log"},
                            }
                        ],
                    ),
                    # Button 3: Traffic Light
                    dict(
                        label="3. Traffic Light System",
                        method="update",
                        args=[
                            # Visibility
                            {"visible": [False] + [False]*2 + [True]*4},
                            {
                                "title": "Chart 3: Criticality & Trend - Traffic Light System",
                                "annotations": [self._get_annotation_dict(text_3)],
                                "xaxis": {"title": "Date", "type": "date"},
                                "yaxis": {"title": "Volatility (Std Dev)", "type": "linear"},
                                "yaxis2": {"title": "Price (USD)", "showgrid": False},
                            }
                        ],
                    ),
                ],
            )
        ]

        fig.update_layout(updatemenus=updatemenus)
        
        # Initialize with Chart 1 state (Annotations need to be added explicitly for initial state)
        fig.add_annotation(**self._get_annotation_dict(text_1))
        
        # Set initial visibility (redundant but safe)
        # Traces: 1 (Ch1) + 2 (Ch2) + 4 (Ch3) = 7 total
        # Ch1 visible, others hidden
        for i in range(len(fig.data)):
            fig.data[i].visible = (i == 0)

        print("✓ Dashboard created successfully")
        return fig

    def _get_annotation_dict(self, text: str) -> Dict[str, Any]:
        """Helper to generate consistent annotation dictionary"""
        return dict(
            text=text,
            xref="paper", yref="paper",
            x=0.5, y=1.12,  # Fixed position at top
            xanchor="center", yanchor="bottom",
            showarrow=False,
            font=dict(size=14, color="rgba(255,255,255,0.95)"),
            align="center",
            bordercolor="rgba(100,100,100,0.5)",
            borderwidth=1,
            borderpad=10,
            bgcolor="rgba(30,30,30,0.9)",
            width=900,
        )

    def _add_volatility_clustering_traces(self, fig: go.Figure) -> None:
        """Chart 1 Traces"""
        abs_returns = self.df["abs_returns"]
        abs_returns_norm = (abs_returns - abs_returns.min()) / (abs_returns.max() - abs_returns.min())

        fig.add_trace(
            go.Scatter(
                x=self.df.index,
                y=self.df["close"],
                mode="markers+lines",
                name="Price (Vol Heatmap)",
                line=dict(width=0.75, color="rgba(255,255,255,0.3)"),
                marker=dict(
                    size=4,
                    color=abs_returns_norm,
                    colorscale=[[0, COLORS["stable"]], [0.5, "#FFFF00"], [1, COLORS["high_volatility"]]],
                    showscale=True,
                    colorbar=dict(title="Vol", x=1.02, thickness=10, len=0.5),
                ),
                hovertemplate="Price: $%{y:,.2f}<extra></extra>",
            ),
            secondary_y=False
        )

    def _add_power_law_traces(self, fig: go.Figure) -> None:
        """Chart 2 Traces"""
        abs_returns = self.df["abs_returns"].dropna().values
        mean = abs_returns.mean()
        std = abs_returns.std()
        normal_samples = np.abs(np.random.normal(mean, std, len(abs_returns)))
        
        min_val = max(abs_returns.min(), normal_samples.min(), 1e-8)
        max_val = max(abs_returns.max(), normal_samples.max())
        bins = np.logspace(np.log10(min_val), np.log10(max_val), 40)
        
        actual_hist, _ = np.histogram(abs_returns, bins=bins)
        normal_hist, _ = np.histogram(normal_samples, bins=bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        actual_mask = actual_hist > 0
        normal_mask = normal_hist > 0

        # Trace 1: Actual
        fig.add_trace(
            go.Scatter(
                x=bin_centers[actual_mask],
                y=actual_hist[actual_mask],
                mode="markers",
                name="Actual Returns",
                marker=dict(size=6, color=COLORS["high_volatility"]),
            )
        )
        
        # Trace 2: Theoretical
        fig.add_trace(
            go.Scatter(
                x=bin_centers[normal_mask],
                y=normal_hist[normal_mask],
                mode="lines",
                name="Normal Dist",
                line=dict(width=1.5, color=COLORS["stable"], dash="dash"),
            )
        )

    def _add_criticality_trend_traces(self, fig: go.Figure) -> None:
        """Chart 3 Traces"""
        color_map = {"low": COLORS["low_volatility"], "medium": COLORS["medium_volatility"], "high": COLORS["high_volatility"]}
        colors_list = self.df["vol_zone"].map(color_map)

        # Trace 1: Volatility Bars
        fig.add_trace(
            go.Bar(
                x=self.df.index,
                y=self.df["volatility"],
                name="Volatility",
                marker=dict(color=colors_list, line=dict(width=0)),
            ),
            secondary_y=False,
        )

        # Trace 2: SMA 200
        fig.add_trace(
            go.Scatter(
                x=self.df.index,
                y=self.df["sma_200"],
                name="SMA 200",
                line=dict(width=0.75, color=COLORS["sma_line"]),
            ),
            secondary_y=True,
        )

        # Trace 3: Price
        fig.add_trace(
            go.Scatter(
                x=self.df.index,
                y=self.df["close"],
                name="Price",
                line=dict(width=0.75, color=COLORS["price_line"]),
            ),
            secondary_y=True,
        )

        # Trace 4: Threshold Lines (Simplified to one trace or just lines)
        # For simplicity in this button-view, I'll omit the static lines or add them as layout shapes 
        # But layout shapes are harder to toggle with buttons.
        # I'll add the invisible colorbar trace here to maintain the legend/colorbar logic
        fig.add_trace(
            go.Scatter(
                x=[self.df.index[0]], y=[0],
                mode="markers",
                name="Heatmap Scale",
                marker=dict(
                    size=0,
                    color=[0],
                    colorscale=[[0, COLORS["stable"]], [0.5, "#FFFF00"], [1, COLORS["high_volatility"]]],
                    showscale=True,
                    colorbar=dict(title="Heatmap", x=1.02, thickness=10, len=0.5, y=0.2),
                ),
                showlegend=False,
                hoverinfo="skip"
            ),
            secondary_y=False
        )

    def save_html(self, fig: go.Figure, filename: str = "soc_analysis.html") -> None:
        fig.write_html(filename)
        print(f"✓ Saved visualization to: {filename}")
