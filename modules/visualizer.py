"""
Visualization Module
Creates interactive Plotly charts for SOC analysis
"""

from typing import Optional

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
        Create complete interactive dashboard with all three charts

        Returns:
            Plotly Figure object with subplots
        """
        print("Generating interactive visualizations...")

        # Create subplot figure with 3 rows
        fig = make_subplots(
            rows=3,
            cols=1,
            row_heights=[0.33, 0.33, 0.33],
            subplot_titles=(
                f"Chart 1: Volatility Clustering - {self.symbol}",
                f"Chart 2: Power Law Proof - Fat Tail Analysis",
                f"Chart 3: Criticality & Trend - Traffic Light System",
            ),
            vertical_spacing=0.25,  # Increased spacing for text boxes ABOVE charts
            specs=[
                [{"secondary_y": False}],
                [{"secondary_y": False}],
                [{"secondary_y": True}],  # Chart 3 needs secondary y-axis
            ],
        )

        # Chart 1: Volatility Clustering
        self._add_volatility_clustering_chart(fig, row=1)

        # Chart 2: Power Law Proof
        self._add_power_law_chart(fig, row=2)

        # Chart 3: Criticality & Trend
        self._add_criticality_trend_chart(fig, row=3)

        # Update overall layout
        fig.update_layout(
            height=CHART_HEIGHT * 4.5,  # Increased total height for spacing
            template=CHART_TEMPLATE,
            showlegend=True,
            title_text=f"Financial SOC Analysis - {self.symbol}",
            title_font_size=20,
            hovermode="x unified",
            margin=dict(t=150, b=50, r=200),  # Increased top margin, reduced bottom
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.98,
                xanchor="left",
                x=1.02,
                bgcolor="rgba(30,30,30,0.8)",
                bordercolor="rgba(255,255,255,0.3)",
                borderwidth=0.75,
                font=dict(size=10),
            ),
        )
        
        # Add explanatory text annotations ABOVE each chart
        # Chart 1 explanation (Above Chart 1)
        fig.add_annotation(
            text=(
                "<b>1. Volatility Clustering (The 'Sandpile'):</b><br>"
                "In SOC systems, extreme events do not occur in isolation. They come in waves (clusters). "
                "The color coding shows phases where the system is 'working'."
            ),
            xref="paper", yref="paper",
            x=0.5, y=1.07,  # Positioned above Chart 1 title
            xanchor="center", yanchor="bottom",
            showarrow=False,
            font=dict(size=13, color="rgba(255,255,255,0.95)"),
            align="left",
            bordercolor="rgba(100,100,100,0.5)",
            borderwidth=1,
            borderpad=10,
            bgcolor="rgba(20,20,20,0.8)",
            width=800,
        )
        
        # Chart 2 explanation (Above Chart 2)
        fig.add_annotation(
            text=(
                "<b>2. The Power Curve (Log/Log Proof):</b><br>"
                "This is the mathematical 'fingerprint'. Red points = Real Bitcoin data. Blue line = Normal distribution.<br>"
                "The straight line of red points proves the 'Fat Tails' (extremely high probability for Black Swans)."
            ),
            xref="paper", yref="paper",
            x=0.5, y=0.64,  # Positioned above Chart 2 title
            xanchor="center", yanchor="bottom",
            showarrow=False,
            font=dict(size=13, color="rgba(255,255,255,0.95)"),
            align="left",
            bordercolor="rgba(100,100,100,0.5)",
            borderwidth=1,
            borderpad=10,
            bgcolor="rgba(20,20,20,0.8)",
            width=800,
        )
        
        # Chart 3 explanation (Above Chart 3)
        fig.add_annotation(
            text=(
                "<b>3. System Criticality & Trading Signals:</b><br>"
                "Combine SOC (Volatility) with Trend (SMA 200) to find signals.<br>"
                "⚠ <b>Rule:</b> Red phases mean instability. If price during red phase is below SMA 200 (Yellow line) = Crash risk (Sell).<br>"
                "If above SMA 200 = Parabolic rally (Caution/Hold).<br>"
                "✓ <b>Rule:</b> Green phases above SMA 200 are often good entries ('Accumulation')."
            ),
            xref="paper", yref="paper",
            x=0.5, y=0.23,  # Positioned above Chart 3 title
            xanchor="center", yanchor="bottom",
            showarrow=False,
            font=dict(size=13, color="rgba(255,255,255,0.95)"),
            align="left",
            bordercolor="rgba(100,100,100,0.5)",
            borderwidth=1,
            borderpad=10,
            bgcolor="rgba(20,20,20,0.8)",
            width=800,
        )

        print("✓ Dashboard created successfully")
        return fig

    def _add_volatility_clustering_chart(self, fig: go.Figure, row: int) -> None:
        """
        Chart 1: Price time series with color-coded volatility magnitude
        Blue = Stable, Red = High Volatility
        """
        # Create color scale based on absolute returns
        abs_returns = self.df["abs_returns"]
        
        # Normalize to 0-1 for colorscale
        abs_returns_norm = (abs_returns - abs_returns.min()) / (
            abs_returns.max() - abs_returns.min()
        )

        # Store abs_returns_norm as instance variable for Chart 3 colorbar
        self.abs_returns_norm = abs_returns_norm
        
        # Create scatter plot with color gradient (NO colorbar here)
        fig.add_trace(
            go.Scatter(
                x=self.df.index,
                y=self.df["close"],
                mode="markers+lines",
                name="Price (color = volatility)",
                legendgroup="chart1",
                line=dict(width=0.75, color="rgba(255,255,255,0.3)"),
                marker=dict(
                    size=3.75,
                    color=abs_returns_norm,
                    colorscale=[
                        [0, COLORS["stable"]],  # Blue for stable
                        [0.5, "#FFFF00"],  # Yellow for medium
                        [1, COLORS["high_volatility"]],  # Red for high volatility
                    ],
                    showscale=False,  # No colorbar on Chart 1
                ),
                hovertemplate="<b>Date:</b> %{x}<br>"
                + "<b>Price:</b> $%{y:,.2f}<br>"
                + "<extra></extra>",
            ),
            row=row,
            col=1,
        )

        # Update axes
        fig.update_xaxes(title_text="Date", row=row, col=1)
        fig.update_yaxes(title_text="Price (USD)", row=row, col=1)

    def _add_power_law_chart(self, fig: go.Figure, row: int) -> None:
        """
        Chart 2: Log-Log histogram showing power law distribution
        Compares actual returns vs theoretical normal distribution
        """
        abs_returns = self.df["abs_returns"].dropna().values

        # Generate theoretical normal with same mean/std (Option A)
        mean = abs_returns.mean()
        std = abs_returns.std()
        normal_samples = np.abs(np.random.normal(mean, std, len(abs_returns)))

        # Create log bins
        min_val = max(abs_returns.min(), normal_samples.min(), 1e-8)
        max_val = max(abs_returns.max(), normal_samples.max())
        bins = np.logspace(np.log10(min_val), np.log10(max_val), 40)

        # Calculate histograms
        actual_hist, _ = np.histogram(abs_returns, bins=bins)
        normal_hist, _ = np.histogram(normal_samples, bins=bins)

        # Bin centers for plotting
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Filter out zeros for log scale
        actual_mask = actual_hist > 0
        normal_mask = normal_hist > 0

        # Actual returns (Fat Tail)
        fig.add_trace(
            go.Scatter(
                x=bin_centers[actual_mask],
                y=actual_hist[actual_mask],
                mode="markers",
                name="Actual Returns (Fat Tail)",
                legendgroup="chart2",
                marker=dict(size=6, color=COLORS["high_volatility"]),
                hovertemplate="<b>Return:</b> %{x:.6f}<br>"
                + "<b>Frequency:</b> %{y}<br>"
                + "<extra></extra>",
            ),
            row=row,
            col=1,
        )

        # Theoretical Normal
        fig.add_trace(
            go.Scatter(
                x=bin_centers[normal_mask],
                y=normal_hist[normal_mask],
                mode="lines",
                name="Theoretical Normal",
                legendgroup="chart2",
                line=dict(width=1.5, color=COLORS["stable"], dash="dash"),
                hovertemplate="<b>Return:</b> %{x:.6f}<br>"
                + "<b>Frequency:</b> %{y}<br>"
                + "<extra></extra>",
            ),
            row=row,
            col=1,
        )

        # Update axes to log scale
        fig.update_xaxes(
            title_text="Absolute Returns (Log Scale)",
            type="log",
            row=row,
            col=1,
        )
        fig.update_yaxes(
            title_text="Frequency (Log Scale)",
            type="log",
            row=row,
            col=1,
        )

    def _add_criticality_trend_chart(self, fig: go.Figure, row: int) -> None:
        """
        Chart 3: Volatility bars with traffic light colors + Price/SMA overlay
        Primary Y-axis: Volatility bars
        Secondary Y-axis: Price and SMA 200
        """
        # Define color mapping
        color_map = {
            "low": COLORS["low_volatility"],
            "medium": COLORS["medium_volatility"],
            "high": COLORS["high_volatility"],
        }

        # Create colored bars for volatility
        colors_list = self.df["vol_zone"].map(color_map)

        fig.add_trace(
            go.Bar(
                x=self.df.index,
                y=self.df["volatility"],
                name="Rolling Volatility (30d)",
                legendgroup="chart3",
                marker=dict(
                    color=colors_list,
                    line=dict(width=0),
                ),
                hovertemplate="<b>Date:</b> %{x}<br>"
                + "<b>Volatility:</b> %{y:.6f}<br>"
                + "<extra></extra>",
            ),
            row=row,
            col=1,
            secondary_y=False,
        )

        # Add SMA 200 on secondary y-axis
        fig.add_trace(
            go.Scatter(
                x=self.df.index,
                y=self.df["sma_200"],
                name="SMA 200 (Trend)",
                legendgroup="chart3",
                line=dict(width=0.75, color=COLORS["sma_line"]),
                hovertemplate="<b>Date:</b> %{x}<br>"
                + "<b>SMA 200:</b> $%{y:,.2f}<br>"
                + "<extra></extra>",
            ),
            row=row,
            col=1,
            secondary_y=True,
        )

        # Add Price on secondary y-axis
        fig.add_trace(
            go.Scatter(
                x=self.df.index,
                y=self.df["close"],
                name="Price",
                legendgroup="chart3",
                line=dict(width=0.75, color=COLORS["price_line"]),
                hovertemplate="<b>Date:</b> %{x}<br>"
                + "<b>Price:</b> $%{y:,.2f}<br>"
                + "<extra></extra>",
            ),
            row=row,
            col=1,
            secondary_y=True,
        )
        
        # Add invisible trace for the heatmap colorbar (from Chart 1)
        # This shows the colorbar next to Chart 3 where it logically belongs
        fig.add_trace(
            go.Scatter(
                x=[self.df.index[0]],
                y=[self.df["volatility"].iloc[0]],
                mode="markers",
                name="Volatility Heatmap Scale",
                legendgroup="chart3",
                marker=dict(
                    size=0.1,  # Nearly invisible
                    color=[0.5],
                    colorscale=[
                        [0, COLORS["stable"]],  # Blue for stable
                        [0.5, "#FFFF00"],  # Yellow for medium
                        [1, COLORS["high_volatility"]],  # Red for high volatility
                    ],
                    colorbar=dict(
                        title="Volatility<br>Heatmap<br>(Chart 1)",
                        x=1.18,
                        y=0.15,
                        len=0.25,
                        thickness=12,
                    ),
                    showscale=True,
                ),
                showlegend=False,  # Don't show in legend
                hoverinfo="skip",
            ),
            row=row,
            col=1,
            secondary_y=False,
        )

        # Add horizontal threshold lines
        if self.vol_low_threshold is not None:
            fig.add_hline(
                y=self.vol_low_threshold,
                line=dict(
                    color=COLORS["low_volatility"],
                    width=0.75,
                    dash="dot",
                ),
                annotation_text="Low Threshold",
                annotation_position="right",
                row=row,
                col=1,
                secondary_y=False,
            )

        if self.vol_high_threshold is not None:
            fig.add_hline(
                y=self.vol_high_threshold,
                line=dict(
                    color=COLORS["high_volatility"],
                    width=0.75,
                    dash="dot",
                ),
                annotation_text="High Threshold",
                annotation_position="right",
                row=row,
                col=1,
                secondary_y=False,
            )

        # Update axes
        fig.update_xaxes(title_text="Date", row=row, col=1)
        fig.update_yaxes(
            title_text="Volatility (Std Dev)",
            row=row,
            col=1,
            secondary_y=False,
        )
        fig.update_yaxes(
            title_text="Price (USD)",
            row=row,
            col=1,
            secondary_y=True,
        )

    def save_html(self, fig: go.Figure, filename: str = "soc_analysis.html") -> None:
        """
        Save figure as standalone HTML file

        Args:
            fig: Plotly figure object
            filename: Output filename
        """
        fig.write_html(filename)
        print(f"✓ Saved visualization to: {filename}")

