"""
Visualization Module
Creates standalone HTML dashboard with custom CSS/JS layout
Matches the specific German layout provided by user
"""

import json
from typing import Optional

import numpy as np
import pandas as pd
from config.settings import COLORS

class PlotlyVisualizer:
    """
    Creates a custom HTML dashboard with specific styling and layout.
    Injects Python-calculated data into a template matching the user's requirements.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        symbol: str = "BTCUSDT",
        vol_low_threshold: Optional[float] = None,
        vol_high_threshold: Optional[float] = None,
    ) -> None:
        self.df = df
        self.symbol = symbol
        self.vol_low_threshold = vol_low_threshold
        self.vol_high_threshold = vol_high_threshold

    def create_interactive_dashboard(self) -> str:
        """
        Generates the complete HTML string with embedded data.
        Returns the HTML content string.
        """
        print("Generating custom HTML dashboard...")
        
        # Prepare Data for Injection
        # We need to convert DataFrame columns to JSON-serializable lists
        
        # 1. Main Data (Dates, Prices, Returns, Magnitudes)
        # Filter out NaN or invalid data first
        valid_df = self.df.dropna(subset=['close', 'returns', 'abs_returns', 'sma_200', 'volatility'])
        
        data_dates = valid_df.index.strftime('%Y-%m-%d').tolist()
        data_prices = valid_df['close'].tolist()
        data_returns = valid_df['returns'].tolist()
        data_magnitudes = valid_df['abs_returns'].tolist()
        data_sma = valid_df['sma_200'].tolist()
        data_volatility = valid_df['volatility'].tolist()
        
        # Prepare Colors based on Volatility Zones
        # Map 'low', 'medium', 'high' to actual colors
        color_map = {
            "low": "#00ff00",      # Green
            "medium": "#ffa500",   # Orange
            "high": "#ff0000",     # Red
        }
        # Default to orange if zone missing
        data_colors = [color_map.get(zone, "#ffa500") for zone in valid_df['vol_zone']]
        json_colors = json.dumps(data_colors)

        # Convert to JSON strings
        json_dates = json.dumps(data_dates)
        json_prices = json.dumps(data_prices)
        json_returns = json.dumps(data_returns)
        json_magnitudes = json.dumps(data_magnitudes)
        json_sma = json.dumps(data_sma)
        json_volatility = json.dumps(data_volatility)
        
        # Thresholds (default to 0 if None)
        vol_low = self.vol_low_threshold if self.vol_low_threshold is not None else 0
        vol_high = self.vol_high_threshold if self.vol_high_threshold is not None else 0

        # HTML Template
        html_content = f"""<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bitcoin SOC & Power Law Analyse</title>
    <!-- Plotly.js für interaktive Grafiken laden -->
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background-color: #111;
            color: #eee;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            font-weight: 300;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }}
        p.subtitle {{
            line-height: 1.6;
            color: #aaa;
            max-width: 800px;
            margin: 0 auto 30px auto;
            text-align: center;
        }}
        .chart-container {{
            background-color: #1e1e1e;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 40px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.5);
            border: 1px solid #333;
        }}
        .status-box {{
            text-align: center;
            font-size: 1em;
            color: #aaa;
            margin: 20px 0;
            padding: 15px;
            border: 1px solid #333;
            border-radius: 8px;
            background: #222;
        }}
        .source-badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
            margin-left: 10px;
        }}
        .badge-binance {{ background-color: #F3BA2F; color: #000; }}
        
        .explanation {{
            background: #222;
            border-left: 4px solid #ff9800;
            padding: 15px;
            margin-bottom: 15px;
            font-size: 0.95em;
            line-height: 1.5;
            border-radius: 0 4px 4px 0;
        }}
        strong {{ color: #fff; }}
        .footer {{
            text-align: center;
            font-size: 0.8em;
            color: #666;
            margin-top: 50px;
        }}
        
        /* Controls für Chart 3 */
        .controls {{
            margin-bottom: 10px;
            text-align: right;
            background: #2a2a2a;
            padding: 10px;
            border-radius: 4px;
            display: flex;
            justify-content: flex-end;
            gap: 20px;
        }}
        .toggle-label {{
            cursor: pointer;
            font-size: 0.9em;
            color: #ddd;
            user-select: none;
            display: flex;
            align-items: center;
        }}
        .toggle-label input {{
            margin-right: 8px;
        }}
    </style>
</head>
<body>

<div class="container">
    <h1>Bitcoin: Self-Organized Criticality (SOC)</h1>
    <p class="subtitle">
        Diese Analyse visualisiert die Signaturen komplexer Systeme:<br>
        1. Zeitliche Instabilität, 2. Power Laws (Fat Tails) und 3. System-Kritikalität mit Trendsignalen.
    </p>
    
    <div id="status" class="status-box" style="border-color: #4CAF50; color: #4CAF50;">
        Daten erfolgreich berechnet durch Python Backend: <span class="source-badge badge-binance">Binance API</span>
    </div>

    <div id="content-area">
        
        <!-- Chart 1: Zeitreihe -->
        <div class="explanation">
            <strong>1. Volatility Clustering (Der "Sandhaufen"):</strong><br>
            In SOC-Systemen treten extreme Ereignisse nicht isoliert auf. Sie kommen in Wellen (Clustern). 
            Die Einfärbung zeigt Phasen, in denen das System "arbeitet".
        </div>
        <div id="chart-time" class="chart-container" style="height: 500px;"></div>

        <!-- Chart 2: Power Law -->
        <div class="explanation">
            <strong>2. Die Power Curve (Log/Log Beweis):</strong><br>
            Dies ist der mathematische "Fingerabdruck". 
            <span style="color:#00ccff">Blaue Punkte</span> = Reale Bitcoin-Daten. 
            <span style="color:#00ff00">Grüne Linie</span> = Normalverteilung.<br>
            Die gerade Linie der blauen Punkte beweist die "Fat Tails" (extrem hohe Wahrscheinlichkeit für Black Swans).
        </div>
        <div id="chart-log" class="chart-container" style="height: 600px;"></div>

        <!-- Chart 3: Kritikalität -->
        <div class="explanation">
            <strong>3. System-Kritikalität & Handelssignale:</strong><br>
            Kombiniere SOC (Volatilität) mit Trend (SMA 200), um Signale zu finden.<br>
            <span style="color:#ffa500">⚠ Regel:</span> <strong>Rote Phasen</strong> bedeuten Instabilität. 
            Wenn der Preis während einer roten Phase <strong>unter</strong> dem SMA 200 (Gelbe Linie) ist = <strong>Crash-Gefahr (Verkauf)</strong>.
            Wenn er <strong>über</strong> dem SMA 200 ist = <strong>Parabolische Rallye (Vorsicht/Halten)</strong>.<br>
            <span style="color:#00ff00">✓ Regel:</span> <strong>Grüne Phasen</strong> über dem SMA 200 sind oft gute Einstiege ("Accumulation").
        </div>
        
        <div class="controls">
            <label class="toggle-label">
                <input type="checkbox" id="overlayToggle" onchange="updateChart3Visibility()" checked> 
                Bitcoin-Kurs (Weiß)
            </label>
            <label class="toggle-label">
                <input type="checkbox" id="smaToggle" onchange="updateChart3Visibility()" checked> 
                SMA 200 Trendlinie (Gelb)
            </label>
        </div>
        <div id="chart-crit" class="chart-container" style="height: 500px;"></div>

    </div>

    <div class="footer">
        Visualisierung: Plotly.js | Analyse basierend auf SOC-Theorie | Generated by soc_analyzer.py
    </div>
</div>

<script>
    // --- DATEN VOM PYTHON BACKEND INJIZIERT ---
    const globalDates = {json_dates};
    const globalPrices = {json_prices};
    const globalReturns = {json_returns};
    const globalMagnitudes = {json_magnitudes};
    const globalSMA = {json_sma};
    const globalVol = {json_volatility};
    const globalColors = {json_colors};
    
    // Thresholds from Python
    const globalVolLow = {vol_low};
    const globalVolHigh = {vol_high};

    // --- LOGIK & RENDERING ---

    function init() {{
        renderCharts();
    }}

    // Globale Funktion für Checkbox-Change
    window.updateChart3Visibility = function() {{
        const showPrice = document.getElementById('overlayToggle').checked;
        const showSMA = document.getElementById('smaToggle').checked;
        drawChart3(showPrice, showSMA);
    }};

    function drawChart3(showOverlay, showSMA) {{
        // Trace 1: Kritikalität (Bar Chart)
        // Farben kommen direkt aus Python (basierend auf korrekten Quantilen)
        
        const traceCrit = {{
            x: globalDates, 
            y: globalVol, 
            type: 'bar', 
            marker: {{
                color: globalColors,
                // colorscale nicht nötig wenn color array explizit ist
            }},
            name: 'Kritikalität (Vol)',
            hovertemplate: '%{{x|%d.%m.%Y}}: %{{y:.4f}}<extra></extra>'
        }};

        const data = [traceCrit];
        const layoutCrit = {{
            title: '3. System-Kritikalität & Trendanalyse',
            paper_bgcolor: 'rgba(0,0,0,0)', 
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ color: '#ddd' }}, 
            margin: {{ t: 40, r: 50, l: 60, b: 40 }},
            xaxis: {{ title: 'Zeit', gridcolor: '#333' }},
            yaxis: {{ title: 'Volatilität (StdDev)', gridcolor: '#333' }},
            bargap: 0,
            showlegend: true,
            legend: {{ x: 0, y: 1.1, orientation: 'h' }},
            shapes: [
                // Low Threshold Line (Green/Orange boundary)
                {{
                    type: 'line',
                    xref: 'paper', x0: 0, x1: 1,
                    yref: 'y', y0: globalVolLow, y1: globalVolLow,
                    line: {{ color: '#00ff00', width: 1, dash: 'dot' }}
                }},
                // High Threshold Line (Orange/Red boundary)
                {{
                    type: 'line',
                    xref: 'paper', x0: 0, x1: 1,
                    yref: 'y', y0: globalVolHigh, y1: globalVolHigh,
                    line: {{ color: '#ff0000', width: 1, dash: 'dot' }}
                }}
            ],
            annotations: [
                {{
                    xref: 'paper', x: 1, y: globalVolLow, yref: 'y',
                    text: 'Low', showarrow: false, 
                    xanchor: 'left', yanchor: 'middle',
                    font: {{ color: '#00ff00', size: 10 }}
                }},
                {{
                    xref: 'paper', x: 1, y: globalVolHigh, yref: 'y',
                    text: 'High', showarrow: false, 
                    xanchor: 'left', yanchor: 'middle',
                    font: {{ color: '#ff0000', size: 10 }}
                }}
            ]
        }};

        // Optionale Achse hinzufügen
        if (showOverlay || showSMA) {{
            layoutCrit.yaxis2 = {{
                title: 'Preis (Log)',
                type: 'log',
                overlaying: 'y',
                side: 'right',
                gridcolor: 'rgba(255,255,255,0.1)'
            }};
        }}

        // Trace 2: Preis Overlay
        if (showOverlay) {{
            const tracePrice = {{
                x: globalDates,
                y: globalPrices,
                type: 'scatter',
                mode: 'lines',
                name: 'BTC Preis',
                line: {{ color: '#ffffff', width: 1 }},
                yaxis: 'y2', 
                opacity: 0.8
            }};
            data.push(tracePrice);
        }}

        // Trace 3: SMA 200 Trend
        if (showSMA) {{
            const traceSMA = {{
                x: globalDates,
                y: globalSMA,
                type: 'scatter',
                mode: 'lines',
                name: 'SMA 200 (Trend)',
                line: {{ color: '#ffff00', width: 2 }}, // Gelb
                yaxis: 'y2',
                opacity: 1.0
            }};
            data.push(traceSMA);
        }}

        Plotly.newPlot('chart-crit', data, layoutCrit, {{responsive: true}});
    }}

    function renderCharts() {{
        // CHART 1: Zeitreihe
        const traceTime = {{
            x: globalDates, y: globalPrices, mode: 'markers+lines',
            marker: {{ color: globalMagnitudes, colorscale: 'Turbo', size: 2 }},
            line: {{ width: 1, color: 'rgba(255,255,255,0.2)' }}, 
            type: 'scatter', name: 'Preis'
        }};
        const layoutTime = {{
            title: '1. Bitcoin Preis & Instabilität',
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ color: '#ddd' }}, margin: {{ t: 40, r: 20, l: 60, b: 40 }},
            xaxis: {{ title: 'Zeit', gridcolor: '#333' }},
            yaxis: {{ title: 'Preis (Log)', type: 'log', gridcolor: '#333' }}
        }};
        Plotly.newPlot('chart-time', [traceTime], layoutTime, {{responsive: true}});

        // CHART 2: Power Law (Log/Log)
        // Berechnung der Histogramm-Daten in JS für flüssige Darstellung, 
        // oder wir könnten Python-berechnete Histogramme injizieren. 
        // Hier replizieren wir die JS Logik mit den Python-Daten.
        
        const magnitudes = globalMagnitudes;
        const minVal = Math.min(...magnitudes.filter(m => m > 0)); // Filter 0
        const maxVal = Math.max(...magnitudes);
        const numBins = 50;
        
        const logMin = Math.log10(minVal);
        const logMax = Math.log10(maxVal);
        const logBins = [];
        for (let i = 0; i <= numBins; i++) logBins.push(Math.pow(10, logMin + (i * (logMax - logMin) / numBins)));

        const counts = new Array(numBins).fill(0);
        for (let m of magnitudes) {{
            if (m <= 0) continue;
            for (let i = 0; i < numBins; i++) {{
                if (m >= logBins[i] && m < logBins[i+1]) {{ counts[i]++; break; }}
            }}
        }}

        const density = [];
        const xPoints = [];
        const totalPoints = magnitudes.length;
        for (let i = 0; i < numBins; i++) {{
            if (counts[i] > 0) {{
                const binWidth = logBins[i+1] - logBins[i];
                const center = Math.sqrt(logBins[i] * logBins[i+1]);
                density.push(counts[i] / (totalPoints * binWidth));
                xPoints.push(center);
            }}
        }}

        // Normalverteilung (theoretisch) berechnen
        // Wir nutzen Mittelwert/Varianz der Returns
        const returns = globalReturns;
        const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
        const variance = returns.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / returns.length;
        const sigma = Math.sqrt(variance);
        
        const normalY = xPoints.map(x => {{
            return (2 / (sigma * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow(x / sigma, 2));
        }});

        const traceReal = {{ x: xPoints, y: density, mode: 'markers', type: 'scatter', name: 'Reale Daten', marker: {{ color: '#00ccff', size: 6 }} }};
        const traceNormal = {{ x: xPoints, y: normalY, mode: 'lines', type: 'scatter', name: 'Normalverteilung', line: {{ color: '#00ff00', dash: 'dash', width: 2 }} }};

        const layoutLog = {{
            title: '2. Power Curve Beweis (Log/Log)',
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ color: '#ddd' }}, margin: {{ t: 40, r: 20, l: 60, b: 60 }},
            xaxis: {{ type: 'log', title: 'Änderung (Log)', gridcolor: '#333' }},
            yaxis: {{ type: 'log', title: 'Häufigkeit', gridcolor: '#333' }},
            annotations: [{{
                x: Math.log10(xPoints[xPoints.length-3] || 0.1), y: Math.log10(density[density.length-3] || 0.001),
                xref: 'x', yref: 'y', text: 'Fat Tails', showarrow: true, arrowhead: 2, ax: -40, ay: -40, font: {{color: 'red'}}
            }}]
        }};
        Plotly.newPlot('chart-log', [traceReal, traceNormal], layoutLog, {{responsive: true}});

        // CHART 3: Kritikalität
        drawChart3(true, true);
    }}

    // Start
    init();

</script>

</body>
</html>
"""
        return html_content

    def save_html(self, fig: str, filename: str = "soc_analysis.html") -> None:
        """
        Save HTML content to file.
        Note: The 'fig' argument here is actually the HTML string, keeping signature compatible.
        """
        with open(filename, "w", encoding="utf-8") as f:
            f.write(fig)
        print(f"✓ Saved visualization to: {filename}")
