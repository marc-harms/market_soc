# Development Notes - SOC Analyzer

## Volatility Threshold Options

The current implementation uses **Option A: Dynamic Quantiles** for the traffic light system in Chart 3. Below are alternative implementations for future consideration.

---

### Current Implementation: Option A - Dynamic Quantiles (Adaptive)

**Status:** ✅ IMPLEMENTED

**Logic:**
- Green Zone: Bottom 33.33% of volatility values
- Orange Zone: Middle 33.34%
- Red Zone: Top 33.33%

**Pros:**
- Automatically adapts to different market regimes
- Works across different assets without recalibration
- Relative to historical behavior

**Cons:**
- Thresholds change with each analysis
- Not comparable across different time periods

**Implementation Location:** `modules/soc_metrics.py` → `_calculate_volatility_thresholds()`

```python
# Current implementation
low_threshold = volatility.quantile(VOLATILITY_LOW_PERCENTILE / 100)
high_threshold = volatility.quantile(VOLATILITY_HIGH_PERCENTILE / 100)
```

---

### Option B - Fixed Statistical Thresholds (Standard Deviations)

**Status:** 🔲 NOT IMPLEMENTED (Alternative)

**Logic:**
- Green Zone: Volatility below mean + 1σ
- Orange Zone: Volatility between mean + 1σ and mean + 2σ
- Red Zone: Volatility above mean + 2σ

**Pros:**
- Based on statistical significance
- Consistent interpretation across analyses
- Highlights true outliers

**Cons:**
- May not adapt well to regime changes
- Assumes normal distribution of volatility

**Proposed Implementation:**
```python
def _calculate_volatility_thresholds_option_b(self) -> Tuple[float, float]:
    """Option B: Statistical thresholds based on standard deviations"""
    volatility = self.df["volatility"].dropna()
    mean_vol = volatility.mean()
    std_vol = volatility.std()
    
    low_threshold = mean_vol + (1 * std_vol)
    high_threshold = mean_vol + (2 * std_vol)
    
    return low_threshold, high_threshold
```

**Configuration Changes Needed:**
```python
# In config/settings.py
VOLATILITY_METHOD = "statistical"  # Options: "quantile", "statistical", "fixed"
VOLATILITY_STD_LOW = 1.0
VOLATILITY_STD_HIGH = 2.0
```

---

### Option C - Custom Fixed Values (Absolute Thresholds)

**Status:** 🔲 NOT IMPLEMENTED (Alternative)

**Logic:**
- User defines absolute volatility values
- Example for BTC: Green < 0.02 (2%), Orange 0.02-0.05, Red > 0.05 (5%)

**Pros:**
- Precise control for known assets
- Comparable across all time periods
- Can incorporate domain expertise

**Cons:**
- Requires manual calibration per asset
- Not portable across different instruments
- May become outdated in changing markets

**Proposed Implementation:**
```python
def _calculate_volatility_thresholds_option_c(
    self, 
    low: float = 0.02, 
    high: float = 0.05
) -> Tuple[float, float]:
    """Option C: Fixed absolute thresholds"""
    return low, high
```

**Configuration Changes Needed:**
```python
# In config/settings.py
VOLATILITY_METHOD = "fixed"
VOLATILITY_FIXED_LOW = 0.02  # 2% daily volatility
VOLATILITY_FIXED_HIGH = 0.05  # 5% daily volatility

# Asset-specific presets
VOLATILITY_PRESETS = {
    "BTCUSDT": {"low": 0.02, "high": 0.05},
    "ETHUSDT": {"low": 0.025, "high": 0.06},
    "AAPL": {"low": 0.01, "high": 0.03},  # For future equity support
}
```

---

## Implementation Roadmap for Multi-Method Support

To enable switching between all three options:

### 1. Update `config/settings.py`
```python
VOLATILITY_METHOD = "quantile"  # Options: "quantile", "statistical", "fixed"

# Quantile settings (Option A)
VOLATILITY_LOW_PERCENTILE = 33.33
VOLATILITY_HIGH_PERCENTILE = 66.67

# Statistical settings (Option B)
VOLATILITY_STD_LOW = 1.0
VOLATILITY_STD_HIGH = 2.0

# Fixed settings (Option C)
VOLATILITY_FIXED_LOW = 0.02
VOLATILITY_FIXED_HIGH = 0.05
```

### 2. Refactor `SOCMetricsCalculator` class
```python
def _calculate_volatility_thresholds(self) -> Tuple[float, float]:
    """Calculate thresholds based on configured method"""
    method = settings.VOLATILITY_METHOD
    
    if method == "quantile":
        return self._thresholds_quantile()
    elif method == "statistical":
        return self._thresholds_statistical()
    elif method == "fixed":
        return self._thresholds_fixed()
    else:
        raise ValueError(f"Unknown method: {method}")
```

### 3. Add CLI argument
```python
parser.add_argument(
    "--threshold-method",
    type=str,
    choices=["quantile", "statistical", "fixed"],
    default="quantile",
    help="Volatility threshold calculation method"
)
```

---

## Recommendation

**For Phase 1:** Keep Option A (quantile-based) as it provides:
- Best out-of-the-box experience
- No manual calibration required
- Adaptability across different assets

**For Phase 2:** Implement all three options with CLI flag to enable user choice based on use case:
- **Traders:** Option C (fixed values based on their strategy)
- **Researchers:** Option B (statistical rigor)
- **General Use:** Option A (adaptive)

---

## Other Technical Notes

### Power Law Distribution (Chart 2)
- Currently using matched normal distribution (Option A: same μ and σ as actual data)
- This clearly demonstrates deviation from expected normal behavior
- Alternative: Could overlay multiple theoretical distributions (Lévy, Cauchy)

### Future Enhancements
- [ ] Add regime detection (trending vs. mean-reverting)
- [ ] Implement alert system for criticality breaches
- [ ] Add export to CSV/Excel functionality
- [ ] Multi-asset comparison view
- [ ] Real-time WebSocket streaming mode

---

*Last Updated: Phase 1 Implementation*

