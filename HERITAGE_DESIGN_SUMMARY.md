# 🎨 TECTONIQ - Scientific Heritage Design System

**Implementation Date:** December 7, 2025  
**Version:** Beta v8  
**Theme:** Scientific Heritage / Paper Journal

---

## ✅ Changes Completed

### 1. User Limitations Removed

**Before:**
- Free tier: 3 portfolio assets max
- Free tier: 5 simulations per day
- Premium tier: Unlimited

**After:**
- **All users:** Unlimited portfolio assets
- **All users:** Unlimited simulations
- No restrictions for registered users

**Why:** Tier system kept for future premium features (email reports, instant alerts)

---

### 2. Typography Enhancements

**TECTONIQ Header:**
- Font size: `2.8rem` (was: 2rem)
- Letter-spacing: `-1px` (tighter, more impactful)
- Font: Rockwell Std Condensed / Merriweather serif
- Color: `#2C3E50` (Midnight Blue)

**Chart Fonts:**
- Legends: `13px` (was: 11px)
- Axis titles: `14-15px` (was: 12px)
- Tick labels: `12px` (was: 10px)
- Hover labels: `13px` (was: 11px)

---

### 3. Button Styling (Scientific Heritage)

**All buttons now have:**
- Background: `#2C3E50` (Midnight Blue)
- Text: `#F9F7F1` (Cream)
- Font: Merriweather serif
- Border: 2px solid #2C3E50
- Radius: 4px (paper-like)
- Hover: Darker shade (#1a252f) with shadow

**Button Types:**
- **Primary:** Solid Midnight Blue
- **Secondary:** Transparent with Midnight Blue border
- **All buttons:** Serif font, bold weight

---

### 4. Color Palette (Earth Tones)

| Color | Hex | Name | Usage |
|-------|-----|------|-------|
| **Background** | `#F9F7F1` | Alabaster | Main canvas |
| **Secondary BG** | `#E6E1D3` | Parchment | Containers |
| **Primary** | `#2C3E50` | Midnight Blue | Brand, buttons |
| **Text** | `#333333` | Charcoal | Body text |
| **Stable** | `#27AE60` | Forest Green | Low risk |
| **Active** | `#F1C40F` | Muted Gold | Medium risk |
| **High Energy** | `#D35400` | Pumpkin/Ochre | High risk |
| **Critical** | `#C0392B` | Terracotta | Extreme risk |
| **Dormant** | `#95A5A6` | Slate Grey | Downtrend |

---

### 5. Chart Styling (Journal Look)

**Backgrounds:**
- Transparent (`rgba(0,0,0,0)`)
- Shows cream background through charts

**Lines:**
- Thinner (1.2-1.5px) for pen-drawn precision
- Midnight Blue (#2C3E50) for data
- Ochre (#D35400) for references

**Grids:**
- Faint color (#E6E1D3)
- Thin width (0.5px)
- Dotted style

**Watermarks:**
- "TECTONIQ.APP (Beta)" on all charts
- Bottom right corner
- Grey, serif, low opacity

**Simulation Charts:**
- Buy & Hold: Grey dashed (#7F8C8D)
- Strategy: Midnight Blue solid (#2C3E50)
- Drawdowns: Earth-tone fills with transparency

---

## 📊 Typography System

**Font Stack:**
```css
Primary: 'Rockwell Std Condensed' (if available)
Fallback 1: 'Rockwell'
Fallback 2: 'Roboto Slab' (Google Fonts)
Fallback 3: 'Roboto Condensed' (Google Fonts)
```

**Usage:**
- Headings (h1-h6): Rockwell, bold, #2C3E50
- Body text: Rockwell, normal, #333333
- Buttons: Rockwell, bold, serif
- Charts: Merriweather serif for labels

---

## 🎯 Brand Identity

**Name:** TECTONIQ  
**Tagline:** Move Beyond Buy & Hope  
**Positioning:** Market crashes aren't random—they are physics.

**Visual DNA:**
- Warm, academic, timeless
- Physics-based authority
- Journal/research paper aesthetic
- Earth tones, not neon
- Serif typography
- Precise, measured presentation

---

## 🔄 Migration from Old Theme

**Removed:**
- ❌ Dark theme (#0E1117)
- ❌ Neon colors (#00FF00, #FF0000)
- ❌ "SOC Market Seismograph" branding
- ❌ Purple gradients (#667eea, #764ba2)
- ❌ User limitations (portfolio, simulation)
- ❌ Sans-serif body text (Lato)

**Added:**
- ✅ Cream background (#F9F7F1)
- ✅ Earth-tone colors
- ✅ TECTONIQ branding
- ✅ Midnight Blue accents (#2C3E50)
- ✅ Unlimited access for all users
- ✅ Rockwell Condensed typography
- ✅ Transparent chart backgrounds
- ✅ Watermarks on charts

---

## 🧪 Testing Checklist

**Visual:**
- [ ] Cream background throughout
- [ ] All text readable (Charcoal on Cream)
- [ ] Buttons use Midnight Blue
- [ ] Charts show earth-tone colors
- [ ] Serif fonts render correctly
- [ ] TECTONIQ header is large and bold

**Functional:**
- [ ] Unlimited portfolio assets work
- [ ] Unlimited simulations work
- [ ] No error messages about limits
- [ ] Charts render properly
- [ ] Watermarks visible on charts

---

**Design system complete!** 🎨
