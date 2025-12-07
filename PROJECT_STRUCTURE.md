# 📁 SOC Market Seismograph - Project Structure

**Last Updated:** Dec 7, 2025  
**Version:** Beta v8 (Supabase Multi-User)

---

## 🎯 Working Directory

**ALL development happens in:**
```
/home/marc/Projects/Market Analysis - SOC - SAM/soc-app/
```

This is the **single source of truth** for the application.

---

## 📂 Root Directory Structure

```
Market Analysis - SOC - SAM/
├── soc-app/                      ← MAIN APPLICATION FOLDER
│   ├── .streamlit/
│   │   ├── config.toml           ← Streamlit configuration
│   │   ├── secrets.toml          ← Supabase credentials (gitignored)
│   │   └── secrets.toml.template ← Template for secrets
│   ├── assets/
│   │   └── logo-soc.png          ← App logo
│   ├── data/                     ← Cached market data (CSV files)
│   ├── __pycache__/              ← Python cache
│   ├── app.py                    ← MAIN APPLICATION ENTRY POINT
│   ├── auth_manager.py           ← Supabase authentication & user management
│   ├── config.py                 ← App configuration & constants
│   ├── logic.py                  ← SOC analysis logic & calculations
│   ├── ui_auth.py                ← Authentication UI & header
│   ├── ui_detail.py              ← Asset detail/deep dive UI
│   ├── ui_simulation.py          ← DCA simulation UI
│   ├── requirements.txt          ← Python dependencies
│   └── README.md                 ← App-specific documentation
│
├── venv/                         ← Python virtual environment
├── tests/                        ← Test files
├── assets/                       ← Old assets folder (can be removed)
├── data/                         ← Old data cache (can be removed)
├── README.md                     ← Project README
├── SUPABASE_SETUP.md            ← Supabase setup instructions
├── PROJECT_STRUCTURE.md         ← This file
└── .git/                        ← Git repository
```

---

## 🗂️ Application Architecture

### Core Files

| File | Purpose | Key Functions |
|------|---------|---------------|
| `app.py` | Main application entry point | `main()`, session state management, routing |
| `logic.py` | SOC analysis engine | `SOCAnalyzer`, `DataFetcher`, calculations |
| `auth_manager.py` | User authentication & management | `login()`, `signup()`, portfolio management |
| `config.py` | Configuration & constants | Theme CSS, tickers, disclaimers |

### UI Modules

| File | Purpose | Key Components |
|------|---------|----------------|
| `ui_auth.py` | Authentication & header | Login/signup forms, header with search |
| `ui_detail.py` | Asset analysis display | Deep dive charts, regime analysis |
| `ui_simulation.py` | DCA simulation | Backtest UI, strategy comparison |

---

## 🔧 Key Configuration Files

### `.streamlit/secrets.toml` (gitignored)
```toml
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-anon-key"
```

### `.streamlit/config.toml`
```toml
[theme]
primaryColor = "#667eea"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#212529"
font = "sans serif"
```

---

## 📦 Dependencies (`requirements.txt`)

```
streamlit>=1.30.0
pandas>=2.0.0
numpy>=1.24.0
yfinance>=0.2.28
plotly>=5.14.0
requests>=2.31.0
python-dateutil>=2.8.2
supabase>=2.3.0
```

---

## 🗄️ Database Schema (Supabase)

### `profiles` table
```sql
- id: UUID (PK)
- user_id: UUID (FK → auth.users) UNIQUE
- email: TEXT
- subscription_tier: TEXT ('free' | 'premium')
- created_at: TIMESTAMP
- updated_at: TIMESTAMP
```

### `portfolios` table
```sql
- id: UUID (PK)
- user_id: UUID (FK → profiles)
- ticker: TEXT
- added_at: TIMESTAMP
- UNIQUE(user_id, ticker)
```

---

## 🚀 Running the Application

### 1. Activate virtual environment
```bash
cd "/home/marc/Projects/Market Analysis - SOC - SAM"
source venv/bin/activate
```

### 2. Install dependencies (if needed)
```bash
cd soc-app
pip install -r requirements.txt
```

### 3. Configure Supabase
Edit `soc-app/.streamlit/secrets.toml` with your credentials.

### 4. Run the app
```bash
cd soc-app
streamlit run app.py
```

---

## 📝 Development Workflow

### Making Changes
1. Always work in `soc-app/` folder
2. Test changes locally with Streamlit
3. Commit to Git with descriptive messages
4. Push to `beta-version8` branch

### File Organization Rules
- ✅ All Python code goes in `soc-app/`
- ✅ Documentation goes in project root
- ✅ No duplicate files between root and `soc-app/`
- ✅ Use `.gitignore` for secrets and cache files

---

## 🎨 UI Architecture

### New Polished Layout (Dec 7, 2025)

```
┌─────────────────────────────────────────────────────────┐
│ [Logo] | SOC Seismograph | User: name | Status: Free   │
├─────────────────────────────────────────────────────────┤
│           [     Search Asset (Enter)     ]              │
├─────────────────────────────────────────────────────────┤
│              ┌─────────────────┐                        │
│              │  ACTIVE ASSET   │                        │
│              │  AAPL 🟢        │                        │
│              │  Criticality: 45│                        │
│              └─────────────────┘                        │
├─────────────────────────────────────────────────────────┤
│  Deep Dive Tab | Simulation Tab                         │
│  ...content...                                          │
└─────────────────────────────────────────────────────────┘
```

### Key Features
- ✅ Centered, clean design
- ✅ Enter key triggers search (no button needed)
- ✅ Active asset prominently displayed
- ✅ Portfolio accessible via header button
- ✅ User info always visible

---

## 🔐 Security Notes

### Gitignored Files
- `soc-app/.streamlit/secrets.toml` ← **NEVER commit this**
- `soc-app/data/*.csv` ← Market data cache
- `__pycache__/` ← Python bytecode

### Supabase RLS Policies
- Users can only read/write their own data
- Database-level security enforced
- No cross-user data leakage

---

## 📊 Feature Tiers

### Free Tier
- 3 portfolio assets max
- 5 simulations per day
- Basic analysis

### Premium Tier
- Unlimited portfolio assets
- Unlimited simulations
- Email reports (coming soon)
- Instant alerts (coming soon)

---

## 🛠️ Troubleshooting

### App won't start
1. Check virtual environment is activated
2. Verify `secrets.toml` exists and has correct credentials
3. Ensure all dependencies installed: `pip install -r requirements.txt`

### Database errors
1. Check Supabase connection in dashboard
2. Verify RLS policies are set up correctly
3. Check user permissions in Supabase auth

### Search not working
1. Verify internet connection (needs Yahoo Finance API)
2. Check ticker symbol is valid
3. Try different ticker or wait (API rate limits)

---

**All systems operational! 🚀**

