# Supabase Integration - Setup Guide

**SOC Market Seismograph - Multi-User SaaS Edition**

This guide will help you set up Supabase for user authentication and portfolio management.

---

## 📋 Prerequisites

1. **Supabase Account:** Create a free account at [supabase.com](https://supabase.com)
2. **Python 3.8+** with pip installed
3. **Git** for version control

---

## 🗄️ Step 1: Create Supabase Project

1. Log in to [Supabase Dashboard](https://app.supabase.com)
2. Click "New Project"
3. Fill in:
   - **Name:** `soc-market-seismograph`
   - **Database Password:** (generate a strong password)
   - **Region:** Choose closest to your users
4. Click "Create new project"
5. Wait 2-3 minutes for provisioning

---

## 🔧 Step 2: Create Database Tables

### Table 1: `profiles`

Stores user profile information and subscription tiers.

```sql
-- Create profiles table
CREATE TABLE profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE UNIQUE NOT NULL,
    email TEXT NOT NULL,
    subscription_tier TEXT NOT NULL DEFAULT 'free' CHECK (subscription_tier IN ('free', 'premium')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable Row Level Security (RLS)
ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;

-- Policy: Users can read their own profile
CREATE POLICY "Users can read own profile"
    ON profiles FOR SELECT
    USING (auth.uid() = user_id);

-- Policy: Users can update their own profile
CREATE POLICY "Users can update own profile"
    ON profiles FOR UPDATE
    USING (auth.uid() = user_id);

-- Policy: Service role can insert profiles (for signup)
CREATE POLICY "Service role can insert profiles"
    ON profiles FOR INSERT
    WITH CHECK (true);

-- Create index for faster lookups
CREATE INDEX idx_profiles_user_id ON profiles(user_id);
```

### Table 2: `portfolios`

Stores user-saved ticker symbols (watchlist).

```sql
-- Create portfolios table
CREATE TABLE portfolios (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
    ticker TEXT NOT NULL,
    added_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, ticker)
);

-- Enable Row Level Security (RLS)
ALTER TABLE portfolios ENABLE ROW LEVEL SECURITY;

-- Policy: Users can read their own portfolio
CREATE POLICY "Users can read own portfolio"
    ON portfolios FOR SELECT
    USING (auth.uid() = user_id);

-- Policy: Users can insert into their own portfolio
CREATE POLICY "Users can insert own portfolio"
    ON portfolios FOR INSERT
    WITH CHECK (auth.uid() = user_id);

-- Policy: Users can delete from their own portfolio
CREATE POLICY "Users can delete own portfolio"
    ON portfolios FOR DELETE
    USING (auth.uid() = user_id);

-- Create index for faster lookups
CREATE INDEX idx_portfolios_user_id ON portfolios(user_id);
CREATE INDEX idx_portfolios_ticker ON portfolios(ticker);
```

### How to Run SQL

1. In Supabase Dashboard, go to **SQL Editor**
2. Click "New Query"
3. Copy-paste the SQL above (one table at a time)
4. Click "Run" or press `Ctrl+Enter`
5. Verify tables appear in **Table Editor**

---

## 🔑 Step 3: Get API Credentials

1. In Supabase Dashboard, go to **Settings** → **API**
2. Copy these values:
   - **Project URL:** `https://xxxxx.supabase.co`
   - **anon/public key:** `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...`

---

## 📝 Step 4: Configure Streamlit Secrets

1. Navigate to your project directory:
   ```bash
   cd "Market Analysis - SOC - SAM/soc-app"
   ```

2. Create secrets file:
   ```bash
   cp .streamlit/secrets.toml.template .streamlit/secrets.toml
   ```

3. Edit `.streamlit/secrets.toml`:
   ```toml
   SUPABASE_URL = "https://your-project-ref.supabase.co"
   SUPABASE_KEY = "your-anon-public-key-here"
   ```

4. **IMPORTANT:** Never commit `secrets.toml` to git! It's already in `.gitignore`.

---

## 📦 Step 5: Install Dependencies

```bash
cd soc-app
pip install -r requirements.txt
```

This will install:
- `streamlit>=1.30.0`
- `supabase>=2.3.0`
- `pandas`, `numpy`, `yfinance`, `plotly`, etc.

---

## 🚀 Step 6: Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 🧪 Step 7: Test User Flow

### Test Signup

1. Click "Sign Up" tab
2. Enter email: `test@example.com`
3. Enter password: `test123`
4. Check "I agree to Terms"
5. Click "Create Account"
6. ✅ Should see success message and redirect to app

### Test Login

1. Logout (click "🚪 Logout" in sidebar)
2. Click "Login" tab
3. Enter same credentials
4. ✅ Should authenticate successfully

### Test Portfolio

1. Search for a ticker (e.g., "AAPL")
2. Click "⭐ Add to Portfolio"
3. ✅ Should appear in sidebar under "My Portfolio"
4. Try adding 4th asset (as free user)
5. ✅ Should show upgrade prompt

### Test Tier Gating

1. Click "🔒 Simulation (Premium)" tab
2. ✅ Should show upgrade prompt (free user)
3. In Supabase Dashboard, manually update user tier:
   ```sql
   UPDATE profiles SET subscription_tier = 'premium' WHERE email = 'test@example.com';
   ```
4. Refresh app
5. ✅ Simulation tab should now be accessible

---

## 🔒 Security Best Practices

### Row Level Security (RLS)

✅ **Enabled by default** in our setup. This ensures:
- Users can only read/write their own data
- No user can access another user's portfolio
- Database-level security (not just app-level)

### API Keys

- **anon/public key:** Safe to use in frontend (limited permissions)
- **service_role key:** **NEVER** expose in frontend code (full database access)

### Password Requirements

- Minimum 6 characters (enforced by Supabase)
- Consider adding complexity requirements in production

---

## 🎯 Feature Tiers

### Free Tier
- ✅ Full Deep Dive analysis
- ✅ Up to 3 portfolio assets
- ❌ DCA Simulation (locked)
- ❌ Instant Alerts (locked)

### Premium Tier
- ✅ Everything in Free
- ✅ Unlimited portfolio assets
- ✅ DCA Simulation with backtesting
- ✅ Instant Alerts (coming soon)

---

## 🐛 Troubleshooting

### Error: "No module named 'supabase'"

**Solution:**
```bash
pip install supabase
```

### Error: "KeyError: 'SUPABASE_URL'"

**Solution:**
1. Check `.streamlit/secrets.toml` exists
2. Verify keys are spelled correctly (case-sensitive)
3. Restart Streamlit app

### Error: "Invalid API key"

**Solution:**
1. Go to Supabase Dashboard → Settings → API
2. Copy the **anon/public** key (not service_role)
3. Update `.streamlit/secrets.toml`
4. Restart app

### Error: "Row Level Security policy violation"

**Solution:**
1. Check RLS policies are created correctly
2. Verify user is authenticated (`st.session_state.user`)
3. Check SQL policies in Supabase Dashboard → Authentication → Policies

### Users can't sign up

**Solution:**
1. Check Supabase Dashboard → Authentication → Settings
2. Ensure "Enable Email Signup" is ON
3. Check email confirmation settings (disable for testing)

---

## 📊 Database Schema Reference

### `profiles` Table

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Primary key |
| `user_id` | UUID | Foreign key to `auth.users` |
| `email` | TEXT | User's email |
| `subscription_tier` | TEXT | 'free' or 'premium' |
| `created_at` | TIMESTAMP | Account creation time |
| `updated_at` | TIMESTAMP | Last profile update |

### `portfolios` Table

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Primary key |
| `user_id` | UUID | Foreign key to `auth.users` |
| `ticker` | TEXT | Stock/crypto ticker symbol |
| `added_at` | TIMESTAMP | When asset was added |

---

## 🔄 Upgrading Users to Premium

### Manual Upgrade (for testing)

```sql
UPDATE profiles 
SET subscription_tier = 'premium', updated_at = NOW()
WHERE email = 'user@example.com';
```

### Programmatic Upgrade (for production)

Use the `update_user_tier()` function in `auth_manager.py`:

```python
from auth_manager import update_user_tier

success = update_user_tier(user_id, "premium")
if success:
    print("User upgraded to Premium!")
```

---

## 📈 Next Steps

1. **Email Verification:** Enable in Supabase → Authentication → Settings
2. **Password Reset:** Implement forgot password flow
3. **Payment Integration:** Add Stripe for Premium subscriptions
4. **Admin Panel:** Create UI for managing users
5. **Analytics:** Track user engagement with Supabase Analytics

---

## 📞 Support

- **Documentation:** [supabase.com/docs](https://supabase.com/docs)
- **Community:** [discord.gg/supabase](https://discord.gg/supabase)
- **App Issues:** Contact support@socseismograph.com

---

**Setup complete! Your SOC Market Seismograph is now a multi-user SaaS application.** 🎉

