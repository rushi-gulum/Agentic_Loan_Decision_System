# 🚀 Deployment Guide — Forever-Free Cloud Stack

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                   STREAMLIT COMMUNITY CLOUD                      │
│            frontend/streamlit_app_cloud.py                       │
│            Auto-deploys from GitHub on every push                │
└─────────────────────────┬────────────────────────────────────────┘
                          │  HTTPS POST  /api/v1/evaluate
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                      RENDER  (Web Service)                       │
│          uvicorn api.app:app  —  free tier, sleeps after 15m    │
└────────────┬───────────────────────────────────┬─────────────────┘
             │                                   │
             ▼                                   ▼
┌─────────────────────────┐         ┌────────────────────────────┐
│   NEON.TECH  (Postgres) │         │   CHROMA CLOUD             │
│   Free · no expiry      │         │   Managed vector DB        │
│                         │         │                            │
│  • loan_decisions       │         │  • RBI policy chunks       │
│  • compliance_violations│         │  • Sentence embeddings     │
└─────────────────────────┘         └────────────────────────────┘
                          │                      │
                          └──────────┬───────────┘
                                     ▼
                          ┌─────────────────────┐
                          │   GROQ API  (free)  │
                          │   Llama 3.1 8B      │
                          └─────────────────────┘
                          ┌─────────────────────┐
                          │   HUGGING FACE HUB  │
                          │   Model artefacts   │
                          │   (downloaded on    │
                          │    container boot)  │
                          └─────────────────────┘
```

| Service | URL | Free tier |
|---|---|---|
| Render | render.com | 750 h/month, sleeps after 15 min inactivity |
| Neon | neon.tech | 0.5 GB storage, free forever |
| Chroma Cloud | trychroma.com | 1 M embeddings, free forever |
| HuggingFace Hub | huggingface.co | Unlimited public repos |
| Groq | console.groq.com | ~14 400 req/day on free tier |
| Streamlit | share.streamlit.io | Unlimited public apps |

**Total monthly cost: $0**

---

## Step 1 — Neon.tech (Postgres)

1. Go to **[neon.tech](https://neon.tech)** → Sign up (GitHub login works).
2. **New Project** → name it `loan-decision-system` → region closest to you.
3. In the project dashboard open **Connection Details**.
4. Copy the **Connection string** — it looks like:
   ```
   postgresql://user:pass@ep-cool-name-123.us-east-1.aws.neon.tech/neondb?sslmode=require
   ```
5. Paste it as `DATABASE_URL` in your `.env` (local) and in Render env vars (production).

> Tables are created automatically on first API boot via `init_db()`.  
> No manual SQL needed.

---

## Step 2 — Chroma Cloud (Vector DB)

1. Go to **[trychroma.com](https://trychroma.com)** → Sign up.
2. Create a **Tenant** (e.g. `loan-system`) and a **Database** named `rbi_guidelines`.
3. Go to **Settings → API Keys** → create a new key.
4. Set these env vars:
   ```
   CHROMA_CLOUD_API_KEY=<your api key>
   CHROMA_CLOUD_TENANT=<your tenant id>
   CHROMA_CLOUD_DATABASE=rbi_guidelines
   ```
5. After deploying the backend, run the ingestion once:
   ```bash
   python pipeline/ingest_rag.py
   ```
   This pushes all RBI PDF chunks to Chroma Cloud.

> If these vars are empty, the system silently falls back to a local
> `PersistentClient` at `./data/embeddings` — safe for development.

---

## Step 3 — Hugging Face Hub (Model Storage)

Render containers start empty. Models must live somewhere persistent.

### 3a — Upload artefacts (run once locally)

```bash
# Install HF CLI
pip install huggingface_hub

# Login
huggingface-cli login        # paste your HF token when prompted

# Create repo (public is fine — models contain no PII)
huggingface-cli repo create loan-decision-assets --type model

# Upload artefacts
python - <<'EOF'
from huggingface_hub import HfApi
api = HfApi()
repo = "YOUR_HF_USERNAME/loan-decision-assets"

files = [
    ("models/preprocessor.joblib",             "preprocessor.joblib"),
    ("models/scaler.joblib",                   "scaler.joblib"),
    ("models/explainer/shap_explainer.joblib", "explainer/shap_explainer.joblib"),
    ("models/explainer/lime_explainer.joblib", "explainer/lime_explainer.joblib"),
    ("models/loan_approval_model.h5",          "loan_approval_model.h5"),
]
for local_path, hub_path in files:
    api.upload_file(path_or_fileobj=local_path, path_in_repo=hub_path, repo_id=repo)
    print(f"✅ Uploaded {hub_path}")
EOF
```

### 3b — Set env vars

```
HUGGINGFACE_REPO_ID=YOUR_HF_USERNAME/loan-decision-assets
HUGGINGFACE_API_TOKEN=hf_xxxxxxxxxxxxxxxx    # only needed for private repos
```

> On every Render cold-start, `download_models()` runs in the lifespan hook.
> Already-cached files are skipped — only missing files are downloaded.

---

## Step 4 — Render (FastAPI Backend)

### 4a — Push code to GitHub

```bash
git add .
git commit -m "feat: cloud architecture with Neon + Chroma + HF Hub"
git push origin main
```

### 4b — Create the Web Service

1. Go to **[render.com](https://render.com)** → **New → Web Service**.
2. Connect your GitHub repo.
3. Fill in:

   | Field | Value |
   |---|---|
   | **Name** | `loan-decision-api` |
   | **Runtime** | `Python 3` |
   | **Build Command** | `pip install -r requirements.txt` |
   | **Start Command** | `uvicorn api.app:app --host 0.0.0.0 --port $PORT` |
   | **Plan** | `Free` |

4. Scroll to **Environment Variables** and add every key from `.env.template`.  
   Minimum required set:

   ```
   ENVIRONMENT           = production
   GROQ_API_KEY          = gsk_...
   DATABASE_URL          = postgresql://...neon.tech/...?sslmode=require
   CHROMA_CLOUD_API_KEY  = ...
   CHROMA_CLOUD_TENANT   = ...
   HUGGINGFACE_REPO_ID   = yourname/loan-decision-assets
   SECRET_KEY            = <random 64-char hex>
   ALLOWED_HOSTS         = loan-decision-api.onrender.com
   CORS_ORIGINS          = https://your-app.streamlit.app
   LOG_LEVEL             = INFO
   DEBUG                 = false
   ```

5. Click **Create Web Service**.  
   First deploy takes ~5 min (pip install + model download).

6. Note your URL: `https://loan-decision-api.onrender.com`

### 4c — Verify

```bash
# Health check
curl https://loan-decision-api.onrender.com/health | python -m json.tool

# Deployment topology
curl https://loan-decision-api.onrender.com/status | python -m json.tool

# Test evaluation (replace with your URL)
curl -X POST https://loan-decision-api.onrender.com/api/v1/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "age_years": 35,
    "bureau_score": 720,
    "monthly_income_inr": 80000,
    "requested_amount_inr": 2500000,
    "tenure_months": 240,
    "interest_rate_annual_pct": 8.5,
    "foir_total_obligations_pct": 40.0,
    "loan_type": "housing"
  }'
```

---

## Step 5 — Streamlit Community Cloud (Frontend)

1. Go to **[share.streamlit.io](https://share.streamlit.io)** → sign in with GitHub.
2. Click **New app**.
3. Fill in:

   | Field | Value |
   |---|---|
   | **Repository** | `your-github-username/Agentic_Loan_Decision_System` |
   | **Branch** | `main` |
   | **Main file path** | `frontend/streamlit_app_cloud.py` |

4. Click **Advanced settings → Secrets** and paste:

   ```toml
   API_BASE_URL = "https://loan-decision-api.onrender.com"
   ```

5. Click **Deploy**.  
   Streamlit installs `streamlit_requirements.txt` automatically if present,  
   otherwise it uses the full `requirements.txt`.

6. Your app is live at `https://your-app.streamlit.app`.

---

## Updating the Deployment

```bash
# Make changes locally, test, then:
git add .
git commit -m "fix: ..."
git push origin main
# → Render redeploys automatically
# → Streamlit Cloud redeploys automatically
```

---

## Re-ingesting RBI Documents

Whenever you add new PDFs to `rules/rbi_guidelines/`:

```bash
# Locally with CHROMA env vars set:
python pipeline/ingest_rag.py

# Or on the Render shell (one-off job):
# Render Dashboard → your service → Shell → run the command
```

---

## Troubleshooting

### Render cold-start is slow
Free tier sleeps after 15 min. Add a UptimeRobot ping every 14 min to keep it warm:
- [uptimerobot.com](https://uptimerobot.com) → New monitor → HTTP → your `/health` URL → 14-min interval

### DATABASE_URL connection refused
Neon pauses free projects after 5 days of inactivity. Log in to Neon and click **Resume Project**.

### Model download fails on startup
Check `HUGGINGFACE_REPO_ID` is correct and the repo is public (or `HUGGINGFACE_API_TOKEN` is set for private).  
Check Render logs: Dashboard → your service → **Logs**.

### Chroma `heartbeat()` times out
Verify `CHROMA_CLOUD_API_KEY` and `CHROMA_CLOUD_TENANT` in Render env vars.  
System falls back to local disk automatically — no crash.

### `ModuleNotFoundError` on Render
Ensure the failing package is in `requirements.txt` and redeploy.

---

## Local Development

```bash
# 1. Clone and set up env
git clone <your-fork>
cd Agentic_Loan_Decision_System
cp .env.template .env        # fill in at minimum GROQ_API_KEY

# 2. Install and build
.\setup.ps1 init             # Windows
make init                    # Mac / Linux

# 3. Run
make api                     # http://localhost:8000
make ui                      # http://localhost:8501

# 4. Test
make test                    # 28 unit tests
python utils/db_utils.py     # verify DB connection
python utils/model_loader.py # verify HF download
```

---

## Environment Variable Quick-Reference

| Variable | Required | Where to get it |
|---|---|---|
| `GROQ_API_KEY` | ✅ | [console.groq.com](https://console.groq.com) → API Keys |
| `DATABASE_URL` | ✅ | [neon.tech](https://neon.tech) → Project → Connection string |
| `CHROMA_CLOUD_API_KEY` | ⚠️ optional | [trychroma.com](https://trychroma.com) → Settings → API Keys |
| `CHROMA_CLOUD_TENANT` | ⚠️ optional | Same page as above |
| `HUGGINGFACE_REPO_ID` | ⚠️ optional | `username/repo-name` on huggingface.co |
| `HUGGINGFACE_API_TOKEN` | ⚠️ optional | [hf.co/settings/tokens](https://huggingface.co/settings/tokens) |
| `SECRET_KEY` | ✅ prod | `python -c "import secrets; print(secrets.token_hex(32))"` |
| `ALLOWED_HOSTS` | ✅ prod | Your Render URL |
| `CORS_ORIGINS` | ✅ prod | Your Streamlit Cloud URL |
