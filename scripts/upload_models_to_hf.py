#!/usr/bin/env python3
"""
scripts/upload_models_to_hf.py
================================
Upload all model artefacts to HuggingFace Hub.

Run once after build_artifacts.py:
    python scripts/upload_models_to_hf.py

Reads HUGGINGFACE_API_TOKEN and HUGGINGFACE_REPO_ID from .env
"""

import os
import sys
from pathlib import Path

# Load .env
sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(override=True)

TOKEN   = os.getenv("HUGGINGFACE_API_TOKEN", "")
REPO_ID = os.getenv("HUGGINGFACE_REPO_ID",   "")

if not TOKEN:
    print("❌ HUGGINGFACE_API_TOKEN not set in .env")
    sys.exit(1)

if not REPO_ID:
    print("❌ HUGGINGFACE_REPO_ID not set in .env")
    sys.exit(1)

from huggingface_hub import HfApi, create_repo

api = HfApi(token=TOKEN)

# ── Step 1: Create repo if it doesn't exist ───────────────────────────────
print(f"📦 Ensuring repo exists: {REPO_ID}")
try:
    create_repo(
        repo_id   = REPO_ID,
        repo_type = "model",
        private   = False,   # Public — no auth needed at download time
        exist_ok  = True,    # No error if it already exists
        token     = TOKEN,
    )
    print(f"   ✅ Repo ready: https://huggingface.co/{REPO_ID}")
except Exception as e:
    print(f"   ⚠️  create_repo warning (may already exist): {e}")

# ── Step 2: Files to upload ───────────────────────────────────────────────
ROOT = Path(__file__).parent.parent

ARTEFACTS = [
    (ROOT / "models" / "preprocessor.joblib",            "preprocessor.joblib"),
    (ROOT / "models" / "scaler.joblib",                  "scaler.joblib"),
    (ROOT / "models" / "loan_approval_model.joblib",     "loan_approval_model.joblib"),
    (ROOT / "models" / "explainer" / "shap_explainer.joblib", "explainer/shap_explainer.joblib"),
    (ROOT / "models" / "explainer" / "lime_explainer.joblib", "explainer/lime_explainer.joblib"),
]

# Optional — only upload if it exists (large legacy file)
optional = ROOT / "models" / "loan_approval_model.h5"
if optional.exists():
    ARTEFACTS.append((optional, "loan_approval_model.h5"))

# ── Step 3: Upload ────────────────────────────────────────────────────────
print(f"\n⬆️  Uploading {len(ARTEFACTS)} artefacts to {REPO_ID} ...")
print("─" * 55)

success = 0
for local_path, hub_path in ARTEFACTS:
    if not local_path.exists():
        print(f"   ⚠️  SKIP (not found): {local_path.name}")
        continue

    size_kb = local_path.stat().st_size // 1024
    try:
        api.upload_file(
            path_or_fileobj = str(local_path),
            path_in_repo    = hub_path,
            repo_id         = REPO_ID,
            repo_type       = "model",
            token           = TOKEN,
            commit_message  = f"Upload {hub_path}",
        )
        print(f"   ✅ {hub_path:<48}  {size_kb} KB")
        success += 1
    except Exception as e:
        print(f"   ❌ FAILED {hub_path}: {e}")

print("─" * 55)
print(f"\n{'✅ All' if success == len(ARTEFACTS) else f'{success}/{len(ARTEFACTS)}'} artefacts uploaded.")
print(f"🔗 https://huggingface.co/{REPO_ID}")
print("\nNext: set HUGGINGFACE_REPO_ID in Render environment variables.")
