"""
utils/model_loader.py
=====================
HuggingFace Hub Model Downloader with Local Cache

On Render (or any ephemeral container) the filesystem is empty on every
cold-start.  This module downloads the required model artefacts from a
HuggingFace Hub repo on first boot and keeps them in the local ./models/
directory so subsequent requests are fast.

Routing:
  HUGGINGFACE_REPO_ID set   →  download from HF Hub (cloud)
  not set / files present  →  use local files (dev / already cached)

Files managed (mirrors what pipeline/build_artifacts.py produces):
  models/preprocessor.joblib
  models/scaler.joblib
  models/loan_approval_model.h5          (optional – large)
  models/explainer/shap_explainer.joblib
  models/explainer/lime_explainer.joblib
"""

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ID   = os.getenv("HUGGINGFACE_REPO_ID", "")
HF_TOKEN  = os.getenv("HUGGINGFACE_API_TOKEN", "")   # needed for private repos

# Local paths that mirror the project layout
MODEL_DIR    = Path(os.getenv("MODEL_CACHE_DIR", "models"))
EXPLAINER_DIR = MODEL_DIR / "explainer"

# Each entry: (hub_filename, local_path, required)
# required=False → missing file is logged but does NOT abort startup
ARTEFACT_MANIFEST = [
    ("preprocessor.joblib",              MODEL_DIR / "preprocessor.joblib",              True),
    ("scaler.joblib",                    MODEL_DIR / "scaler.joblib",                    True),
    ("explainer/shap_explainer.joblib",  EXPLAINER_DIR / "shap_explainer.joblib",        True),
    ("explainer/lime_explainer.joblib",  EXPLAINER_DIR / "lime_explainer.joblib",        True),
    ("loan_approval_model.h5",           MODEL_DIR / "loan_approval_model.h5",           False),  # large optional
]

# ---------------------------------------------------------------------------
# Core download logic
# ---------------------------------------------------------------------------

def _hf_download_file(repo_id: str, filename: str, local_path: Path, token: Optional[str]) -> bool:
    """
    Download a single file from HF Hub into local_path.
    Returns True on success, False on failure.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.error("huggingface_hub not installed — run: pip install huggingface_hub")
        return False

    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # hf_hub_download returns the cached path; we copy it to our layout
        cached = hf_hub_download(
            repo_id   = repo_id,
            filename  = filename,
            repo_type = "model",
            token     = token or None,
            local_dir = str(local_path.parent),  # saves to parent dir with original filename
        )

        # If HF saved it with the original filename, rename/move if needed
        saved_at = Path(cached)
        if saved_at.resolve() != local_path.resolve() and not local_path.exists():
            saved_at.rename(local_path)

        size_kb = local_path.stat().st_size // 1024
        logger.info("📥 Downloaded %-45s  (%d KB)", filename, size_kb)
        return True

    except Exception as exc:
        logger.error("❌ Failed to download %s: %s", filename, exc)
        return False


def download_models(force: bool = False) -> dict:
    """
    Ensure all artefacts are present.  Downloads missing files from HF Hub.

    Args:
        force: Re-download even if local file already exists.

    Returns:
        dict with keys: success (bool), downloaded (list), skipped (list), failed (list)
    """
    downloaded: list = []
    skipped:    list = []
    failed:     list = []

    if not REPO_ID:
        logger.info("ℹ️  HUGGINGFACE_REPO_ID not set — using local artefacts only")
        # Just validate local files exist
        for _, local_path, required in ARTEFACT_MANIFEST:
            if local_path.exists():
                skipped.append(str(local_path))
            elif required:
                logger.warning("⚠️  Required artefact missing locally: %s", local_path)
                failed.append(str(local_path))
        return {
            "success": len(failed) == 0,
            "repo": "local",
            "downloaded": downloaded,
            "skipped": skipped,
            "failed": failed,
        }

    logger.info("🤗 Checking artefacts from HF Hub repo: %s", REPO_ID)

    for hub_filename, local_path, required in ARTEFACT_MANIFEST:
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if local_path.exists() and not force:
            size_kb = local_path.stat().st_size // 1024
            logger.debug("✅ Cached  %-45s  (%d KB)", hub_filename, size_kb)
            skipped.append(hub_filename)
            continue

        logger.info("⬇️  Fetching %s ...", hub_filename)
        ok = _hf_download_file(REPO_ID, hub_filename, local_path, HF_TOKEN)

        if ok:
            downloaded.append(hub_filename)
        elif required:
            failed.append(hub_filename)
            logger.error("Required artefact download failed: %s", hub_filename)
        else:
            logger.warning("Optional artefact unavailable: %s", hub_filename)

    success = len(failed) == 0
    status_emoji = "✅" if success else "❌"
    logger.info(
        "%s Model sync complete — downloaded=%d  cached=%d  failed=%d",
        status_emoji, len(downloaded), len(skipped), len(failed),
    )
    return {
        "success": success,
        "repo": REPO_ID,
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
    }


# ---------------------------------------------------------------------------
# Convenience loaders — called by agents and API routes
# ---------------------------------------------------------------------------

def load_preprocessor():
    """Load the sklearn preprocessing pipeline. Downloads from HF if missing."""
    path = MODEL_DIR / "preprocessor.joblib"
    if not path.exists():
        download_models()
    if not path.exists():
        raise FileNotFoundError(
            f"preprocessor.joblib not found at {path}. "
            "Run `python pipeline/build_artifacts.py` or set HUGGINGFACE_REPO_ID."
        )
    import joblib
    logger.debug("📂 Loading preprocessor from %s", path)
    return joblib.load(path)


def load_scaler():
    """Load the feature scaler. Downloads from HF if missing."""
    path = MODEL_DIR / "scaler.joblib"
    if not path.exists():
        download_models()
    if not path.exists():
        raise FileNotFoundError(f"scaler.joblib not found at {path}.")
    import joblib
    return joblib.load(path)


def load_shap_explainer():
    """Load the SHAP explainer. Downloads from HF if missing."""
    path = EXPLAINER_DIR / "shap_explainer.joblib"
    if not path.exists():
        download_models()
    if not path.exists():
        raise FileNotFoundError(f"shap_explainer.joblib not found at {path}.")
    import joblib
    return joblib.load(path)


def load_lime_explainer():
    """Load the LIME explainer. Downloads from HF if missing."""
    path = EXPLAINER_DIR / "lime_explainer.joblib"
    if not path.exists():
        download_models()
    if not path.exists():
        raise FileNotFoundError(f"lime_explainer.joblib not found at {path}.")
    import joblib
    return joblib.load(path)


def check_model_health() -> dict:
    """
    Return status of all artefacts — used by /health endpoint.
    Does NOT trigger downloads; just reports what's on disk.
    """
    status = {}
    all_ok = True

    for hub_filename, local_path, required in ARTEFACT_MANIFEST:
        exists   = local_path.exists()
        size_kb  = (local_path.stat().st_size // 1024) if exists else 0
        if required and not exists:
            all_ok = False
        status[hub_filename] = {
            "present":   exists,
            "size_kb":   size_kb,
            "required":  required,
            "local_path": str(local_path),
        }

    return {
        "status":    "healthy" if all_ok else "degraded",
        "hf_repo":   REPO_ID or "local",
        "artefacts": status,
    }


# ---------------------------------------------------------------------------
# CLI helper — useful during Render build step or local dev
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    print("=" * 55)
    print("  HuggingFace Hub Model Loader")
    print("=" * 55)
    result = download_models()
    print(json.dumps(result, indent=2))

    print("\n📋 Health check:")
    print(json.dumps(check_model_health(), indent=2))
