from dotenv import load_dotenv
load_dotenv(override=True)
import os

checks = [
    ("GROQ_API_KEY",          os.getenv("GROQ_API_KEY",          ""), 10),
    ("DATABASE_URL",          os.getenv("DATABASE_URL",          ""), 20),
    ("CHROMA_CLOUD_API_KEY",  os.getenv("CHROMA_CLOUD_API_KEY",  ""),  5),
    ("CHROMA_CLOUD_TENANT",   os.getenv("CHROMA_CLOUD_TENANT",   ""),  5),
    ("HUGGINGFACE_REPO_ID",   os.getenv("HUGGINGFACE_REPO_ID",   ""),  3),
    ("HUGGINGFACE_API_TOKEN", os.getenv("HUGGINGFACE_API_TOKEN", ""),  5),
    ("SECRET_KEY",            os.getenv("SECRET_KEY",            ""), 10),
]

all_ok = True
print("\n🔍 .env key check\n" + "─" * 60)
for name, val, min_len in checks:
    stripped = val.strip()
    ok = len(stripped) >= min_len
    if not ok:
        all_ok = False
    status  = "✅" if ok else "❌ EMPTY/SHORT"
    preview = stripped[:8] + "..." if stripped else "(empty)"
    print(f"  {status}  {name:<28} = {preview}  (len={len(stripped)})")

# Whitespace guard on HF repo ID
hf_repo = os.getenv("HUGGINGFACE_REPO_ID", "")
if hf_repo != hf_repo.strip():
    print(f"\n  ❌ HUGGINGFACE_REPO_ID has whitespace!  repr={repr(hf_repo)}")
    all_ok = False
else:
    print(f"\n  ✅ HUGGINGFACE_REPO_ID clean  value=\"{hf_repo}\"")

# channel_binding guard
db_url = os.getenv("DATABASE_URL", "")
if "channel_binding" in db_url:
    print("  ❌ DATABASE_URL still contains &channel_binding — remove it")
    all_ok = False
else:
    print("  ✅ DATABASE_URL clean  (no channel_binding)")

print("\n" + "─" * 60)
print("Overall:", "✅ ALL KEYS VALID" if all_ok else "❌ FIX ISSUES ABOVE")
