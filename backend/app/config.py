import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DATA = BASE_DIR / "data"
DOCS_DIR = PROJECT_DATA / "documents"
# CHROMA_DIR = PROJECT_DATA / "chroma"

# Local development uses the project database.
# Vercel uses temporary writable storage.
if os.getenv("VERCEL"):
    CHROMA_DIR = Path("/tmp/chroma")
else:
    CHROMA_DIR = PROJECT_DATA / "chroma"

    
load_dotenv(BASE_DIR / ".env")

OPENROUTER_URL = os.getenv(
    "OPENROUTER_URL",
    "https://openrouter.ai/api/v1/chat/completions",
)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_EMBED_URL = os.getenv("OLLAMA_EMBED_URL", "http://localhost:11434/api/embed")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
FINETUNE_PYTHON = os.getenv("FINETUNE_PYTHON", "").strip()

TOP_K = int(os.getenv("TOP_K", "8"))
MIN_RELEVANCE = float(os.getenv("MIN_RELEVANCE", "0.34"))
FALLBACK_MY = (
    "မေးခွန်းအတွက် လိုအပ်သော အချက်အလက်ကို ထည့်သွင်းထားသော စာရွက်စာတမ်းများတွင် "
    "မတွေ့ရှိပါ။"
)
FALLBACK_EN = "We don't have data for your question in the uploaded documents."
FALLBACK = f"{FALLBACK_MY} ({FALLBACK_EN})"
