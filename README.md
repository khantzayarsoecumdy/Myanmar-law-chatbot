# Myanmar Law Assistant — OpenRouter with Ollama Fallback

This project provides a React chat website and a Python API. It indexes law documents locally and sends the question plus selected passages to OpenRouter for evidence selection. If OpenRouter reaches a quota or rate limit, or is temporarily unavailable, the API automatically uses the local Ollama model. Ollama is also used for all generation when `OPENROUTER_API_KEY` is blank. The full document collection stays local, but passages sent to OpenRouter are processed by that hosted service.

## Important: two programs must run

The website and API are separate. **A Python version command does not start the website.** Start the backend in one PowerShell window and the frontend in another. Then open `http://localhost:5173` in your browser.

## Requirements

- Windows 10/11, 64-bit
- Python 3.12 or 3.14 (64-bit). The setup script prefers Python 3.14 if it is installed, otherwise Python 3.12.
- Node.js LTS (includes npm)
- OpenRouter API key for the primary provider (optional if you want to use Ollama only)
- Ollama for Windows, to provide the local fallback or local-only generation
- Enough memory to run `qwen2.5:7b`; if it is too slow or runs out of memory, change `OLLAMA_MODEL` to a smaller model you have pulled.

## 1. Configure Ollama fallback

In PowerShell:

```powershell
ollama --version
ollama list
ollama pull qwen2.5:7b
ollama pull bge-m3
```

Ollama normally runs in the background on Windows. If the API is not responding, open another PowerShell window and run `ollama serve` (if it says the port is already in use, Ollama is already running). `qwen2.5:7b` generates evidence selections; `bge-m3` creates the document and question embeddings locally.

The defaults are `OLLAMA_URL=http://localhost:11434/api/chat`, `OLLAMA_MODEL=qwen2.5:7b`, `OLLAMA_EMBED_URL=http://localhost:11434/api/embed`, and `EMBEDDING_MODEL=bge-m3`. Set `OPENROUTER_API_KEY` in `backend\.env` to use OpenRouter as the primary provider. Leave it blank to use Ollama for all generation.

## 2. Install backend dependencies (one time)

Extract the ZIP fully. In PowerShell, change directory to the extracted `myanmar-law-chatbot` project root, for example:

```powershell
cd "$HOME\Desktop\myanmar-law-chatbot"
```

Check which Python versions are installed:

```powershell
py -0p
py --version
```

The setup script uses your Python 3.14.8 if available, otherwise Python 3.12. If dependency installation fails under 3.14, install Python 3.12 (64-bit) from the official Python downloads page and rerun the setup script.

Run the included setup script from the project root:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup-backend.ps1
```

If the script reports an existing `.venv` created with a different Python version, remove only that environment and rerun setup:

```powershell
Remove-Item -Recurse -Force .\backend\.venv
.\scripts\setup-backend.ps1
```

The script installs dependencies and creates `backend\.env` from `.env.example`. Add your OpenRouter key there if you want it as the primary provider. The local fallback defaults to `qwen2.5:7b`; change `OLLAMA_MODEL` if you prefer another model already pulled. If you want local-only generation, leave `OPENROUTER_API_KEY` blank.

## 3. Index the law documents (first run and after adding files)

In the same PowerShell window:

```powershell
cd .\backend
.\.venv\Scripts\Activate.ps1
python -m app.ingest
```

Wait until it prints the indexed chunk count and embedding model. The index uses Ollama's local `bge-m3` multilingual embedding model; make sure `ollama list` shows it. Changing the embedding model requires a full re-index. The API checks the model recorded in the index and reports when rebuilding is required.

To add more documents, copy `.pdf`, `.docx`, or `.txt` files into `backend\data\documents\`, then stop the API and rerun `python -m app.ingest`. Scanned/image-only PDFs need OCR before indexing.

## 4. Start the backend API

Keep the terminal in `backend` and run:

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Keep this window open. Visit `http://127.0.0.1:8000/health` in your browser. It should show JSON with `"status":"ok"` and `"indexed":true`.

To try a question from the API, open `http://127.0.0.1:8000/docs` and use **POST /api/ask**. Set `language` to `auto`, `my` (Burmese), or `en` (English). The `openapi.json` URL describes the API; use `/docs` to submit requests.

## 5. Start the website UI (a second PowerShell window)

Open another PowerShell window. Change to the project root, then run:

```powershell
cd "$HOME\Desktop\myanmar-law-chatbot"
.\scripts\start-frontend.ps1
```

Or run the commands manually:

```powershell
cd .\frontend
npm install
npm run dev
```

Open the exact `Local:` URL printed by Vite, usually **http://localhost:5173**. Keep the frontend terminal open too. The top-right status tells you if the API is offline, documents are not indexed, or the app is ready.

## 6. Troubleshooting

- **Website does not appear:** make sure you ran `npm run dev` in the `frontend` folder. `py --version` only prints Python's version; it does not launch the website. Open the Vite URL shown in the terminal.
- **Dependency installation fails under Python 3.14:** install Python 3.12 (64-bit), remove `backend\.venv`, and rerun `scripts\setup-backend.ps1`.
- **`No module named fastapi` or another package:** run `cd backend`, then `.\.venv\Scripts\Activate.ps1`, then `python -m pip install -r requirements.txt`.
- **API offline:** leave the backend server running at port 8000. Visit `http://127.0.0.1:8000/health` to see the actual error.
- **`indexed:false` or no indexed documents:** run `python -m app.ingest` from `backend` and check it ends with `Indexed ... chunks ...`.
- **Ollama fallback error:** run `ollama list`; verify the model named by `OLLAMA_MODEL` is listed. If necessary start `ollama serve` in a separate terminal. OpenRouter quota/rate limits automatically trigger this local fallback.
- **Embedding error or index mismatch:** run `ollama list`; verify `bge-m3` is listed, stop the API, and run `python -m app.ingest` from `backend` to rebuild the index.
- **Very slow answers / memory error:** a 7B model uses more memory than 3B. Use `qwen2.5:3b` by changing `OLLAMA_MODEL` in `backend\.env`, and make sure that model is pulled.
- **False fallback / irrelevant answer:** `MIN_RELEVANCE` in `backend\.env` controls how similar a retrieved passage must be. The default is `0.34`; tune using test questions from your documents. Lowering it can retrieve more passages but may increase unsupported answers.
- **PDF text missing:** scanned PDFs require OCR. DOCX tables and paragraphs are supported; PDF page text extraction is supported.

## မဖြေနိုင်သောမေးခွန်းများနှင့် fine-tuning

API က စာတမ်းထဲမှ တိုက်ရိုက်အထောက်အထားမရသည့်အခါ မေးခွန်း၊ အချိန်၊ မဖြေနိုင်ရသည့်အကြောင်းရင်းတို့ကို `backend\data\feedback\unanswered.jsonl` တွင် local အဖြစ်မှတ်ထားသည်။ တူညီသောမေးခွန်းကို ထပ်မေးလျှင်လည်း အကြိမ်တစ်ကြိမ်စီ မှတ်တမ်းတင်သည်။ Provider အလုပ်မလုပ်ခြင်းကိုလည်း သီးခြားအကြောင်းရင်းဖြင့် မှတ်ထားပြီး ၎င်းကို အဖြေမှန်အဖြစ် သင်တန်းဒေတာထဲ အလိုအလျောက်မထည့်ပါ။

မေးခွန်းကို စာတမ်းနှင့်တိုက်စစ်ပြီး အဖြေမှန်ကို အတည်ပြုရန် backend folder ထဲမှ အောက်ပါ command ကို run ပါ။ စာတမ်းအမည်နှင့် အဖြေကိုထောက်ခံသော မူရင်းစာပိုဒ်ကို ထည့်ပေးရမည်။ အတည်ပြုထားသည့်အဖြေများသာ `reviewed_qa.jsonl` ထဲဝင်သည်။

```powershell
cd .\backend
.\.venv\Scripts\Activate.ps1
python -m app.review_feedback
```

အတည်ပြုထားသောအဖြေများကို training ဖိုင်အဖြစ်ပြင်ဆင်ရန်:

```powershell
python -m app.prepare_finetune
```

ဒီအဆင့်က `backend\data\feedback\finetune.jsonl` ကို ထုတ်ပေးသည်။ အဖြေကို ထည့်ထားသည့် source passage ထဲတွင် တကယ်ပါ/မပါ စစ်ဆေးပြီးမှ ထည့်သည်။ Training အတွက် မဖြေရှင်းရသေးသောမေးခွန်းကို သီးသန့်မသင်ပေးဘဲ၊ စာတမ်းထဲက အထောက်အထားနှင့် လူက စစ်ဆေးအတည်ပြုထားသော အဖြေကိုတွဲပေးရသည်။

LoRA fine-tuning ကို CUDA GPU ရှိသော Linux သို့မဟုတ် WSL ပတ်ဝန်းကျင်တွင် လုပ်ရန် `backend\requirements-training.txt` ကို သီးခြား training environment ထဲ install လုပ်ပါ။ ထို့နောက် training data အနည်းဆုံး ၁၀ ခု စုဆောင်းပြီး:

```bash
python -m pip install -r requirements-training.txt
python -m app.prepare_finetune
python -m app.train_lora --base-model Qwen/Qwen2.5-7B-Instruct
```

Training က LoRA adapter နှင့် Ollama က import လုပ်နိုင်သော merged Safetensors model ကို `backend/data/feedback/training_output/` ထဲထုတ်ပေးသည်။ Training ပြီးပါက ဖန်တီးပေးထားသော `Modelfile` ကို အသုံးပြု၍ `ollama create myanmar-law-finetuned -f <Modelfile လမ်းကြောင်း>` ကို run လုပ်နိုင်သည်။ ထို့နောက် `backend\.env` ထဲက `OLLAMA_MODEL` ကို `myanmar-law-finetuned` ဟု ပြောင်းပြီး API ကို restart လုပ်ပါ။ Ollama တွင် import လုပ်နိုင်မှုသည် base architecture ကို ထောက်ပံ့မှုနှင့် training hardware ပေါ်မူတည်သည်။

လက်ရှိပရောဂျက်တွင် မေးခွန်းမှတ်တမ်း၊ လူစစ်ဆေးသည့်အဆင့်၊ source passage ကိုစစ်ပြီး JSONL ထုတ်ပေးခြင်းနှင့် LoRA training script တို့ ပါဝင်သည်။ ဥပဒေစာတမ်းမှ အချက်အလက်အသစ်ကို chatbot သိစေရန် အဓိကနည်းမှာ စာတမ်းထည့်ပြီး `python -m app.ingest` ဖြင့် re-index လုပ်ခြင်းဖြစ်သည်။ Fine-tuning improves response style but does not replace document retrieval.

## Grounding and legal caution

To avoid adding unsupported legal details, the chatbot returns short, exact excerpts from the uploaded documents in their original language. The model selects excerpts but does not paraphrase or translate them. The API checks that each excerpt exists in the cited document chunk and shows it under the answer. If it cannot find a passage that directly answers the question, it returns the fallback.

The assistant is instructed to answer only from retrieved passages and return this fallback when it cannot find sufficient evidence:

`မေးခွန်းအတွက် လိုအပ်သော အချက်အလက်ကို ထည့်သွင်းထားသော စာရွက်စာတမ်းများတွင် မတွေ့ရှိပါ။ (We don't have data for your question in the uploaded documents.)`

Retrieval thresholds and model behavior need to be tested on your own questions; no generative model can guarantee zero hallucinations. Uploaded laws may be outdated or amended. This is a document research aid, not a substitute for legal advice.
