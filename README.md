//changes 1- Grok API  2- Path for dataset

//to run  1-  cd A:\GENAI\stylegpt\frontend
              npm run dev

          2-  cd A:\GENAI\stylegpt\backend
              venv\Scripts\activate
              python app.py

# StyleGPT — Conversational Multimodal Fashion Search

A v2 upgrade of a visual fashion search platform: instead of only
"upload an image, get similar images," you can now **chat** to refine
results — "show me something like this but in blue, for a wedding,
under ₹3000" — and the app remembers your constraints across the
conversation.

Built entirely with **free, open-source tools**. No paid API required.

## How it works

```
User uploads image
      │
      ▼
 OpenCLIP embeds the image ──────► ChromaDB (vector search) ──► results
      │
User types a refinement ("but in blue, for a wedding")
      │
      ▼
LangChain + free LLM (Groq/Ollama) extracts structured filters
      │  {"color": "blue", "occasion": "wedding"}
      ▼
Filters merged into session memory + re-run vector search ──► refined results
```

This is a **hybrid retrieval** system: CLIP handles "what does it look
like," the LLM handles "what did the user just say in plain English,"
and ChromaDB combines both (visual similarity + metadata filters) in
one query.

## Project structure

```
stylegpt/
├── backend/
│   ├── app.py               # Flask API (upload, search, chat, image routes)
│   ├── embeddings.py         # OpenCLIP image/text embedding
│   ├── vectorstore.py        # ChromaDB storage + search
│   ├── filter_parser.py      # LangChain: text -> structured filters
│   ├── auto_tag.py           # CLIP zero-shot auto-tagging (for unlabeled datasets)
│   ├── bulk_seed_from_folder.py  # Resumable bulk seeder for large datasets
│   ├── seed_data.py          # Small-scale seeder (CSV + few images)
│   ├── requirements.txt
│   └── .env.example
└── frontend/                  # React + Vite (matches your React/Node.js stack)
    ├── src/
    │   ├── App.jsx            # Main component - state, chat, search
    │   ├── api.js             # Centralized fetch calls to the backend
    │   ├── index.css          # Design system (boutique/editorial theme)
    │   └── components/
    │       ├── ChatMessage.jsx
    │       ├── FilterPills.jsx   # swing-tag styled filter chips
    │       └── ResultsGrid.jsx
    ├── index.html
    ├── vite.config.js
    └── package.json
```

## Full Setup Guide (from absolute zero, all free)

### Step 0 — Check you have Python
Open a terminal (Command Prompt / PowerShell on Windows, Terminal on Mac/Linux) and run:
```bash
python --version
```
You need Python 3.10 or higher. If this fails, install Python from
https://www.python.org/downloads/ (tick "Add Python to PATH" during install on Windows),
then close and reopen your terminal.

### Step 1 — Get the project files
Unzip `stylegpt.zip` anywhere on your computer (e.g. Desktop), then in your terminal:
```bash
cd path/to/stylegpt/backend
```
(Replace `path/to/` with wherever you unzipped it — e.g. `cd Desktop/stylegpt/backend`)

### Step 2 — Create a virtual environment (keeps this project's packages isolated)
```bash
python -m venv venv
```
Activate it:
- **Windows:** `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

You'll know it worked if you see `(venv)` at the start of your terminal line.
You need to run this activate command every time you open a new terminal for this project.

### Step 3 — Install all dependencies
```bash
pip install -r requirements.txt
```
This takes a few minutes the first time (downloads torch, chromadb, langchain, etc. — all free).

### Step 4 — Choose a free LLM provider
You need ONE of these two so the chat/refinement feature works:

**Option A — Groq (recommended: easiest, no install, generous free tier)**
1. Go to https://console.groq.com and sign up (no credit card needed)
2. Click "API Keys" → "Create API Key" → copy it
3. In the `backend` folder, copy `.env.example` to a new file named `.env`
4. Open `.env` in any text editor and paste your key:
   `GROQ_API_KEY=gsk_your_actual_key_here`
5. Save. Done — the code auto-detects this.

**Option B — Ollama (fully offline, zero API key, needs ~5GB free disk)**
1. Download and install from https://ollama.com
2. In a terminal run: `ollama pull llama3.1`
3. Leave `.env` empty/don't create it — the code falls back to Ollama automatically.

### Step 5 — Start the backend server
Still inside `backend/` with `(venv)` active:
```bash
python app.py
```
The first time you run this, it downloads the CLIP model (~600MB, one-time, free) —
this can take a few minutes depending on your internet. You'll see:
`* Running on http://127.0.0.1:5000`
Leave this terminal window open and running.

### Step 6 — Add sample fashion images to search
Open a **new** terminal window (keep the server running in the first one):
```bash
cd path/to/stylegpt/backend
venv\Scripts\activate        # Windows (or: source venv/bin/activate on Mac/Linux)
```
1. Put a few fashion images (any .jpg/.png — reuse images from your v1 project)
   into the `backend/sample_images/` folder
2. Open `sample_items.csv` in Excel/Notepad and edit the filenames/colors/prices
   to match your actual images (a template with example rows is already there)
3. Run the seeding script:
```bash
python seed_data.py
```
You should see `[ok] saree1.jpg -> <some-id>` for each image.

### Step 7 — Run the React frontend
Open a **new** terminal window (keep the backend running in the first one):
```
cd path/to/stylegpt/frontend
npm install
npm run dev
```
This starts a dev server (Vite) - open the URL it prints, usually
http://localhost:5173

Try:
- Upload an image → click "Search by Image"
- Type in the chat box: "show me something in blue for a wedding under 3000"

### Troubleshooting
- **"Can't reach the backend"** → make sure `python app.py` is still running in its terminal
- **CORS errors in browser console** → make sure you're opening `index.html` directly
  and that `app.py` is running
- **Groq errors about invalid key** → double check `.env` has no extra spaces/quotes around the key
- **Very slow first request** → normal, CLIP model is loading into memory the first time

## What makes this resume-worthy

- **Hybrid retrieval**: combines dense vector search (CLIP) with structured
  metadata filters extracted by an LLM — a real-world RAG pattern, not a toy demo.
- **Multi-turn memory**: filters persist and merge across a conversation
  (see `SESSIONS` dict in `app.py` and the merge test in `filter_parser.py`).
- **Zero-cost stack**: every component (CLIP, ChromaDB, LangChain, Groq free
  tier / Ollama) is free — a deliberate design choice worth mentioning in interviews.

## Possible next steps
- Swap the in-memory `SESSIONS` dict for Redis for real persistence
- Add a re-ranking step so text and image similarity are properly weighted
- Deploy backend on Render/Railway free tier, frontend on Vercel/GitHub Pages
