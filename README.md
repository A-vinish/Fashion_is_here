# 👗 StyleGPT – Conversational AI Fashion Search

StyleGPT is an AI-powered fashion search platform that combines **computer vision**, **vector search**, and **natural language understanding** to help users discover visually similar fashion products and refine results through conversation.

Instead of relying only on image similarity, users can upload an image and continue refining the search with prompts such as:

> "Show me something similar in blue for a wedding under ₹3000."

The application maintains conversational context and combines image embeddings with structured metadata filters to deliver more relevant recommendations.

---

## ✨ Features

- 📷 Search fashion products using an uploaded image
- 💬 Conversational refinement of search results
- 🎨 Filter by color, occasion, category, gender, and price
- 🧠 CLIP-based visual similarity search
- 🔍 Vector search powered by ChromaDB
- 🤖 Natural language filter extraction using LangChain + Groq LLM
- 🏷 Automatic metadata tagging using zero-shot classification
- ⚡ Fast React frontend with Vite
- 🐍 Flask backend with REST APIs
- 📦 Resumable dataset indexing for large image collections

---

# Demo Workflow

```text
Upload Image
      │
      ▼
Generate CLIP Embedding
      │
      ▼
Search Similar Images (ChromaDB)
      │
      ▼
Display Initial Results
      │
      ▼
User Refines Search
("Blue for a wedding under ₹3000")
      │
      ▼
LangChain + Groq
Extract Structured Filters
      │
      ▼
Combine Filters + Vector Search
      │
      ▼
Updated Personalized Results
```

---

# Tech Stack

## Frontend

- React.js
- Vite
- CSS3
- JavaScript (ES6+)

## Backend

- Python
- Flask
- LangChain
- OpenCLIP
- ChromaDB

## AI & Machine Learning

- OpenCLIP
- Vector Embeddings
- Zero-shot Image Classification
- Semantic Search

## LLM

- Groq API
- Llama 3 (via Groq)

---

# Project Structure

```text
stylegpt
│
├── backend
│   ├── app.py
│   ├── embeddings.py
│   ├── vectorstore.py
│   ├── filter_parser.py
│   ├── auto_tag.py
│   ├── bulk_seed_from_folder.py
│   ├── seed_data.py
│   ├── requirements.txt
│   └── .env.example
│
├── frontend
│   ├── src
│   │   ├── components
│   │   ├── api.js
│   │   ├── App.jsx
│   │   └── index.css
│   │
│   ├── package.json
│   ├── vite.config.js
│   └── index.html
│
└── README.md
```

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/yourusername/stylegpt.git

cd stylegpt
```

---

## 2. Backend Setup

Navigate to backend

```bash
cd backend
```

Create virtual environment

```bash
python -m venv venv
```

Activate environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Configure Environment Variables

Create a `.env` file inside the backend folder.

Example:

```env
GROQ_API_KEY=your_groq_api_key
```

---

## 4. Index Dataset

Update the dataset path inside

```
bulk_seed_from_folder.py
```

Run

```bash
python bulk_seed_from_folder.py
```

This creates CLIP embeddings and stores them inside ChromaDB.

---

## 5. Start Backend

```bash
python app.py
```

Backend runs at

```
http://127.0.0.1:5000
```

---

## 6. Frontend Setup

Open another terminal.

```bash
cd frontend

npm install
```

Start development server

```bash
npm run dev
```

Frontend runs at

```
http://localhost:5173
```

---

# How to Use

### Image Search

1. Upload a fashion image
2. Click **Search by Image**
3. Browse visually similar products

---

### Conversational Search

After image search, continue refining results naturally.

Example prompts

```
Show me something in blue.

Under ₹2500.

For a wedding.

More casual.

Show only sneakers.

Something similar but black.
```

---

# AI Pipeline

```
Image Upload
      │
      ▼
OpenCLIP Embedding
      │
      ▼
ChromaDB Similarity Search
      │
      ▼
Top Matching Products
      │
      ▼
User Query
      │
      ▼
LangChain
      │
      ▼
Groq LLM
      │
      ▼
Structured Filters
      │
      ▼
Filtered Vector Search
      │
      ▼
Updated Recommendations
```

---

# APIs

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/upload` | POST | Upload image |
| `/search` | POST | Search similar products |
| `/chat` | POST | Refine search using natural language |
| `/image/<id>` | GET | Retrieve stored image |

---

# Future Improvements

- User authentication
- Wishlist support
- Shopping cart
- Brand filtering
- Voice search
- Fashion recommendation history
- Dark mode
- Mobile application
- Multi-image search
- Real-time product catalog integration

---

# Author

**Avinish Kumar Mahato**

Software Engineering Student

Interested in

- Artificial Intelligence
- Machine Learning
- Computer Vision
- Full Stack Development
- Generative AI

---

# License

This project is developed for learning and portfolio purposes.
