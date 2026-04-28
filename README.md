# 🤖 Askify — Ask Anything From Any PDF

> Upload any PDF. Ask any question. Get instant answers powered by AI.

---

## 📌 What is Askify?

**Askify** is an AI-powered document assistant that lets you upload **any PDF** — books, research papers, manuals, reports, novels — and ask questions about them in plain English. No more scrolling through hundreds of pages. Just ask, and Askify finds the answer for you instantly.

Whether it's a textbook, a legal document, a medical report, or your favorite novel — **Askify reads it so you don't have to.**

---

## ✨ Features

- 📄 **Multi-PDF Upload** — Upload one or multiple PDFs at once
- 💬 **Natural Language Q&A** — Ask questions in plain English
- 🧠 **Smart Retrieval** — Uses FAISS vector search to find the most relevant passages
- 🤖 **Dual AI Support** — Powered by **Gemini** with **OpenRouter** as fallback
- 📚 **Source Passages** — See exactly which part of the PDF the answer came from
- ⚡ **Fast & Lightweight** — Built with Streamlit for a smooth experience

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **Streamlit** | Web UI |
| **PyPDF2** | PDF text extraction |
| **LangChain** | AI chain & retrieval |
| **FAISS** | Vector similarity search |
| **HuggingFace Embeddings** | Text embedding (`all-MiniLM-L6-v2`) |
| **Google Gemini** | Primary LLM (`gemini-2.5-flash`) |
| **OpenRouter** | Fallback LLM provider |

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/askify.git
cd askify
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Environment Variables
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_API_BASE=https://openrouter.ai/api/v1
OPENROUTER_MODEL=your_preferred_model
```

### 4. Run the App
```bash
streamlit run app.py
```

---

## 🎮 How to Use

1. **Upload PDFs** — Use the sidebar to upload one or more PDF files
2. **Choose AI Provider** — Select Gemini, OpenRouter, or Auto mode
3. **Click Process** — Askify will extract and embed your PDFs
4. **Ask Anything** — Type your question and get an instant AI-powered answer
5. **View Sources** — Expand "Source Passages" to see where the answer came from

---

## 🔄 AI Provider Modes

| Mode | Description |
|---|---|
| **Auto** | Tries Gemini first, falls back to OpenRouter if it fails |
| **Gemini** | Uses Google Gemini only |
| **OpenRouter** | Uses OpenRouter only |

---

## 📁 Project Structure

```
askify/
├── app.py              # Main Streamlit application
├── .env                # API keys (not committed to git)
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## 🔮 Future Plans

- [ ] Chat history & memory across questions
- [ ] Support for Word (.docx) and text files
- [ ] Export answers to PDF or Word
- [ ] User authentication & saved sessions
- [ ] Summarize entire PDF in one click

---

## 🧑‍💻 Built With ❤️ using LangChain + Streamlit

> **Askify** — *Because every page has an answer, you just need to ask.*
