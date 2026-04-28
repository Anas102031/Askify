import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

# Use OpenAI client (OpenRouter compatible)
try:
    from openai import OpenAI as OpenAIClient
    OPENAI_CLIENT_AVAILABLE = True
except Exception:
    OPENAI_CLIENT_AVAILABLE = False

# ---------- Load .env for API keys ----------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_BASE = os.getenv("OPENROUTER_API_BASE")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL")

# ---------- Set Page Config ----------
st.set_page_config(page_title="AI PDF Assistant", layout="wide")

# ---------- Styles ----------
st.markdown("""
    <style>
        .header-text {
            font-size: 40px;
            font-weight: bold;
            text-align: center;
            color: #FFD700;
            margin-top: 10px;
            margin-bottom: 30px;
            text-shadow: 0 0 15px #FFD700;
        }
        .sub-text {
            font-size: 18px;
            color: #D2D2D2;
        }
    </style>
""", unsafe_allow_html=True)
st.markdown("<div class='header-text'>🤖 AI PDF Assistant</div>", unsafe_allow_html=True)

# ---------- Extract Text from PDF ----------
def extract_text_from_pdfs(pdf_files):
    pdf_texts = {}
    for pdf in pdf_files:
        try:
            reader = PdfReader(pdf)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            pdf_texts[pdf.name] = text
        except Exception as e:
            pdf_texts[pdf.name] = f"Error reading file: {e}"
    return pdf_texts

# ---------- Chunk Text ----------
def create_chunks(texts: dict, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks, refs = [], []
    for fname, content in texts.items():
        split_chunks = splitter.split_text(content)
        chunks.extend(split_chunks)
        refs += [fname] * len(split_chunks)
    return chunks, refs

# ---------- Embeddings ----------
def generate_faiss_vectorstore(chunks):
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_texts(chunks, embedding=embedding_model)
    return vector_store

# ---------- LLM: Gemini (Langchain wrapper) ----------
def create_gemini_chain(vector_store):
    if not GEMINI_API_KEY:
        st.error("Missing GEMINI_API_KEY in .env file")
        return None

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        google_api_key=GEMINI_API_KEY
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
        chain_type="stuff",
        verbose=False,
        return_source_documents=True
    )
    return qa_chain

# ---------- LLM: OpenRouter fallback (OpenAI-compatible) ----------
def query_openrouter_with_context(openro_api_key, vector_store, question, k=4, model=OPENROUTER_MODEL):
    # Preconditions
    if not OPENAI_CLIENT_AVAILABLE:
        raise RuntimeError("openai python package (with OpenAI client) not installed. Please `pip install openai` to use OpenRouter fallback.")
    if not openro_api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY in environment for OpenRouter fallback.")

    # Build retriever (robust across langchain versions)
    retriever = None
    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
    except Exception:
        # if as_retriever fails, we'll attempt to use vector_store directly below
        retriever = None

    docs = None
    # Try multiple retrieval APIs for compatibility
    try:
        if retriever is not None:
            if hasattr(retriever, "get_relevant_documents"):
                docs = retriever.get_relevant_documents(question)
            elif hasattr(retriever, "retrieve"):
                docs = retriever.retrieve(question)
            else:
                # attempt call-as-function
                try:
                    docs = retriever(question)
                except Exception:
                    docs = None

        if docs is None and hasattr(vector_store, "similarity_search"):
            docs = vector_store.similarity_search(question, k=k)

        if docs is None:
            raise RuntimeError("Could not retrieve documents: retriever/vector_store does not expose compatible retrieval methods.")
    except Exception as e:
        raise RuntimeError(f"Retrieval failed: {e}")

    # Build a context string from retrieved docs
    context_parts = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get('source', f'doc{i}') if hasattr(d, "metadata") else f"doc{i}"
        excerpt = d.page_content if hasattr(d, "page_content") else str(d)
        context_parts.append(f"[DOC {i} | {src}]\n{excerpt}\n")

    context_text = "\n---\n".join(context_parts)

    # A simple instruction prompt
    prompt = (
        "You are an assistant who answers questions using provided context.\n"
        "Use only the information in the context to answer. If the answer is not in the context, say you don't know.\n\n"
        "Context:\n" + context_text + "\nQuestion: " + question + "\nAnswer:"
    )

    # Create OpenAI (OpenRouter) client and call chat completions
    client = OpenAIClient(base_url=OPENROUTER_API_BASE, api_key=openro_api_key)

    # Use same pattern as your example: client.chat.completions.create(...)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=800,
        extra_headers={
            # optional: helpful for OpenRouter analytics/rankings (replace as needed)
            # "HTTP-Referer": "<YOUR_SITE_URL>",
            # "X-Title": "<YOUR_SITE_NAME>",
        },
        extra_body={},  # passthrough additional body fields if needed
    )

    # Parse response depending on returned shape
    # OpenRouter/OpenAI client returns .choices similar to OpenAI: use first choice
    try:
        answer = response.choices[0].message.content.strip()
    except Exception:
        # fallback extraction (some clients return slightly different shape)
        try:
            answer = response.choices[0].text.strip()
        except Exception:
            answer = str(response)

    return {"result": answer, "source_documents": docs}

# ---------- Create a chain handler that supports provider choice and fallback ----------
class MultiProviderQA:
    def __init__(self, vector_store, provider="auto"):
        # provider: "auto", "gemini", "openrouter"
        self.vs = vector_store
        self.provider = provider
        self.gemini_chain = None
        if provider in ("auto", "gemini"):
            try:
                self.gemini_chain = create_gemini_chain(self.vs)
            except Exception as e:
                # keep None and allow fallback
                st.warning(f"Gemini chain init warning: {e}")

    def __call__(self, question):
        # Try Gemini first for auto or gemini
        if self.provider in ("auto", "gemini") and self.gemini_chain:
            try:
                # Gemini Langchain chain returns a dict-like result when called
                return self.gemini_chain(question)
            except Exception as e:
                st.warning(f"Gemini error: {e}")
                # If provider was explicitly gemini, return the error; if auto, try OpenRouter
                if self.provider == "gemini":
                    raise

        # If we reach here, use OpenRouter fallback
        try:
            return query_openrouter_with_context(OPENROUTER_API_KEY, self.vs, question, k=4)
        except Exception as e:
            # If both fail, raise final error
            raise RuntimeError(f"Both providers failed. Last error: {e}")

# ---------- Upload & Process PDFs (Sidebar) ----------
with st.sidebar:
    st.title("📄 Upload PDFs")
    uploaded_files = st.file_uploader("Upload one or more PDF files", type=["pdf"], accept_multiple_files=True)

    provider_choice = st.radio("LLM Provider / Mode", options=["auto", "gemini", "openrouter"], index=0,
                                format_func=lambda x: "Auto (Gemini → OpenRouter fallback)" if x=="auto" else ("Gemini" if x=="gemini" else "OpenRouter"))

    if uploaded_files and st.button("⚙️ Process PDFs"):
        with st.spinner("Extracting & embedding..."):
            texts = extract_text_from_pdfs(uploaded_files)
            chunks, refs = create_chunks(texts)
            vector_store = generate_faiss_vectorstore(chunks)
            st.session_state["vectordb"] = vector_store
            st.session_state["provider"] = provider_choice
            st.success("✅ PDF processed & vector DB created!")

# ---------- Main Layout ----------
st.markdown("### 💬 Ask a question to your documents")

if "vectordb" in st.session_state:
    query = st.text_input("Ask something related to the uploaded documents:")

    if query:
        # Create multi-provider handler with user choice
        chain = MultiProviderQA(st.session_state["vectordb"], provider=st.session_state.get("provider", "auto"))

        try:
            with st.spinner("Thinking..."):
                result = chain(query)
                answer = result.get("result") or result.get("answer") or ""
                source_docs = result.get("source_documents", [])

            st.markdown("#### 🤖 Answer:")
            st.markdown(answer)

            with st.expander("📚 Source Passages"):
                for i, doc in enumerate(source_docs):
                    st.markdown(f"**Doc {i+1} - {doc.metadata.get('source', 'unknown')}**")
                    st.text(doc.page_content)
                    st.markdown("---")
        except Exception as e:
            st.error(f"Error while getting answer: {e}")
else:
    st.info("📎 Upload and process PDFs to begin.")

# ---------- Usage Notes ----------
st.write("---")
st.write("**Notes:** Make sure to set environment variables in your .env file:\n- GEMINI_API_KEY=...\n- OPENROUTER_API_KEY=...\n(Optional) OPENROUTER_API_BASE and OPENROUTER_MODEL if you need custom endpoint/model names.")
