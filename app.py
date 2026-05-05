import streamlit as st
from pypdf import PdfReader
import os
import io
from datetime import datetime
from dotenv import load_dotenv

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.enums import TA_LEFT, TA_CENTER

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ── Load environment variables ──────────────────────────────────────────────
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartDocAI · by Akanksha",
    page_icon="📄",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Sora', sans-serif;
    }

    .main-title {
        font-family: 'Space Mono', monospace;
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }

    .subtitle {
        color: #888;
        font-size: 0.95rem;
        margin-top: 0;
        margin-bottom: 2rem;
    }

    .chat-bubble-user {
        background: #0072ff;
        color: white;
        padding: 0.7rem 1.1rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.4rem 0;
        max-width: 80%;
        margin-left: auto;
        font-size: 0.95rem;
    }

    .chat-bubble-bot {
        background: #1e1e2e;
        color: #e0e0e0;
        padding: 0.7rem 1.1rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.4rem 0;
        max-width: 85%;
        font-size: 0.95rem;
        border: 1px solid #2e2e4e;
    }

    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 1.5px solid #0072ff55;
        padding: 0.6rem 1rem;
    }

    .stButton > button {
        border-radius: 10px;
        background: linear-gradient(135deg, #00c6ff, #0072ff);
        color: white;
        border: none;
        font-weight: 600;
        padding: 0.5rem 1.4rem;
    }

    .stButton > button:hover {
        opacity: 0.88;
        transform: translateY(-1px);
    }

    .info-box {
        background: #1e1e2e;
        border: 1px solid #2e2e4e;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        font-size: 0.88rem;
        color: #aaa;
        margin-top: 1rem;
    }

    .badge {
        display: inline-block;
        background: #0072ff22;
        color: #00c6ff;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.78rem;
        font-family: 'Space Mono', monospace;
        margin-right: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_embeddings():
    """Load HuggingFace embeddings (cached across reruns)."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def extract_text_from_pdfs(pdf_files) -> str:
    """Extract all text from a list of uploaded PDF files."""
    full_text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    return full_text


def split_text(raw_text: str):
    """Split raw text into overlapping chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len,
    )
    return splitter.split_text(raw_text)


def build_vector_store(chunks):
    """Embed chunks and store in a FAISS vector index."""
    embeddings = get_embeddings()
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_store


def answer_question(vector_store, question: str) -> str:
    """Retrieve relevant chunks and generate an answer using LCEL chain."""
    prompt_template = PromptTemplate.from_template("""
You are a helpful, expert assistant named AskMyPDF.
Use the provided context from the uploaded PDF(s) to answer the question accurately and concisely.
If the answer is not contained within the context, say:
"I couldn't find this information in the uploaded documents."

Context:
{context}

Question:
{question}

Answer:
""")

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-flash-latest",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3,
    )

    # Retrieve top-4 relevant docs and join their text as context
    docs = vector_store.similarity_search(question, k=4)
    context_text = "\n\n".join(doc.page_content for doc in docs)

    # Modern LCEL chain: prompt | llm | output parser
    chain = prompt_template | llm | StrOutputParser()

    return chain.invoke({"context": context_text, "question": question})


def generate_chat_pdf(chat_history: list, pdf_names: list) -> bytes:
    """Generate a downloadable PDF of the full chat conversation."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ChatTitle",
        parent=styles["Title"],
        fontSize=20,
        textColor=colors.HexColor("#0072ff"),
        spaceAfter=6,
        alignment=TA_CENTER,
    )
    meta_style = ParagraphStyle(
        "ChatMeta",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#888888"),
        spaceAfter=14,
        alignment=TA_CENTER,
    )
    label_user_style = ParagraphStyle(
        "LabelUser",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#0072ff"),
        fontName="Helvetica-Bold",
        spaceBefore=12,
        spaceAfter=2,
    )
    label_bot_style = ParagraphStyle(
        "LabelBot",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#00aa55"),
        fontName="Helvetica-Bold",
        spaceBefore=12,
        spaceAfter=2,
    )
    bubble_user_style = ParagraphStyle(
        "BubbleUser",
        parent=styles["Normal"],
        fontSize=11,
        textColor=colors.HexColor("#1a1a2e"),
        backColor=colors.HexColor("#ddeeff"),
        borderPadding=(6, 10, 6, 10),
        spaceAfter=4,
        leading=16,
    )
    bubble_bot_style = ParagraphStyle(
        "BubbleBot",
        parent=styles["Normal"],
        fontSize=11,
        textColor=colors.HexColor("#1a1a2e"),
        backColor=colors.HexColor("#eef8f2"),
        borderPadding=(6, 10, 6, 10),
        spaceAfter=4,
        leading=16,
    )

    story = []

    # Header
    story.append(Paragraph(" SmartDocAI- Chat Export", title_style))
    docs_str = ", ".join(pdf_names) if pdf_names else "N/A"
    now = datetime.now().strftime("%d %b %Y, %H:%M")
    story.append(Paragraph(f"Documents: {docs_str}  |  Exported: {now}", meta_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0072ff"), spaceAfter=16))

    if not chat_history:
        story.append(Paragraph("No conversation to export yet.", styles["Normal"]))
    else:
        for role, msg in chat_history:
            safe_msg = (msg
                        .replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;"))
            if role == "user":
                story.append(Paragraph("You asked:", label_user_style))
                story.append(Paragraph(safe_msg, bubble_user_style))
            else:
                story.append(Paragraph("AskMyPDF answered:", label_bot_style))
                story.append(Paragraph(safe_msg, bubble_bot_style))

    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc")))
    story.append(Paragraph("SmartDocAI| Built by Akanksha", meta_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


# ── Session state init ────────────────────────────────────────────────────────

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # list of (role, message) tuples

if "pdf_names" not in st.session_state:
    st.session_state.pdf_names = []


# ── UI Layout ─────────────────────────────────────────────────────────────────

st.markdown('<p class="main-title">📄 SmartDocAI</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload PDFs · Ask anything · Get instant answers &nbsp;|&nbsp; Built by <strong>Akanksha</strong></p>', unsafe_allow_html=True)

col_left, col_right = st.columns([1, 2], gap="large")

# ── LEFT PANEL: Upload & Process ─────────────────────────────────────────────
with col_left:
    st.markdown("### 📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        names = [f.name for f in uploaded_files]
        badges = " ".join([f'<span class="badge">{n}</span>' for n in names])
        st.markdown(f"**Loaded:** {badges}", unsafe_allow_html=True)

    process_btn = st.button("⚡ Process PDFs", use_container_width=True)

    if process_btn:
        if not uploaded_files:
            st.warning("Please upload at least one PDF first.")
        elif not GOOGLE_API_KEY:
            st.error("GOOGLE_API_KEY is missing. Add it to your .env file.")
        else:
            with st.spinner("Reading & indexing your documents…"):
                raw_text = extract_text_from_pdfs(uploaded_files)
                if not raw_text.strip():
                    st.error("Could not extract text. Make sure the PDFs are not scanned images.")
                else:
                    chunks = split_text(raw_text)
                    vector_store = build_vector_store(chunks)
                    st.session_state.vector_store = vector_store
                    st.session_state.pdf_names = [f.name for f in uploaded_files]
                    st.session_state.chat_history = []
                    st.success(f"✅ Indexed {len(chunks)} chunks from {len(uploaded_files)} file(s).")

    if st.session_state.vector_store:
        st.markdown("""
        <div class="info-box">
        🟢 <strong>Ready to chat!</strong><br>
        Ask anything about your uploaded documents in the chat panel.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="info-box">
    <strong>How it works</strong><br><br>
    1. Upload PDF(s)<br>
    2. Click <em>Process PDFs</em><br>
    3. Ask questions in the chat<br><br>
    Powered by <strong>Gemini 1.5 Flash</strong> + <strong>FAISS</strong> + <strong>HuggingFace Embeddings</strong>
    </div>
    """, unsafe_allow_html=True)


# ── RIGHT PANEL: Chat ─────────────────────────────────────────────────────────
with col_right:
    st.markdown("### 💬 Chat with your PDFs")

    # Chat history display
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            st.info("Process your PDFs on the left, then ask a question below.")
        else:
            for role, msg in st.session_state.chat_history:
                if role == "user":
                    st.markdown(f'<div class="chat-bubble-user">🧑 {msg}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-bubble-bot">🤖 {msg}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Input row
    input_col, btn_col = st.columns([5, 1])
    with input_col:
        user_question = st.text_input(
            "Ask a question…",
            placeholder="e.g. What are the main conclusions?",
            label_visibility="collapsed",
            key="user_input",
        )
    with btn_col:
        ask_btn = st.button("Send", use_container_width=True)

    # Clear chat + Download chat as PDF
    if st.session_state.chat_history:
        action_col1, action_col2 = st.columns([1, 1])
        with action_col1:
            if st.button("🗑 Clear chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        with action_col2:
            pdf_bytes = generate_chat_pdf(
                st.session_state.chat_history,
                st.session_state.pdf_names,
            )
            st.download_button(
                label="⬇️ Download Chat PDF",
                data=pdf_bytes,
                file_name=f"askmypdf_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    # Answer generation
    if ask_btn and user_question.strip():
        if st.session_state.vector_store is None:
            st.warning("Please process your PDFs first (left panel).")
        else:
            st.session_state.chat_history.append(("user", user_question))
            with st.spinner("Thinking…"):
                answer = answer_question(st.session_state.vector_store, user_question)
            st.session_state.chat_history.append(("bot", answer))
            st.rerun()