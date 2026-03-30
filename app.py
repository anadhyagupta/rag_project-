import streamlit as st
import speech_recognition as sr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

st.set_page_config(page_title="PDF RAG Chatbot", layout="wide")

# 🔝 FIXED HEADER
st.markdown("""
<style>
header {visibility: hidden;}

.fixed-header {
    position: fixed;
    top: 0;
    width: 100%;
    background-color: #0e1117;
    color: white;
    text-align: center;
    padding: 15px;
    font-size: 26px;
    font-weight: bold;
    z-index: 9999;
}

.block-container {
    padding-top: 80px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="fixed-header">🤖 PDF RAG Chatbot</div>', unsafe_allow_html=True)

# 🔥 LOAD DATA
@st.cache_resource
def load_data():
    loader = PyPDFLoader("notes.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    documents = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(documents, embeddings)

vectorstore = load_data()

# 💬 HISTORY
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "mic_on" not in st.session_state:
    st.session_state.mic_on = False

# 🎤 MIC FUNCTION
def speech_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Listening...")
        audio = r.listen(source)
    try:
        return r.recognize_google(audio)
    except:
        return ""

# 📂 SIDEBAR
st.sidebar.title("📜 Chat History")
for q, a in st.session_state.chat_history:
    st.sidebar.write(f"Q: {q}")
    st.sidebar.write(f"A: {a[:50]}...")
    st.sidebar.write("---")

# 🧠 CLEAN TEXT
def clean_text(text):
    lines = text.split("\n")
    clean_lines = []

    for line in lines:
        line = line.strip()

        if not line:
            continue
        if len(line) < 40:
            continue
        if any(word in line.lower() for word in ["name", "prn", "roll", "unit", "ques", "internal"]):
            continue

        clean_lines.append(line)

    return " ".join(clean_lines[:2])

# 💬 SHOW CHAT
for q, a in st.session_state.chat_history:
    st.chat_message("user").write(q)
    st.chat_message("assistant").write(a)

# 🎤 MIC TOGGLE BUTTON
col1, col2 = st.columns([9,1])

with col2:
    if st.button("🎙️"):
        st.session_state.mic_on = not st.session_state.mic_on

# 💬 INPUT (ChatGPT style)
query = st.chat_input("Ask your question...")

# 🎤 IF MIC ON
if st.session_state.mic_on:
    st.info("🎙️ Recording... (Click mic again to stop)")
    voice_text = speech_to_text()
    if voice_text:
        query = voice_text
        st.session_state.mic_on = False

# 🚀 PROCESS QUERY
if query:
    results = vectorstore.similarity_search(query, k=3)

    answer_parts = []

    for doc in results:
        cleaned = clean_text(doc.page_content)

        if cleaned and cleaned not in answer_parts:
            answer_parts.append(cleaned)

        if len(answer_parts) >= 2:
            break

    answer = "\n\n".join(answer_parts)

    if not answer:
        answer = "No relevant answer found."

    st.session_state.chat_history.append((query, answer))
    st.rerun()