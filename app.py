import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

st.title("🤖 PDF RAG Chatbot")
st.write("Ask questions from your PDF 📄")

# Load PDF
loader = PyPDFLoader("notes.pdf")
docs = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
documents = text_splitter.split_documents(docs)

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vector store
vectorstore = FAISS.from_documents(documents, embeddings)

# Input
query = st.text_input("💬 Ask your question:")

if query:
    results = vectorstore.similarity_search(query, k=2)

    st.subheader("📌 Answer:")

    found = False

    for doc in results:
        text = str(doc.page_content)
        sentences = text.split(".")

        for sent in sentences:
            sent = sent.strip()

            if len(sent) < 40:
                continue
            if any(x in sent for x in ["Name", "PRN", "Roll", "Ques", "Define", "Describe"]):
                continue
            if sent.isdigit():
                continue

            st.success(sent + ".")
            found = True
            break

        if found:
            break

    if not found:
        st.error("No relevant answer found.")