import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

st.title("📄 PDF RAG Chatbot")

# Load PDF
loader = PyPDFLoader("notes.pdf")
docs = loader.load()

# Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
documents = text_splitter.split_documents(docs)

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vector DB
vectorstore = FAISS.from_documents(documents, embeddings)

# Input box
query = st.text_input("Ask something from PDF:")

if query:
    results = vectorstore.similarity_search(query, k=2)

    st.subheader("Answer:")

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
            if sent.startswith("•") or sent.startswith("-"):
                continue

            st.write(sent + ".")
            found = True
            break

        if found:
            break

    if not found:
        st.write("No relevant answer found.")