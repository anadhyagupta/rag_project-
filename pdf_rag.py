from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Step 1: Load PDF
loader = PyPDFLoader("notes.pdf")
docs = loader.load()

# Step 2: Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
documents = text_splitter.split_documents(docs)

# Step 3: Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Step 4: Vector DB
vectorstore = FAISS.from_documents(documents, embeddings)

# Step 5: Chat loop
while True:
    query = input("\nAsk something (or type 'exit'): ")

    if query.lower() == "exit":
        break

    # Search relevant chunks
    results = vectorstore.similarity_search(query, k=2)

    print("\n--- Answer ---\n")

    found = False

    for doc in results:
        text = str(doc.page_content)

        sentences = text.split(".")

        for sent in sentences:
            sent = sent.strip()

            # Skip useless lines
            if len(sent) < 40:
                continue
            if any(x in sent for x in ["Name", "PRN", "Roll", "Ques", "Define", "Describe"]):
                continue
            if sent.isdigit():
                continue
            if sent.startswith("•") or sent.startswith("-"):
                continue

            print(sent + ".")
            print("-----")
            found = True
            break

        if found:
            break

    if not found:
        print("No relevant answer found.")