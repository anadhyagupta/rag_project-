from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

# Load file
loader = TextLoader("data.txt")
documents = loader.load()

# Split text line by line (IMPORTANT CHANGE)
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=100,
    chunk_overlap=0
)

docs = text_splitter.split_documents(documents)

# Ask user
query = input("Ask something: ").lower()

best_match = ""
max_score = 0

for doc in docs:
    content = doc.page_content.lower()
    words = query.split()

    score = sum(1 for word in words if word in content)

    if score > max_score:
        max_score = score
        best_match = doc.page_content

# Output
if best_match:
    print("\nAnswer:", best_match)
else:
    print("\nAnswer: Not found in data")