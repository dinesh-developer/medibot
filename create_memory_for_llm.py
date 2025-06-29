from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Step 1: Load PDF documents from a directory
DATA_PATH= "medical_documents/"

def load_documents(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

documents = load_documents(DATA_PATH)
print(f"length of pdf pages: {len(documents)}")


# Step 2: Split the documents into smaller chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

text_chunks = split_documents(documents)
print(f"length of text chunks: {len(text_chunks)}")


# step 3: Create vector embeddings for the text chunks

def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model = get_embedding_model()

# step 4: Store the embeddings in a vector db FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)

