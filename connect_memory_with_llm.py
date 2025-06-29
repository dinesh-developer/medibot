import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Setup LLM (Mistral with HuggingFace)

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
print(f"hf_token: {HF_TOKEN}")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is not set.")

HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def get_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature= 0.5,
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=512,
        provider="hf-inference", # <--- Use Hugging Face's own inference endpoint
        task="conversational"
    )
    print("LLM ", llm)
    return llm


# Step 2: Connect to the vector database (FAISS) and create chain

CUSTOM_PROMPT_TEMPLATE = """
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context: {context}
Question: {question}

Start the answer directly, no small talk, no preamble, no "I think", just the answer.
"""

def custom_prompt(custom_prompt_template):
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )

# Load database
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db= FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)


# Create RetrievalQA chain
def create_retrieval_chain():
    llm = get_llm(HUGGINGFACE_REPO_ID)
    prompt = custom_prompt(CUSTOM_PROMPT_TEMPLATE)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return chain


# Invoke with single query
user_query = input("Enter your query: ")
response = create_retrieval_chain().invoke({"query": user_query})
print("Response:", response['result'])
print("Source Documents:", response['source_documents'])