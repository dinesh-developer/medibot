from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama, HuggingFaceHub
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI


"""
import os
HF_TOKEN = os.getenv("HF_TOKEN")

# HuggingFaceEndpoint : Requires paid endpoint deployment, uses HuggingFace API token
llm = HuggingFaceEndpoint(
        endpoint=novita_model_endpoint_url,  # your text-generation endpoint
        model="mistralai/Mistral-7B-Instruct-v0.3", # your text-generation model
        temperature= 0.5,
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=512
    )

    
# HuggingFacePipeline : Downloads model locally
llm = HuggingFacePipeline.from_model_id(
    model_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    model_kwargs={
        "temperature": 0.7,
        "max_new_tokens": 200
    })

    
# InferenceClient in LangChain : Requires custom LLM wrapper
from huggingface_hub import InferenceClient

# For Hugging Face Inference API
os.environ["HF_TOKEN"] = "hf_YOUR_HF_TOKEN"
client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.2")

# For Novita AI (using their OpenAI-compatible endpoint directly)
os.environ["NOVITA_API_KEY"] = "nvt_YOUR_NOVITA_KEY"
client = InferenceClient(base_url="https://api.novita.ai/v3/openai")


# HuggingFaceHub : Uses free HuggingFace model, requires HF_TOKEN in .env
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",  # your text-generation model
    huggingfacehub_api_token=HF_TOKEN,
    model_kwargs={
        "temperature": 0.7,
        "max_new_tokens": 200,
        "top_p": 0.95
    }
)

"""


# Step 1: Define LLM model

def get_llm():
    """
    Returns an instance of the LLM model to be used in the retrieval chain.
    You can uncomment the desired model to use.
    
    Ollama : Downloads model locally, requires Ollama CLI installed. Not required API key.
    ChatGroq : Uses Groq AI model, requires GROQ_API_KEY in .env
    ChatOpenAI : Uses OpenAI model, requires OPENAI_API_KEY in .env
    ChatMistralAI : Uses Mistral AI model, requires MISTRAL_API_KEY in .env

    Note: Ensure you have the necessary API keys set in your environment variables.
    """
    
    # llm = Ollama(model="deepseek-r1:1.5b")
    llm = ChatGroq(model="deepseek-r1-distill-llama-70b")
    # llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    # llm = ChatMistralAI(model="mistral-large-latest", temperature=0.7)
    return llm


# Step 2: Define custom prompt template

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


# Step 3: Load vector store FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db= FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)


# Step 4: Create retrieval chain with LLM and vector store
def create_retrieval_chain():
    llm = get_llm()
    prompt = custom_prompt(CUSTOM_PROMPT_TEMPLATE)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return chain


# Invoke with single query
user_query = input("Enter your query: ")
response = create_retrieval_chain().invoke({"query": user_query})
print("Response:", response['result'])
print("Source Documents:", response['source_documents'])    