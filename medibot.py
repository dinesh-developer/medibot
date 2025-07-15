import streamlit as st
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return db

def custom_prompt(custom_prompt_template):
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )

llm = ChatGroq(model="deepseek-r1-distill-llama-70b")

def main():
    st.title("MediBot: Your Medical Assistant")
    st.write("Welcome to MediBot, your personal medical assistant powered by AI.")
    st.write("Ask me any medical question, and I'll do my best to provide you with accurate information.")
    st.write("Please note that this is not a substitute for professional medical advice.")
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Ask your question here:")
    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
            Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
            Context: {context}
            Question: {question}

            Start the answer directly, no small talk, no preamble, no "I think", just the answer.
        """

        try:
            vectorstore = load_vector_store()
            if vectorstore is None:
                st.error("Failed to load the vector store.")
            chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )
            
            response = chain.invoke({"query": prompt})
            result = response['result']
            source_documents = response['source_documents']
            final_response = result + '\n\n'
            
            final_response += "--- Documents ---\n\n"
            for index, doc in enumerate(source_documents, start=1):
                final_response += f"{index}: {doc}\n\n"
            # final_response += str(source_documents)
            # response = "I'm Medibot, your AI medical assistant. I can help you with general medical information, but please consult a healthcare professional for specific medical advice."
            st.chat_message('assistant').markdown(final_response)
            st.session_state.messages.append({'role': 'assistant', 'content': final_response})
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()