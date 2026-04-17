import streamlit as st
from dotenv import load_dotenv
import os
import time
import warnings
import logging

from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

# ------------------ CLEAN LOGS ------------------
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["USER_AGENT"] = "rag-app"

# ------------------ LOAD ENV ------------------
load_dotenv()
groq_api_key = st.secrets["GROQ_API_KEY"]

# ------------------ UI ------------------
st.set_page_config(page_title="Dynamic RAG with Groq", layout="wide")
st.image("PragyanAI_Transperent.png")
st.title("Dynamic RAG with Groq, FAISS, and Llama3")

# ------------------ SESSION STATE ------------------
if "vector" not in st.session_state:
    st.session_state.vector = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload your PDF documents", type="pdf", accept_multiple_files=True
    )

    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                docs = []

                for file in uploaded_files:
                    with open(file.name, "wb") as f:
                        f.write(file.getbuffer())

                    loader = PyPDFLoader(file.name)
                    docs.extend(loader.load())

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )

                final_documents = text_splitter.split_documents(docs)

                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )

                st.session_state.vector = FAISS.from_documents(
                    final_documents, embeddings
                )

                st.success("✅ Documents processed successfully!")
        else:
            st.warning("Please upload at least one document.")

# ------------------ MAIN ------------------
st.header("Chat with your Documents")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile"
)

# ✅ FIXED PROMPT
prompt = ChatPromptTemplate.from_template(
    """
You are an AI assistant. Answer ONLY using the provided context.

<context>
{context}
</context>

Question: {input}

Answer:
"""
)

# ------------------ CHAT HISTORY ------------------
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ------------------ USER INPUT ------------------
if prompt_input := st.chat_input("Ask a question about your documents..."):

    if st.session_state.vector is not None:

        with st.chat_message("user"):
            st.markdown(prompt_input)

        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt_input
        })

        with st.spinner("Thinking..."):
            start = time.process_time()

            # ✅ FIXED RETRIEVER
            retriever = st.session_state.vector.as_retriever()

            docs = retriever.invoke(prompt_input)

            if docs:
                context = "\n\n".join(
                    [doc.page_content for doc in docs[:5]]
                )

                chain = prompt | llm | StrOutputParser()

                response = chain.invoke({
                    "context": context,
                    "input": prompt_input
                })
            else:
                response = "No relevant information found."

            response_time = time.process_time() - start

        # ✅ FIXED RESPONSE DISPLAY
        with st.chat_message("assistant"):
            st.markdown(response)
            st.info(f"Response time: {response_time:.2f} seconds")

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response
        })

    else:
        st.warning("Please process your documents before asking questions.")
