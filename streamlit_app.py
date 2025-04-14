import streamlit as st
import tempfile
import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyB2J1hXDVQEKX8UTt6NZPkHG1XPWtARXgE"


from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Function to load documents from an uploaded PDF file
def load_docs_from_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    loader = PyMuPDFLoader(tmp_file_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

# Function to create vectorstore
def create_vectorstore(docs):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embeddings-001")
    return FAISS.from_documents(docs, embeddings)

# Function to create RetrievalQA chain
def create_qa_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)
    retriever = vectorstore.as_retriever()
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Streamlit App UI
st.title("🔍 Ask your PDF (Gemini + LangChain)")

uploaded_file = st.file_uploader("📄 Upload your PDF file", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Processing your document..."):
        docs = load_docs_from_uploaded_file(uploaded_file)
        vectorstore = create_vectorstore(docs)
        qa_chain = create_qa_chain(vectorstore)

    st.success("Document processed! Ask a question below:")

    query = st.text_input("💬 Ask a question:")

    if query:
        with st.spinner("Searching for an answer..."):
            response = qa_chain.run(query)
            st.write("✅ Answer:", response)
