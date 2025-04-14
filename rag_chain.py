import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

def load_documents(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

def create_vectorstore(docs):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    return vectorstore

def create_rag_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="models/chat-bison-001",  # ✅ Use supported model
        temperature=0.2,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    retriever = vectorstore.as_retriever()
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
