import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFaceEndpoint
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter

def load_pdf(file_path):
    """
    Loads a PDF file and splits it into text chunks.
    """
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()

def create_vector_store(documents):
    """
    Creates a FAISS vector store from document chunks using HuggingFace embeddings.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(documents, embeddings)

def build_qa_chain(vector_store, api_token):
    """
    Creates the QA chain using HuggingFaceEndpoint (with a smaller model)
    and the provided vector store.
    """
    llm = HuggingFaceEndpoint(
        repo_id="google/flan-t5-small",      # A smaller model within free-tier limits
        task="text2text-generation",           # Task specification for T5-based models
        huggingfacehub_api_token=api_token,
        temperature=0.5,
        max_new_tokens=500
    )
    # Build the QA chain using the "stuff" chain type (refer to migration guides for alternatives)
    return load_qa_chain(llm, chain_type="stuff")

def main():
    print("📄 Welcome to ChatWithDocs (Terminal Edition)")
    pdf_path = input("Enter path to your PDF file: ").strip()

    if not os.path.isfile(pdf_path):
        print("❌ File not found. Please check the path and try again.")
        return

    print("📚 Loading and processing document...")
    try:
        # Load and split the PDF into chunks
        documents = load_pdf(pdf_path)
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)
        
        # Create vector store using embeddings
        vector_store = FAISS.from_documents(docs, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
        
        # Retrieve the Hugging Face API token from the environment
        api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not api_token:
            print("❌ Error: HUGGINGFACEHUB_API_TOKEN environment variable is not set.")
            return
        
        # Build the QA chain using the LLM endpoint
        qa_chain = build_qa_chain(vector_store, api_token)
    except Exception as e:
        print(f"❌ Error initializing the document or embeddings: {e}")
        return

    print("✅ Document ready! Ask your questions below.\n")

    while True:
        query = input("🤔 Ask a question (or type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            print("👋 Goodbye!")
            break

        try:
            # Retrieve relevant document chunks via similarity search
            relevant_docs = vector_store.similarity_search(query)
            # Get answer from QA chain; here we use .invoke() as .run() is deprecated
            response = qa_chain.invoke({"input_documents": relevant_docs, "question": query})
            print(f"\n🧠 Answer:\n{response['output_text']}\n")
        except Exception as e:
            print(f"❌ Error while processing the question: {e}")

if __name__ == "__main__":
    main()
