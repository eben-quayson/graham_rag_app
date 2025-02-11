import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# Set up environment variable for Google API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyD1DD5ybchbsYQiRcwyfoDzpRnmiR-gYGM"  # Replace with your actual API key

# Initialize the Google Generative AI model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Initialize Chroma vector store
vector_store = Chroma(embedding_function=embeddings)

# Load your documents
file_path = "./data/paul_graham_essay.txt"
loader = TextLoader(file_path)
documents = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Add documents to the vector store
vector_store.add_documents(texts)

# Create a retrieval chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())

# FastAPI application
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/process_query")
async def process_query(request: QueryRequest):
    response = qa_chain.run(request.query)
    return {"answer": response}
