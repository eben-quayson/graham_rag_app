import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv


load_dotenv()

# Set up environment variable for Google API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

system_prompt = """
You are a happy and bubbly assistant. You will be given a question and you will answer it. 
If the answer is not in the documents, you will decline to answer politely.
"""

# Initialize the Google Generative AI model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", system_prompt=system_prompt, api_key=GOOGLE_API_KEY)

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
    """
    A class representing a query request to the API.

    Attributes:
        query (str): The query string.
    
    query: str
"""

@app.post("/process_query")
async def process_query(request: QueryRequest):
    """
    Process a user query and return the result.

    Args:
        request (QueryRequest): A QueryRequest object containing the query string.

    Returns:
        dict: A dictionary containing the answer to the query.
    """

    # Run the query through the retrieval chain
    response = qa_chain.invoke(request.query)
    
    # Return the answer to the query
    return {"answer": response}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
