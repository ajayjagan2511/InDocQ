from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
import PyPDF2
from langchain.schema import Document

load_dotenv()  # Load environment variables from .env file
#os.environ["GOOGLE_API_KEY"] = "Place your own API here"


# Create Google Palm LLM model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=1,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectordb_file_path = "faiss_index"

def chunk_text(text, max_tokens=512):
    """
    Splits a large text into smaller chunks, each with a maximum of `max_tokens`.
    """
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in text.split('. '):  # Split by sentences
        sentence_length = len(sentence.split())
        if current_length + sentence_length > max_tokens:
            # Add the current chunk to the list
            chunks.append(' '.join(current_chunk))
            # Start a new chunk
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length

    # Add the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def create_vector_db(pdf_text):
    # Split the text into chunks
    chunks = chunk_text(pdf_text, max_tokens=512)
    
    # Create LangChain Document objects from the chunks
    documents = [Document(page_content=chunk) for chunk in chunks]
    
    # Create a FAISS vector store
    vectordb = FAISS.from_documents(documents=documents, embedding=embeddings)
    
    # Save the vector store locally
    vectordb.save_local(vectordb_file_path)


def get_qa_chain():
    # Load the vector database
    vectordb = FAISS.load_local(vectordb_file_path, embeddings, allow_dangerous_deserialization=True)

    # Create a retriever from the FAISS database
    retriever = vectordb.as_retriever()

    # Define the prompt template for question-answering
    prompt_template = """Given the following context and a question, generate an answer based on this context only.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Initialize RetrievalQA chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return chain


def extract_pdf_text(uploaded_file):
    # Initialize PDF reader
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    
    # Extract text from each page of the PDF
    pdf_text = ""
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()
    pdf_text = pdf_text.replace('\n', ' ')  # Clean up newlines
    
    # Preview first 1000 characters
    pdf_preview = pdf_text[:1000]
    
    return pdf_text, pdf_preview
