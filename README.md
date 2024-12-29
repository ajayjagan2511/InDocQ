# InDocQ: Intelligent Document Q&A System 🦜🔗

## Overview

InDocQ is an advanced document question-answering system powered by LangChain and Large Language Models (LLMs) and hosted using StreamLit. This application enables users to upload PDF documents and engage in interactive Q&A sessions about the document's content, leveraging the power of semantic search and state-of-the-art language models.

## Technical Architecture

### Components

The system architecture consists of three main components:

$$
\text{PDF} \xrightarrow{\text{Text Extraction}} \text{Vector Database} \xrightarrow{\text{LLM Processing}} \text{Answer Generation}
$$

1. **Document Processing Pipeline**
   - PDF text extraction using PyPDF2
   - Text chunking for optimal processing
   - Vector embeddings generation using HuggingFace

2. **Knowledge Base**
   - FAISS vector store for efficient similarity search
   - Document chunking with configurable token size
   - Persistent storage of embeddings

3. **Query Processing**
   - LangChain for orchestrating the Q&A pipeline
   - Google's Gemini Pro model for answer generation
   - Context-aware response formulation

## Installation

1. Clone Repository and Install Dependencies

```bash
git clone https://github.com/ajayjagan2511/InDocQ.git
cd InDocQ
pip install -r requirements.txt
```

2. Create environment file
```bash
cp .env.example .env
```

3. Update the `.env` file with your API key

## Usage

Launch the application using Streamlit:

```bash
streamlit run main.py
```


### Application Flow

1. **Document Upload**
   - Upload your PDF document through the web interface
     
     <p align="center">
  <img src="Screenshots/WhatsApp Image 2024-12-28 at 23.30.42.jpeg" alt="InDocQ Demo" width="1000"/>
   </p>  

     
   - Preview extracted text for verification

<br/>

2. **Knowledge Base Creation**
   - Click "Create Knowledge Base" to process the document
     
     <p align="center">
  <img src="Screenshots/WhatsApp Image 2024-12-28 at 23.33.31.jpeg" alt="InDocQ Demo" width="1000"/>
   </p>  

   
   - Wait for the vector database initialization
<br/>

         

3. **Question Answering**
   - Enter your questions in the text input field
     
    <p align="center">
  <img src="Screenshots/WhatsApp Image 2024-12-28 at 23.33.55.jpeg" alt="InDocQ Demo" width="1000"/>
   </p>  

   
   - View answers generated based on document context
     
   <p align="center">
  <img src="Screenshots/WhatsApp Image 2024-12-28 at 23.34.02.jpeg" alt="InDocQ Demo" width="1000"/>
   </p>  

   
   - Access history of previous Q&A interactions
     
   <p align="center">
  <img src="Screenshots/WhatsApp Image 2024-12-28 at 23.34.14.jpeg" alt="InDocQ Demo" width="1000"/>
   </p>


## Dependencies

- langchain
- streamlit
- PyPDF2
- FAISS
- Google Generative AI
- Hugging Face Transformers
- python-dotenv
- PIL

## Configuration

Environment Variables

```bash
GOOGLE_API_KEY="your_api_key_here"
```


### Model Parameters

- Embedding Model: `sentence-transformers/all-mpnet-base-v2`
- LLM Model: `gemini-1.5-pro`
- Chunk Size: 512 tokens
- Temperature: 1.0

## Contributing

Contributions are welcome! Here's how you can get involved:

1. **Report Issues**:
   - Found a bug? Have a feature request? Open an issue in the GitHub repository.

2. **Suggest Enhancements**:
   - Propose ideas to improve the algorithm or its implementation.

3. **Submit Pull Requests**:
   - Fork the repository.
   - Make your changes in a new branch.
   - Open a pull request for review.

Please ensure all contributions adhere to the repository's coding standards and include sufficient documentation.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- LangChain for the robust Q&A framework
- Google for the Gemini Pro API
- Streamlit for the intuitive web interface
- FAISS for efficient vector similarity search





