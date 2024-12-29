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

###Clone Repository and Install Dependencies

```bash
git clone https://github.com/ajayjagan2511/InDocQ.git
cd InDocQ
pip install -r requirements.txt
```
