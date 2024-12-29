import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db, extract_pdf_text
from PIL import Image
import time

# Initialize session state variables
if "knowledge_base_created" not in st.session_state:
    st.session_state.knowledge_base_created = False
if "knowledge_base_path" not in st.session_state:
    st.session_state.knowledge_base_path = None
if "asked_questions" not in st.session_state:
    st.session_state.asked_questions = []
if "current_question" not in st.session_state:
    st.session_state.current_question = ""
if "question_answered" not in st.session_state:
    st.session_state.question_answered = False  # Flag to track if the current question is answered
if "question_counter" not in st.session_state:
    st.session_state.question_counter = 0  # Initialize a counter for unique keys

# Open an image file for the logo
image = Image.open("logo.png")


# Display the logo image with width and caption
st.image(image, width=800)

# Streamlit UI for Document Upload and Q&A
st.title("InDocQ 🦜🔗")
st.subheader("Intelligent Document Q&A, powered by LangChain and LLMs")

# File uploader for PDF
uploaded_file = st.file_uploader("Upload a Document", type="pdf")

if uploaded_file is not None:
    # Extract text from the PDF
    pdf_text, pdf_preview = extract_pdf_text(uploaded_file)

    # Display extracted text preview
    st.write("Preview of extracted PDF text (First 1000 characters):")
    st.text(pdf_preview)

    # Optionally display the full text
    if st.button("Show Full Text"):
        st.text(pdf_text)

    # Button to create a knowledge base
    if not st.session_state.knowledge_base_created:
        if st.button("Create Knowledge Base"):
            # Create vector database from the PDF text
            create_vector_db(pdf_text)

            # Update session state
            st.session_state.knowledge_base_created = True
            st.success("Knowledge base created successfully!")

# Main Q&A Interface (if the knowledge base exists)
if st.session_state.knowledge_base_created:
    st.subheader("Ask Questions about the Document")

    # Ask the user for a question (if it's not answered yet)
    if not st.session_state.question_answered:
        question_input_key = f"question_input_{st.session_state.question_counter}"

        # Display previously asked questions and their answers
        if st.session_state.asked_questions:
            st.subheader("Previous Questions")
            for q, ans in st.session_state.asked_questions:
                st.write(f"**Q:** {q}")
                st.write(f"**A:** {ans}")

        # Ask the question
        question = st.text_input(
            "Ask a question:",
            placeholder="Type your question here and press Enter...",
            key=question_input_key  # Unique key for each question input
        )

        if question:
            # Update session state with the current question
            st.session_state.current_question = question

            # Display the loading bar
            progress_bar = st.progress(0)

            # Simulate progress and generate response
            for i in range(100):
                time.sleep(0.05)  # Simulate time taken to generate an answer
                progress_bar.progress(i + 1)

            # Process the current question with LangChain
            chain = get_qa_chain()  # Replace this with your retrieval function
            response = chain(st.session_state.current_question)

            # Store the question and the response
            st.session_state.asked_questions.append((st.session_state.current_question, response["result"]))
            st.write(f"**Question:** {st.session_state.current_question}")
            st.write(f"**Answer:** {response['result']}")

            # Mark the question as answered
            st.session_state.question_answered = True

            # Inform the user
            st.success("Answer generated and stored successfully!")

            # Increment the question counter for the next unique key
            st.session_state.question_counter += 1

    # Reset flag to allow for the next question once the current question is answered
    if st.session_state.question_answered:
        st.session_state.question_answered = False
