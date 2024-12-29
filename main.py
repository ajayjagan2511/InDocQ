import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db, extract_pdf_text
from PIL import Image
import time

# Initializing session state variables
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

# Streamlit UI/UX
image = Image.open("logo.png")
st.image(image, width=800)
st.title("InDocQ 🦜🔗")
st.subheader("Intelligent Document Q&A, powered by LangChain and LLMs")


uploaded_file = st.file_uploader("Upload a Document", type="pdf")

if uploaded_file is not None:
    pdf_text, pdf_preview = extract_pdf_text(uploaded_file)
    st.write("Preview of extracted PDF text (First 1000 characters):")
    st.text(pdf_preview)
    if st.button("Show Full Text"):
        st.text(pdf_text)

    # Creating Knowledge Base
    if not st.session_state.knowledge_base_created:
        if st.button("Create Knowledge Base"):
            create_vector_db(pdf_text)
            st.session_state.knowledge_base_created = True
            st.success("Knowledge base created successfully!")


if st.session_state.knowledge_base_created:
    st.subheader("Ask Questions about the Document")

  
    if not st.session_state.question_answered:
        question_input_key = f"question_input_{st.session_state.question_counter}"

        if st.session_state.asked_questions:
            st.subheader("Previous Questions")
            for q, ans in st.session_state.asked_questions:
                st.write(f"**Q:** {q}")
                st.write(f"**A:** {ans}")

        question = st.text_input(
            "Ask a question:",
            placeholder="Type your question here and press Enter...",
            key=question_input_key  
        )

        if question:
            st.session_state.current_question = question
            progress_bar = st.progress(0)
            for i in range(100): 
                time.sleep(0.05)  
                progress_bar.progress(i + 1)
            chain = get_qa_chain()  
            response = chain(st.session_state.current_question)
            st.session_state.asked_questions.append((st.session_state.current_question, response["result"]))
            st.write(f"**Question:** {st.session_state.current_question}")
            st.write(f"**Answer:** {response['result']}")
            st.session_state.question_answered = True
            st.success("Answer generated and stored successfully!")
            st.session_state.question_counter += 1
    if st.session_state.question_answered:
        st.session_state.question_answered = False
