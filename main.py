import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)

# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to create a conversational chain for QA
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
     If the answer is not in the provided context, search in the Gemini model directly.\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to search Gemini for the answer
def search_gemini(user_question):
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key)
        response = llm.invoke(user_question)
        return response.content
    except Exception as e:
        return f"Error: {str(e)}"

# Function to handle user input and get the response
def user_input(user_question, knowledge_base):
    if knowledge_base:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()

        context = "\n".join([doc.page_content for doc in docs])
        response = chain({"context": context, "question": user_question}, return_only_outputs=True)
        if "I cannot answer this question from the provided context" in response["output_text"]:
            gemini_response = search_gemini(user_question)
            response["output_text"] = gemini_response

        return response["output_text"]
    else:
        return search_gemini(user_question)

# Function to generate feedback based on user data and knowledge base
def generate_feedback(user_data, knowledge_base):
    feedback_question = f"""
    Generate feedback for a student with the following details:
    - User ID: {user_data['user_id']}
    - User Name: {user_data['user_name']}
    - Email: {user_data['email']}
    - Phone Number: {user_data['phone']}
    - ML Course Completion: {user_data['ml_completion']}%
    - DL Course Completion: {user_data['dl_completion']}%
    - Quant Course Completion: {user_data['quant_completion']}%
    - SQL Course Completion: {user_data['sql_completion']}%
    - Python Course Completion: {user_data['python_completion']}%
    - No. of Weekly Tests: {user_data['weekly_tests']}
    - Average Weekly Test Marks: {user_data['avg_weekly_marks']}%
    - No. of Sectional Tests: {user_data['sectional_tests']}
    - Average Sectional Marks: {user_data['avg_sectional_marks']}%

    Based on the knowledge base, please provide feedback on the following:
    1. Subjects where the student is excelling.
    2. Subjects where the student needs improvement.
    3. Suggestions to improve in the subjects where the student is lagging.
    """
    return user_input(feedback_question, knowledge_base)

# Create a sample pandas DataFrame with user data
user_data = pd.DataFrame({
    'user_id': ['U001', 'U002', 'U003', 'U004', 'U005', 'U006', 'U007', 'U008', 'U009', 'U010'],
    'user_name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Hannah', 'Ivy', 'Jack'],
    'email': ['alice@example.com', 'bob@example.com', 'charlie@example.com', 'david@example.com', 'eve@example.com',
              'frank@example.com', 'grace@example.com', 'hannah@example.com', 'ivy@example.com', 'jack@example.com'],
    'phone': ['123-456-7890', '123-456-7891', '123-456-7892', '123-456-7893', '123-456-7894',
              '123-456-7895', '123-456-7896', '123-456-7897', '123-456-7898', '123-456-7899'],
    'ml_completion': [80, 50, 90, 70, 60, 85, 95, 40, 75, 65],
    'dl_completion': [75, 65, 80, 85, 55, 60, 70, 90, 50, 95],
    'quant_completion': [60, 70, 50, 75, 80, 85, 90, 55, 95, 65],
    'sql_completion': [85, 95, 75, 60, 70, 55, 65, 90, 80, 50],
    'python_completion': [90, 80, 85, 95, 70, 75, 65, 60, 55, 50],
    'weekly_tests': [5, 4, 6, 3, 7, 8, 5, 2, 9, 1],
    'avg_weekly_marks': [88, 76, 90, 85, 60, 75, 95, 70, 80, 65],
    'sectional_tests': [3, 4, 5, 6, 2, 1, 7, 8, 9, 10],
    'avg_sectional_marks': [78, 85, 88, 92, 75, 80, 95, 70, 85, 90]
})

# Main function to set up Streamlit interface
def main():
    st.set_page_config(page_title="Chat with PDF using Gemini💁", layout="wide")
    st.header("Chat with PDF using Gemini💁")

    st.sidebar.header("User Selection")
    selected_user = st.sidebar.selectbox("Select User", user_data['user_name'])
    user_details = user_data[user_data['user_name'] == selected_user].iloc[0]

    st.sidebar.header("User Details")
    st.sidebar.text(f"User ID: {user_details['user_id']}")
    st.sidebar.text(f"Email: {user_details['email']}")
    st.sidebar.text(f"Phone: {user_details['phone']}")
    st.sidebar.text(f"ML Completion: {user_details['ml_completion']}%")
    st.sidebar.text(f"DL Completion: {user_details['dl_completion']}%")
    st.sidebar.text(f"Quant Completion: {user_details['quant_completion']}%")
    st.sidebar.text(f"SQL Completion: {user_details['sql_completion']}%")
    st.sidebar.text(f"Python Completion: {user_details['python_completion']}%")
    st.sidebar.text(f"Weekly Tests: {user_details['weekly_tests']}")
    st.sidebar.text(f"Avg Weekly Marks: {user_details['avg_weekly_marks']}%")
    st.sidebar.text(f"Sectional Tests: {user_details['sectional_tests']}")
    st.sidebar.text(f"Avg Sectional Marks: {user_details['avg_sectional_marks']}%")

    st.header("Upload PDF for Knowledge Base")
    pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)

    knowledge_base = ""

    if st.button("Process PDF"):
        if pdf_docs:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                knowledge_base = raw_text
                st.success("Knowledge base created successfully!")
                st.write("Knowledge Base Content:")
                st.write(knowledge_base)
        else:
            st.warning("Please upload at least one PDF file.")

    if st.button("Generate Feedback"):
        with st.spinner("Generating feedback..."):
            user_info = user_details.to_dict()
            feedback = generate_feedback(user_info, knowledge_base)
            st.write("Feedback: ", feedback)

if __name__ == "__main__":
    main()
