# MAS_work
https://mas-work.onrender.com/
# Feedback Mechanism with AI

This project implements an AI-driven feedback mechanism using a Retrieval-Augmented Generation (RAG) engine, enhancing student feedback through personalized responses based on their performance data.

## Features

- **AI Feedback Generation**: Uses a RAG system for personalized feedback.
- **Student Data Integration**: Leverages course performance data to generate tailored insights.
- **Interactive Q&A**: Allows querying of student data for performance insights.

## Technologies Used

- **Streamlit**: Web interface for the app.
- **LangChain**: Manages the RAG engine and AI integration.
- **Google Gemini**: Powers the AI models for feedback and queries.
- **Pandas**: Processes student data for feedback generation.
- **FAISS**: Stores embeddings for efficient search.

## Setup Instructions

1. **Install Dependencies**:
    ```bash
    pip install streamlit langchain google-generativeai faiss-cpu pandas python-dotenv
    ```

2. **Configuration**:
    - Create a `.env` file with your Google API key:
    ```
    GOOGLE_API_KEY=your_api_key
    ```

3. **Run the App**:
    ```bash
    streamlit run app.py
    ```

## Usage

1. **Upload Data**: Upload student performance data to generate feedback.
2. **Select Student**: Choose a student to view their personalized feedback.
3. **Ask Questions**: Query data for performance-based insights.
