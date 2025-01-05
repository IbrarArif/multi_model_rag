# Multi-Model RAG Dog Health Analyzer

Welcome to the **Dog Health Analyzer**! This application utilizes Retrieval-Augmented Generation (RAG) with advanced AI models to process PDFs and provide actionable insights related to dog health.

---

## Features

- **Multi-Model Support**: Combines GPT-3.5, GPT-4 Vision, and embeddings for robust analysis.
- **PDF Parsing**: Extracts text, tables, and images from uploaded PDFs.
- **Summarization**: Generates summaries for text and tables.
- **Image Analysis**: Describes and analyzes images related to dog health.
- **Interactive Q&A**: Answers your queries by retrieving relevant documents using vector-based search.

---

## How to Run

### Prerequisites

- Python 3.9+
- OpenAI API Key

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/IbrarArif/multi_model_rag.git
   cd multi_model_rag
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

---

## Usage

1. **Upload PDF**: Upload a PDF via the sidebar.
2. **Processing**: Automatically processes text, tables, and images.
3. **Ask Questions**: Enter your question, and the app retrieves and displays the answer.

---

## File Structure

```plaintext
.
├── app.py               # Main application script
├── requirements.txt     # List of required dependencies
├── README.md            # Project documentation
└── images/              # Directory for extracted images
```

---

## Environment Variables

Set up your OpenAI API key:
```bash
export OPENAI_API_KEY='your_openai_api_key'
```

---

## Key Technologies

- **LangChain**: For chaining LLM tasks and managing embeddings.
- **FAISS**: For efficient vector-based document retrieval.
- **Streamlit**: For an interactive web interface.
- **Unstructured**: For PDF parsing and data extraction.

---

For more information, visit the [GitHub Repository](https://github.com/IbrarArif/multi_model_rag).

