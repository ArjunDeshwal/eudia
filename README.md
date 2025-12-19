# Eudexa

**Eudexa** is an intelligent document analysis and automation platform powered by advanced AI models. It allows users to chat with their documents, visualize relationships, simulate contract changes, automate workflows, and analyze negotiations.

## Features

- **🤖 Chat with Documents**: Interactive Q&A with your PDF documents using RAG (Retrieval-Augmented Generation).
- **📄 Document Summarization**: Automatically generates concise summaries of uploaded documents.
- **🕸️ Knowledge Graph**: Visualizes document structure and relationships between sections/clauses.
- **🔮 What-If Analysis**: Simulates the impact of proposed changes to contract clauses, identifying ripple effects and risks.
- **⚡ Automation**: Extracts actionable data (obligations, payments, dates) and triggers external workflows (n8n) for reminders and alerts.
- **⚖️ Negotiation Analysis**: Compares vendor vs. client drafts to identify key issues, risks, and suggested counters.

## Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **LLM**: [Google Gemini](https://deepmind.google/technologies/gemini/) (via `google.generativeai`)
- **OCR**: [Mistral AI](https://mistral.ai/)
- **Embeddings**: [Voyage AI](https://www.voyageai.com/) (`voyage-law-2` model)
- **Vector Database**: [Pinecone](https://www.pinecone.io/)
- **Automation**: [n8n](https://n8n.io/) Webhooks

## Setup & Installation

### Prerequisites

- Python 3.8+
- API Keys for:
    - Mistral AI
    - Google Gemini
    - Voyage AI
    - Pinecone

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd eudia
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**
    Create a `.env` file in the root directory and add your API keys:
    ```env
    MISTRAL_API_KEY=your_mistral_key
    GEMINI_API_KEY=your_gemini_key
    VOYAGE_API_KEY=your_voyage_key
    PINECONE_API_KEY=your_pinecone_key
    N8N_WEBHOOK_URL=your_n8n_webhook_url (optional)
    ```

## Usage

1.  **Run the application:**
    ```bash
    streamlit run app6.py
    ```

2.  **Navigate to the UI:**
    Open your browser and go to `http://localhost:8501`.

3.  **Get Started:**
    -   Upload PDF documents via the sidebar.
    -   Click "Process Documents" to index them.
    -   Use the sidebar to switch between different views (Chat, Knowledge Graph, Automation, etc.).

## Project Structure

- `app6.py`: Main Streamlit application file.
- `doc_graph.py`: Logic for building the document knowledge graph.
- `requirements.txt`: Python dependencies.
- `README.md`: Project documentation.
