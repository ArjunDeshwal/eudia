import streamlit as st
import base64
import os
import re
from dotenv import load_dotenv
from mistralai import Mistral
import voyageai
from pinecone import Pinecone, ServerlessSpec
import pinecone
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. INITIALIZATION ---
st.set_page_config(page_title="DocuChat", layout="wide")
st.title("📄 DocuChat: Chat with your Documents")

# Load environment variables
load_dotenv()

# Configure APIs using Streamlit's caching for resources
@st.cache_resource
def configure_clients():
    """Initializes and returns all the necessary API clients."""
    try:
        # Mistral
        mistral_client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        # Gemini
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        gmodel = genai.GenerativeModel("gemini-2.5-flash")
        # VoyageAI
        voyage_client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
        # Pinecone
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        return mistral_client, gmodel, voyage_client, pc
    except KeyError as e:
        st.error(f"API key not found in environment variables: {e}. Please check your .env file.")
        st.stop()

mistral_client, gmodel, vo, pc = configure_clients()

INDEX_NAME = "streamlit-doc-chat-demo"

# --- 2. CORE LOGIC (Refactored from ocr.py) ---

@st.cache_data(show_spinner=False)
def get_ocr_text(_pdf_bytes):
    """Performs OCR on PDF bytes and returns the extracted text."""
    base64_pdf = base64.b64encode(_pdf_bytes).decode('utf-8')
    ocr_response = mistral_client.ocr.process(
        model="mistral-ocr-latest",
        document={"type": "document_url", "document_url": f"data:application/pdf;base64,{base64_pdf}"}
    )
    ocr_text = ""
    for page in ocr_response.pages:
        if getattr(page, "markdown", None):
            ocr_text += page.markdown + "\n\n"
    if not ocr_text.strip() and getattr(ocr_response, "text", None):
        ocr_text = ocr_response.text
    return ocr_text

@st.cache_data(show_spinner=False)
def generate_summary(_ocr_text):
    """Generates a summary for the given text."""
    try:
        summary_prompt = f"""
Provide a concise, one-paragraph summary of the following document text.
Focus on the main purpose, key parties, and primary subject matter.

Document Text (first 15000 chars):
{_ocr_text[:15000]}
"""
        summary_response = gmodel.generate_content(summary_prompt)
        return summary_response.text.strip()
    except Exception as e:
        st.error(f"Could not generate summary: {e}")
        return ""

def get_pinecone_index():
    """Creates or connects to a Pinecone index."""
    if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
        st.write(f"Creating new Pinecone index: '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=1024,  # voyage-law-2 model dimension
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(INDEX_NAME)

def process_and_embed(ocr_text, summary_text, uploaded_file_name):
    """Chunks text, embeds, and upserts to Pinecone."""
    index = get_pinecone_index()
    try:
        index.delete(delete_all=True)  # Clear index for the new document
    except pinecone.exceptions.NotFoundException:
        # This can happen if the default namespace doesn't exist yet on a new index.
        pass

    # Chunk the text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_text(ocr_text)
    
    # Embed chunks and summary
    chunk_embeddings = vo.embed(chunks, model="voyage-law-2", input_type="document").embeddings
    summary_embedding = vo.embed([summary_text], model="voyage-law-2", input_type="document").embeddings[0]

    # Upsert vectors
    vectors = [{
        "id": f"chunk-{i}", "values": chunk_embeddings[i],
        "metadata": {"text": chunks[i], "source": uploaded_file_name}
    } for i in range(len(chunks))]
    vectors.append({
        "id": "summary-0", "values": summary_embedding,
        "metadata": {"text": summary_text, "source": uploaded_file_name}
    })

    index.upsert(vectors=vectors)
    return index

def get_answer_from_model(query, index):
    """Retrieves context from Pinecone and generates an answer."""
    # Retrieve relevant chunks
    qv = vo.embed([query], model="voyage-law-2", input_type="query").embeddings[0]
    res = index.query(vector=qv, top_k=4, include_metadata=True)
    matches = res.get("matches", [])
    
    # Fetch summary
    summary_res = index.fetch(ids=["summary-0"])
    summary_vec = summary_res.vectors.get("summary-0")

    # Build context
    context_parts = []
    references = set()
    if summary_vec and summary_vec.metadata:
        context_parts.append(f"Document Summary:\n{summary_vec.metadata['text']}")
        references.add("summary-0")
    
    if matches:
        context_parts.append("\nRelevant Chunks:")
        for m in matches:
            if m.id != 'summary-0' and m.metadata and 'text' in m.metadata:
                context_parts.append(m.metadata['text'])
                references.add(m.id)

    if len(context_parts) < 2:
        return "Insufficient context to answer.", []

    context = "\n---\n".join(context_parts)
    prompt = f"""
You are a helpful assistant. Answer the user's question based ONLY on the context provided below.
The context includes a 'Document Summary' and 'Relevant Chunks'.

- Use the summary for general questions and the chunks for specific details.
- If the answer is not in the context, state "I cannot answer based on the provided context."

Context:
{context}

Question:
{query}
"""
    response = gmodel.generate_content(prompt)
    return response.text.strip(), sorted(list(references))

# --- 3. STREAMLIT UI ---

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False
if "summary" not in st.session_state:
    st.session_state.summary = ""

# Sidebar for file upload
with st.sidebar:
    st.header("1. Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        if st.button("Process Document"):
            with st.spinner("Processing document... This may take a few moments."):
                pdf_bytes = uploaded_file.getvalue()
                
                ocr_text = get_ocr_text(pdf_bytes)
                if not ocr_text.strip():
                    st.error("OCR failed to extract text. Please try another document.")
                else:
                    summary = generate_summary(ocr_text)
                    st.session_state.summary = summary
                    
                    process_and_embed(ocr_text, summary, uploaded_file.name)
                    
                    st.session_state.document_processed = True
                    st.session_state.messages = [] 
                    st.success("Document is ready for chat!")
                    st.rerun()

# Main content area
if not st.session_state.document_processed:
    st.info("Please upload and process a PDF document to begin chatting.")
else:
    # Display Summary
    st.header("2. Document Summary")
    with st.expander("Click to view the document summary", expanded=True):
        st.markdown(st.session_state.summary)

    # Chat interface
    st.header("3. Chat with the Document")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about the document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                index = get_pinecone_index()
                answer, refs = get_answer_from_model(prompt, index)
                response_with_refs = f"{answer}\n\n*References: `{', '.join(refs)}`*" if refs else answer
                st.markdown(response_with_refs)
        
        st.session_state.messages.append({"role": "assistant", "content": response_with_refs})
