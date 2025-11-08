import streamlit as st
import base64
import os
import re
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import httpx

from dotenv import load_dotenv
from mistralai import Mistral
import voyageai
from pinecone import Pinecone, ServerlessSpec
import pinecone
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from doc_graph import build_document_graph

# --- 1. INITIALIZATION ---
st.set_page_config(page_title="Eudexa", layout="wide")
st.title("🤖 Eudexa: Chat and Act with your Documents")

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
N8N_WEBHOOK_URL = os.getenv(
    "N8N_WEBHOOK_URL",
    "https://utk.app.n8n.cloud/webhook-test/dcf2f8b7-bec4-41f6-9a4b-a6d178540908",
)

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

def get_answer_from_model(query, index, chat_history, memory_summary, negotiation_result):
    """Retrieves context, combines with memory and negotiation results, and generates an answer."""
    # Retrieve relevant chunks from the document
    qv = vo.embed([query], model="voyage-law-2", input_type="query").embeddings[0]
    res = index.query(vector=qv, top_k=5, include_metadata=True) # Retrieve more chunks for multi-doc
    matches = res.get("matches", [])
    
    # Fetch document summary
    summary_res = index.fetch(ids=["summary-0"])
    summary_vec = summary_res.vectors.get("summary-0")

    # Build Document Context
    doc_context_parts = []
    references = set()
    if summary_vec and summary_vec.metadata:
        doc_context_parts.append(f"Document Summary:\n{summary_vec.metadata['text']}")
        references.add("Global Summary")
    
    if matches:
        doc_context_parts.append("\nRelevant Chunks:")
        for m in matches:
            if m.id != 'summary-0' and m.metadata and 'text' in m.metadata:
                source = m.metadata.get('source', 'unknown')
                doc_context_parts.append(f"Source: {source}\n{m.metadata['text']}")
                references.add(source)
    
    doc_context = "\n---\n".join(doc_context_parts)

    # Build Conversation History Context
    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history])

    # Build Negotiation Context
    negotiation_context = ""
    if negotiation_result:
        try:
            issues = negotiation_result.get("issues", [])
            overall_summary = negotiation_result.get("overall_summary", "Not available.")
            
            formatted_issues = []
            for i, issue in enumerate(issues, 1):
                formatted_issues.append(
                    f"  - Issue {i}: {issue.get('topic', 'N/A')}\n"
                    f"    - Risk: {issue.get('risk', 'N/A')}\n"
                    f"    - Suggested Counter: {issue.get('suggested_counter', 'N/A')}"
                )
            
            negotiation_context = f"""
### Negotiation Analysis Context
A negotiation analysis was performed. If the user asks about negotiation points, risks, or differences between document versions, use this context.

**Overall Summary:** {overall_summary}

**Key Issues Identified:**
{chr(10).join(formatted_issues) if formatted_issues else "No specific issues were identified."}
"""
        except Exception:
            negotiation_context = f"""
### Negotiation Analysis Context
{json.dumps(negotiation_result, indent=2)}
"""

    prompt = f"""
You are a helpful AI assistant. Your task is to answer the user's question based on the sources of information provided below. The sources are prioritized: Negotiation Context is highest, then Document Context, then Conversation History.

{negotiation_context if negotiation_context else ""}

### Document Context
{doc_context if doc_context_parts else "No document context available."}

### Conversation History
**Summary of earlier conversation:**
{memory_summary if memory_summary else "No prior conversation summary."}

**Recent messages:**
{history_str if history_str else "This is the first message."}

### Instructions
- **Prioritize the most relevant context.** If the user asks about negotiation, use the Negotiation Analysis. If they ask about the uploaded document, use the Document Context. Use Conversation History for follow-ups.
- **Cite Sources:** When you use information from the Document Context, cite the source document name (e.g., `[contract.pdf]`, `[Global Summary]`). You do not need to cite the negotiation or conversation history.
- **Be Concise:** Keep your answers to the point.
- **If Unsure:** If the answer cannot be found in the provided context, state that clearly.

### User's Current Question
{query}
"""
    response = gmodel.generate_content(prompt)
    return response.text.strip(), sorted(list(references))


@st.cache_data(show_spinner=False)
def summarize_messages(_messages_to_summarize, _existing_summary):
    """Summarizes a list of messages to condense the history."""
    if not _messages_to_summarize:
        return _existing_summary

    messages_str = "\n".join([f"{m['role']}: {m['content']}" for m in _messages_to_summarize])
    
    prompt = f"""
You are a conversation summarizer. Your task is to create a concise summary of the provided conversation.
If there is an existing summary, integrate the new conversation turns into it, creating a new, cohesive summary.

Existing Summary:
{_existing_summary if _existing_summary else "None"}

New Conversation Turns:
{messages_str}

Please provide the new, integrated summary:
"""
    try:
        summary_response = gmodel.generate_content(prompt)
        return summary_response.text.strip()
    except Exception as e:
        st.warning(f"Could not summarize messages: {e}")
        return _existing_summary


STRUCTURED_DOC_PROMPT = """
You are a senior legal knowledge engineer. Convert the document text below into structured JSON
with this shape:
{
  "title": "",
  "summary": "",
  "sections": [
    {
      "title": "",
      "summary": "",
      "clauses": [{"id": "", "text": ""}],
      "children": []
    }
  ]
}
Include as many sections/clauses as necessary. Preserve numbering where possible. Return ONLY JSON.
"""

AUTOMATION_EXTRACTION_PROMPT = """
You are an expert legal document analysis and automation AI.
Your task is to read the structured JSON below (extracted from a contract or agreement)
and return a single, consolidated JSON response capturing all relevant actionable data.

Follow these goals carefully:


MAIN OBJECTIVE

Understand and summarize every obligation, payment term, deadline, and risk inside the document.
Classify them by urgency and recommend next actions suitable for automation.


 INPUT

The input JSON will contain contract text segmented into clauses, with entities and metadata.

Example input:
{{ $json.output }}


 OUTPUT STRUCTURE

Return a single JSON object containing these sections:

{
  "obligations": [
    {
      "party_responsible": "",
      "party_email" : "",
      "obligation_description": "",
      "due_date": "",
      "category": "delivery / payment / reporting / renewal / other",
      "risk_level": "low / medium / high",
      "urgency_level": "urgent / soon / normal",
      "confidence": 0.0,
      "recommended_action": "create_calendar_event / send_email / legal_review / store_only"
    }
  ],

  "payments": [
    {
      "payer": "",
      "payee": "",
      "amount": "",
      "currency": "",
      "payment_due_date": "",
      "payment_type": "advance / milestone / recurring / final",
      "penalty_if_delayed": "",
      "risk_level": "",
      "recommended_action": ""
    }
  ],

  "high_risk_clauses": [
    {
      "clause_text": "",
      "reason_for_risk": "",
      "risk_score": 0.0,
      "recommended_action": ""
    }
  ],

  "immediate_actions": [
    {
      "task_description": "",
      "responsible_party": "",
      "urgency_level": "",
      "trigger_date": "",
      "recommended_workflow": "email_alert / calendar_event / report_generation / escalation"
    }
  ],

  "summary": {
    "summary_text": "",
    "key_entities": [],
    "risk_summary": [],
    "action_items": []
  }
}


 ANALYSIS GUIDELINES

- Treat any sentence with "shall", "must", "agree to", "responsible for", "liable for" as an obligation.
- Identify dates, time frames, or phrases like "within 7 days", "by end of month" as deadlines.
- Label urgency:
    - `urgent` → due within 7 days or high risk
    - `soon` → due within 30 days
    - `normal` → beyond 30 days
- Classify risks:
    - High → ambiguous terms, missing indemnity, unlimited liability, penalties, or conflicts
    - Medium → moderate penalties or unclear responsibilities
    - Low → straightforward, compliant obligations
- For payments, detect all mentions of money, invoices, or compensation.
- In “immediate_actions”, include only items with urgency = urgent or risk_level = high.


 OUTPUT FORMAT RULES

- Must be valid JSON (no text before or after).
- Keep date format as YYYY-MM-DD if possible.
- Use concise text and avoid legal jargon.
- Include confidence scores (0-1) for extraction certainty.


 EXAMPLE SUMMARY

If a clause says: “Vendor A shall deliver goods by 15 Dec 2025 and pay ₹1,00,000 within 30 days,”
you should output:
- obligation: deliver goods by 15 Dec 2025
- payment: ₹1,00,000 due in 30 days
- risk: medium (payment delay penalty present)
- immediate action: calendar event for 15 Dec 2025, reminder 7 days prior.


OUTPUT

Now generate the consolidated JSON response following the structure above.
"""

NEGOTIATION_AGENT_PROMPT = """
You are a senior contracts counsel asked to reconcile a vendor draft and a client redline.
Given the JSON payload below, produce STRICT JSON with this structure:
{
  "issues": [
    {
      "topic": "",
      "vendor_position": "",
      "client_position": "",
      "risk": "low|medium|high",
      "impact": "",
      "suggested_counter": ""
    }
  ],
  "overall_summary": "",
  "memo_markdown": ""
}
Focus on material differences only. Memo markdown should be a concise negotiation memo summarizing risks and counters.
"""


def safe_json_parse(raw_text: str) -> Any:
    if not raw_text:
        return {}
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("```json")
        if len(parts) > 1:
            cleaned = parts[1].split("```", 1)[0]
        else:
            cleaned = cleaned.strip("`")
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"raw": raw_text}


@st.cache_data(show_spinner=False)
def generate_structured_doc_json_cached(doc_text: str) -> Dict[str, Any]:
    if not doc_text:
        return {}
    clipped = doc_text[:20000]
    prompt = f"{STRUCTURED_DOC_PROMPT}\n\nDocument Text:\n{clipped}"
    try:
        response = gmodel.generate_content(prompt)
        return safe_json_parse(response.text)
    except Exception as exc:
        return {"error": str(exc), "raw_text_preview": clipped[:5000]}


def cosine_similarity_local(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def ensure_structured_doc_json(force: bool = False) -> Dict[str, Any]:
    if force or not st.session_state.get("structured_doc_json"):
        doc_text = st.session_state.get("ocr_text", "")
        if not doc_text:
            raise ValueError("No OCR text available. Process a document first.")
        structured = generate_structured_doc_json_cached(doc_text)
        st.session_state.structured_doc_json = structured
    return st.session_state.structured_doc_json or {}


def get_relevant_chunks_from_session(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    chunk_texts = st.session_state.get("chunks") or []
    chunk_embs = st.session_state.get("chunk_embeddings") or []
    if not chunk_texts or not chunk_embs:
        return []
    q_emb = vo.embed([query], model="voyage-law-2", input_type="query").embeddings[0]
    scored = []
    for idx, emb in enumerate(chunk_embs):
        score = cosine_similarity_local(q_emb, emb)
        scored.append((score, idx))
    top = sorted(scored, reverse=True)[: max(1, top_k)]
    return [
        {"chunk_id": idx, "score": round(score, 4), "text": chunk_texts[idx]}
        for score, idx in top
        if idx < len(chunk_texts)
    ]


def record_agent_run(name: str, payload: Any, meta: Optional[Dict[str, Any]] = None) -> None:
    entry = {
        "name": name,
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "result": payload,
        "meta": meta or {},
    }
    st.session_state.agent_runs.insert(0, entry)


def run_knowledge_graph_agent() -> Dict[str, Any]:
    structured = ensure_structured_doc_json()
    if not structured or structured.get("error"):
        raise ValueError(f"Structured JSON unavailable: {structured.get('error', 'unknown error')}")
    graph_payload = build_document_graph(structured)
    return {"structured_document": structured, "graph": graph_payload}


def run_what_if_agent(change_instruction: str) -> Dict[str, Any]:
    structured = ensure_structured_doc_json()
    if not structured or structured.get("error"):
        raise ValueError(f"Structured JSON unavailable: {structured.get('error', 'unknown error')}")
    prompt = f"""
You are a contract dependency simulator. Review the structured JSON below and outline the impact of the proposed change.
Respond with JSON: {{"impact":[{{"clause_id":"","title":"","reason":"","severity":"low/medium/high","action":"amend/remove/add/review","proposed_edit":""}}],"ripple":[{{"from":"","to":"","reason":""}}],"notes":""}}

Structured JSON:
{json.dumps(structured, ensure_ascii=False)}

Proposed change:
{change_instruction}
"""
    response = gmodel.generate_content(prompt)
    plan = safe_json_parse(response.text)
    return {"change_instruction": change_instruction, "plan": plan}


def run_automation_agent(question: str) -> Dict[str, Any]:
    relevant_chunks = get_relevant_chunks_from_session(question, top_k=5)
    input_payload = {
        "doc_title": st.session_state.get("doc_title") or "Document",
        "summary": st.session_state.get("summary", ""),
        "question": question,
        "chunks": relevant_chunks,
    }
    prompt = f"""{AUTOMATION_EXTRACTION_PROMPT}\n\nInput JSON:\n{json.dumps(input_payload, ensure_ascii=False)}"""
    response = gmodel.generate_content(prompt)
    extracted = safe_json_parse(response.text)
    result = {"question": question, "input": input_payload, "extracted": extracted}
    st.session_state.last_automation_payload = extracted
    return result


def run_negotiation_agent(vendor_text: str, client_text: str, title: str) -> Dict[str, Any]:
    payload = {
        "doc_title": title or "Contract",
        "vendor_text": vendor_text,
        "client_text": client_text,
    }
    prompt = f"""{NEGOTIATION_AGENT_PROMPT}\n\nPayload:\n{json.dumps(payload, ensure_ascii=False)}"""
    response = gmodel.generate_content(prompt)
    return safe_json_parse(response.text)


def render_what_if_result(data: Dict[str, Any]) -> None:
    change = data.get("change_instruction") or "(missing change instruction)"
    plan = data.get("plan") or {}
    st.markdown(f"**Change Instruction:** {change}")
    impact = plan.get("impact") or []
    if impact:
        st.markdown("### Impacted Clauses")
        for idx, item in enumerate(impact, 1):
            st.markdown(
                f"**{idx}. {item.get('title') or item.get('clause_id') or 'Clause'}**"
            )
            st.write(f"Reason: {item.get('reason', 'n/a')}")
            st.write(f"Severity: {item.get('severity', 'n/a')} · Action: {item.get('action', 'n/a')}")
            if item.get("proposed_edit"):
                st.write(f"Proposed Edit: {item['proposed_edit']}")
            st.markdown("---")
    else:
        st.info("No impact items returned by the agent.")

    ripple = plan.get("ripple") or plan.get("ripples") or []
    if ripple:
        st.markdown("### Ripple Effects")
        for edge in ripple:
            st.write(f"{edge.get('from','?')} → {edge.get('to','?')}: {edge.get('reason','')}" )

    notes = plan.get("notes")
    if notes:
        st.markdown("### Notes")
        st.write(notes)


def build_n8n_payload() -> Dict[str, Any]:
    return {
        "doc_title": st.session_state.get("doc_title"),
        "summary": st.session_state.get("summary"),
        "ocr_text": st.session_state.get("ocr_text"),
        "structured_document": st.session_state.get("structured_doc_json"),
        "automation_payload": st.session_state.get("last_automation_payload"),
        "timestamp": datetime.utcnow().isoformat(),
    }


def send_payload_to_n8n(channel: str) -> Dict[str, Any]:
    payload = build_n8n_payload()
    if not payload.get("ocr_text") and not payload.get("automation_payload"):
        raise ValueError("No data available to send. Process a document first.")
    params = {
        "channel": channel,
        "doc_title": payload.get("doc_title") or "Document",
        "payload": json.dumps(payload, ensure_ascii=False),
    }
    with httpx.Client(timeout=30) as client:
        resp = client.get(N8N_WEBHOOK_URL, params=params)
    if resp.status_code >= 400:
        raise RuntimeError(f"n8n webhook error {resp.status_code}: {resp.text}")
    try:
        return resp.json()
    except ValueError:
        return {"raw": resp.text}


def doc_graph_to_dot(graph_payload: Dict[str, Any]) -> str:
    if not graph_payload:
        return ""
    nodes = graph_payload.get("graph", {}).get("nodes", [])
    edges = graph_payload.get("graph", {}).get("edges", [])
    if not nodes:
        return ""
    lines = [
        "digraph Document {",
        "rankdir=LR;",
        'node [shape=box, style="rounded,filled", fillcolor="#eef2ff", color="#4c1d95", fontname="Inter"];',
        'edge [color="#94a3b8"];',
    ]
    for node in nodes:
        node_id = str(node.get("id"))
        title = (node.get("title") or "Section").strip()
        snippet = (node.get("snippet") or "").strip()
        if len(snippet) > 80:
            snippet = snippet[:77].rstrip() + "…"
        label = f"{title}\n{snippet}" if snippet else title
        label = label.replace("\"", "\\\"")
        lines.append(f'"{node_id}" [label="{label}"];')
    for edge in edges:
        src = str(edge.get("from"))
        dst = str(edge.get("to"))
        if src and dst:
            lines.append(f'"{src}" -> "{dst}";')
    lines.append("}")
    return "\n".join(lines)

# --- 3. STREAMLIT UI ---

# Initialize session state
if "view" not in st.session_state:
    st.session_state.view = "Chat"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "agent_runs" not in st.session_state:
    st.session_state.agent_runs = []
if "structured_doc_json" not in st.session_state:
    st.session_state.structured_doc_json = None
if "ocr_text" not in st.session_state:
    st.session_state.ocr_text = ""
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "chunk_embeddings" not in st.session_state:
    st.session_state.chunk_embeddings = []
if "doc_title" not in st.session_state:
    st.session_state.doc_title = ""
if "last_automation_payload" not in st.session_state:
    st.session_state.last_automation_payload = None
if "vendor_text_box" not in st.session_state:
    st.session_state.vendor_text_box = ""
if "client_text_box" not in st.session_state:
    st.session_state.client_text_box = ""
if "negotiation_output" not in st.session_state:
    st.session_state.negotiation_output = None
if "memory_summary" not in st.session_state:
    st.session_state.memory_summary = ""


def reset_agent_state() -> None:
    st.session_state.agent_runs = []
    st.session_state.structured_doc_json = None
    st.session_state.last_automation_payload = None
    st.session_state.negotiation_output = None
    st.session_state.memory_summary = ""
    st.session_state.view = "Chat"

# --- Sidebar ---
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        if st.button("Process Documents"):
            with st.spinner("Processing documents... This may take a few moments."):
                # 1. Clear old data from Pinecone
                index = get_pinecone_index()
                try:
                    index.delete(delete_all=True)
                except pinecone.exceptions.NotFoundException:
                    pass

                # 2. Process each file
                all_ocr_text = []
                all_chunks = []
                all_vectors = []
                doc_titles = []

                for uploaded_file in uploaded_files:
                    pdf_bytes = uploaded_file.getvalue()
                    file_name = uploaded_file.name
                    doc_titles.append(file_name)

                    ocr_text = get_ocr_text(pdf_bytes)
                    if not ocr_text.strip():
                        st.warning(f"OCR failed to extract text from {file_name}. Skipping this file.")
                        continue
                    
                    all_ocr_text.append(ocr_text)

                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                    chunks = text_splitter.split_text(ocr_text)
                    
                    chunk_embeddings = vo.embed(chunks, model="voyage-law-2", input_type="document").embeddings
                    
                    vectors = [{
                        "id": f"{file_name}-chunk-{i}", "values": chunk_embeddings[i],
                        "metadata": {"text": chunks[i], "source": file_name}
                    } for i in range(len(chunks))]
                    
                    # Add to combined lists
                    all_chunks.extend(chunks)
                    all_vectors.extend(vectors)

                if not all_ocr_text:
                    st.error("Could not extract text from any of the documents.")
                    st.stop()

                # 3. Create a combined summary
                combined_text = "\n\n--- NEW DOCUMENT ---\n\n".join(all_ocr_text)
                summary = generate_summary(combined_text)
                
                # 4. Embed and add summary vector
                summary_embedding = vo.embed([summary], model="voyage-law-2", input_type="document").embeddings[0]
                all_vectors.append({
                    "id": "summary-0", "values": summary_embedding,
                    "metadata": {"text": summary, "source": "Global Summary"}
                })

                # 5. Upsert all vectors to Pinecone
                index.upsert(vectors=all_vectors)

                # 6. Update session state
                st.session_state.summary = summary
                st.session_state.ocr_text = combined_text
                st.session_state.doc_title = ", ".join(doc_titles)
                
                all_chunk_embeddings = [vec['values'] for vec in all_vectors if vec['id'] != 'summary-0']
                st.session_state.chunk_embeddings = all_chunk_embeddings
                st.session_state.chunks = all_chunks

                # Reset state and rerun
                reset_agent_state()
                st.session_state.document_processed = True
                st.session_state.messages = []
                st.success(f"Processed {len(doc_titles)} documents. Ready for chat!")
                st.rerun()

    st.divider()

    if not st.session_state.document_processed:
        st.info("Process a document to see available tools.")
    else:
        st.header("Views")
        view_options = ["Chat", "Document Summary", "Knowledge Graph", "What-If Analysis", "Automation", "Negotiation Analysis"]
        
        st.session_state.view = st.radio(
            "Select a View",
            view_options,
            index=view_options.index(st.session_state.view),
            key="view_selection"
        )


# --- Main Content Area ---
if not st.session_state.document_processed:
    st.info("Please upload and process a PDF document to begin.")
else:
    # --- CHAT VIEW ---
    if st.session_state.view == "Chat":
        st.header('Chat')
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        MEMORY_WINDOW_SIZE = 10  # 5 pairs of user/assistant messages

        if prompt := st.chat_input("Ask a question about the document..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    index = get_pinecone_index()
                    answer, refs = get_answer_from_model(
                        prompt,
                        index,
                        chat_history=st.session_state.messages[:-1],
                        memory_summary=st.session_state.get("memory_summary", ""),
                        negotiation_result=st.session_state.get("negotiation_output"),
                    )
                    response_with_refs = f"{answer}\n\n*References: `{', '.join(refs)}`*" if refs else answer
                    st.markdown(response_with_refs)
            
            st.session_state.messages.append({"role": "assistant", "content": response_with_refs})

            if len(st.session_state.messages) > MEMORY_WINDOW_SIZE:
                messages_to_prune = st.session_state.messages[:-MEMORY_WINDOW_SIZE]
                current_summary = st.session_state.get("memory_summary", "")
                new_summary = summarize_messages(messages_to_prune, current_summary)
                st.session_state.memory_summary = new_summary
                st.session_state.messages = st.session_state.messages[-MEMORY_WINDOW_SIZE:]
            
            st.rerun()

    # --- DOCUMENT SUMMARY VIEW ---
    elif st.session_state.view == "Document Summary":
        st.header("Document Summary")
        st.markdown(st.session_state.summary)

    # --- KNOWLEDGE GRAPH VIEW ---
    elif st.session_state.view == "Knowledge Graph":
        st.header("Knowledge Graph Agent")
        if st.button("Build Knowledge Graph", use_container_width=True):
            with st.spinner("Running knowledge graph agent..."):
                try:
                    result = run_knowledge_graph_agent()
                    record_agent_run("Knowledge Graph Agent", result)
                    st.success("Knowledge graph generated.")
                except Exception as exc:
                    st.error(f"Knowledge graph failed: {exc}")
        
        # Display the output
        kg_run = next((run for run in st.session_state.agent_runs if run['name'] == 'Knowledge Graph Agent'), None)
        if kg_run:
            st.subheader("Latest Graph")
            graph_payload = kg_run.get("result", {}).get("graph", {})
            stats = graph_payload.get("stats", {})
            if stats:
                st.caption(f"Nodes: {stats.get('total_nodes', 0)} · Leaves: {stats.get('leaf_nodes', 0)} · Depth: {stats.get('max_depth', 0)}")
            dot = doc_graph_to_dot(graph_payload)
            if dot:
                st.graphviz_chart(dot)
            else:
                st.info("Graph data unavailable.")
            with st.expander("Structured hierarchy", expanded=False):
                st.json(graph_payload or {})
            structured_doc = kg_run.get("result", {}).get("structured_document")
            if structured_doc:
                with st.expander("Structured contract JSON", expanded=False):
                    st.json(structured_doc)
        else:
            st.info("No knowledge graph has been generated yet.")

    # --- WHAT-IF ANALYSIS VIEW ---
    elif st.session_state.view == "What-If Analysis":
        st.header("What-If Analysis Agent")
        change_instruction = st.text_area("Change Instruction", key="what_if_instruction", placeholder="Extend delivery to 90 days...")
        if st.button("Run What-If Agent", use_container_width=True):
            if not change_instruction.strip():
                st.warning("Provide a change instruction for the What-If agent.")
            else:
                with st.spinner("Simulating impact..."):
                    try:
                        result = run_what_if_agent(change_instruction.strip())
                        record_agent_run("What-If Agent", result)
                        st.success("What-If plan ready.")
                    except Exception as exc:
                        st.error(f"What-If agent failed: {exc}")

        # Display the output
        what_if_run = next((run for run in st.session_state.agent_runs if run['name'] == 'What-If Agent'), None)
        if what_if_run:
            st.subheader("Latest Analysis")
            render_what_if_result(what_if_run.get("result", {}))
        else:
            st.info("No What-If analysis has been run yet.")

    # --- AUTOMATION VIEW ---
    elif st.session_state.view == "Automation":
        st.header("Automation JSON Agent")
        automation_question = st.text_area("Automation Question", key="automation_question", placeholder="List every payment obligation due this quarter")
        if st.button("Run Automation JSON Agent", use_container_width=True):
            if not automation_question.strip():
                st.warning("Enter a question for the automation agent.")
            else:
                with st.spinner("Extracting structured obligations..."):
                    try:
                        result = run_automation_agent(automation_question.strip())
                        record_agent_run("Automation JSON Agent", result)
                        st.success("Automation JSON generated.")
                    except Exception as exc:
                        st.error(f"Automation agent failed: {exc}")
        
        st.subheader("Automated AI Workflows")
        st.caption("Send the latest automation JSON to reminders, messages, and voice in one shot.")
        if st.button("Trigger Reminders · Messages · Voice", use_container_width=True):
            with st.spinner("Triggering n8n workflow..."):
                try:
                    send_payload_to_n8n("all")
                    st.success("n8n workflow triggered.")
                except Exception as exc:
                    st.error(f"n8n webhook failed: {exc}")

        # Display the output
        automation_run = next((run for run in st.session_state.agent_runs if run['name'] == 'Automation JSON Agent'), None)
        if automation_run:
            st.subheader("Latest Extracted JSON")
            st.json(automation_run.get("result", {}))
        else:
            st.info("No automation data has been extracted yet.")

    # --- NEGOTIATION ANALYSIS VIEW ---
    elif st.session_state.view == "Negotiation Analysis":
        st.header("Vendor vs Client Negotiation Agent")
        st.caption("Upload vendor/client drafts, auto-OCR them, and generate a negotiation memo.")
        
        col_v, col_c = st.columns(2)
        with col_v:
            vendor_pdf = st.file_uploader("Vendor PDF", type="pdf", key="vendor_pdf_upload")
        with col_c:
            client_pdf = st.file_uploader("Client PDF", type="pdf", key="client_pdf_upload")

        if st.button("OCR Selected PDFs", key="ocr_vendor_client"):
            with st.spinner("Running OCR on uploaded PDFs..."):
                try:
                    if vendor_pdf:
                        st.session_state.vendor_text_box = get_ocr_text(vendor_pdf.getvalue())
                    if client_pdf:
                        st.session_state.client_text_box = get_ocr_text(client_pdf.getvalue())
                    if not vendor_pdf and not client_pdf:
                        st.info("Upload at least one PDF to OCR.")
                    else:
                        st.success("OCR complete. Review or edit the extracted text below.")
                except Exception as exc:
                    st.error(f"OCR failed: {exc}")

        vendor_text_input = st.text_area("Vendor Clauses", key="vendor_text_box", height=200, help="Paste or OCR vendor clauses")
        client_text_input = st.text_area("Client Clauses", key="client_text_box", height=200, help="Paste or OCR client clauses")
        negotiation_title = st.text_input("Negotiation Title", value=st.session_state.doc_title or "Contract Draft")

        if st.button("Run Negotiation Agent", key="run_negotiation_agent"):
            if not vendor_text_input.strip() or not client_text_input.strip():
                st.warning("Provide both vendor and client text before running the agent.")
            else:
                with st.spinner("Generating negotiation memo..."):
                    try:
                        output = run_negotiation_agent(vendor_text_input.strip(), client_text_input.strip(), negotiation_title)
                        st.session_state.negotiation_output = output
                        st.success("Negotiation analysis ready.")
                    except Exception as exc:
                        st.error(f"Negotiation agent failed: {exc}")

        # Display the output
        if st.session_state.negotiation_output:
            st.subheader("Negotiation Results")
            negotiation_data = st.session_state.negotiation_output
            issues = negotiation_data.get("issues", [])
            if issues:
                for idx, issue in enumerate(issues, 1):
                    st.markdown(f"**Issue {idx}: {issue.get('topic','(untitled)')}**")
                    st.write(f"Vendor: {issue.get('vendor_position','')}")
                    st.write(f"Client: {issue.get('client_position','')}")
                    st.write(f"Risk: {issue.get('risk','n/a')} · Impact: {issue.get('impact','')}")
                    st.write(f"Suggested Counter: {issue.get('suggested_counter','')}")
                    st.divider()
            else:
                st.info("No issues returned by the agent.")
            
            overall = negotiation_data.get("overall_summary")
            if overall:
                st.markdown(f"**Overall Summary**: {overall}")
            
            memo_md = negotiation_data.get("memo_markdown")
            if memo_md:
                st.markdown("### Negotiation Memo")
                st.markdown(memo_md)
        else:
            st.info("No negotiation analysis has been run yet.")