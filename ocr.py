import base64
import os,json,re
from dotenv import load_dotenv
from mistralai import Mistral
import voyageai
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()
Mistral_api = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=Mistral_api)

# Configure Gemini Model
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
gmodel = genai.GenerativeModel("gemini-2.5-flash")

def encode_pdf(pdf_path):
    with open(pdf_path, "rb") as pdf_file:
        return base64.b64encode(pdf_file.read()).decode('utf-8')

pdf_path = "affidavit.pdf"
base64_pdf = encode_pdf(pdf_path)

ocr_response = client.ocr.process(
    model="mistral-ocr-latest",
    document={
        "type": "document_url",
        "document_url": f"data:application/pdf;base64,{base64_pdf}" 
    },
    include_image_base64=True
)

vo = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
# -------- CHUNKING + SUMMARY PIPELINE --------
# 1) Extract full text from OCR
ocr_text = ""
for page in ocr_response.pages:
    if getattr(page, "markdown", None):
        ocr_text += page.markdown + "\n\n"
if not ocr_text.strip() and getattr(ocr_response, "text", None):
    ocr_text = ocr_response.text

assert ocr_text.strip(), "OCR produced empty text."

# 2) Generate a summary of the whole document
print("📄 Generating document summary...")
try:
    summary_prompt = f"""
Please provide a concise, one-paragraph summary of the following document text.
Focus on the main purpose of the document, the key parties involved, and the primary subject matter.

Document Text (first 15000 chars):
{ocr_text[:15000]}
"""
    summary_response = gmodel.generate_content(summary_prompt)
    summary_text = summary_response.text.strip()
    print("✅ Summary generated.")
except Exception as e:
    print(f"⚠️ Could not generate summary: {e}")
    summary_text = ""

# 3) Chunk the text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=150, length_function=len
)
chunks = text_splitter.split_text(ocr_text)
print(f"📄 Split document into {len(chunks)} chunks.")

# 4) Embed chunks and summary
chunk_embeddings = vo.embed(chunks, model="voyage-law-2", input_type="document").embeddings
if summary_text:
    summary_embedding = vo.embed([summary_text], model="voyage-law-2", input_type="document").embeddings[0]

# 5) Create index and upsert vectors
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = "affidavit-demo"
if index_name in [i.name for i in pc.list_indexes()]:
    print("⚠️ Index already exists. Deleting and recreating for fresh start.")
    pc.delete_index(index_name)

pc.create_index(
    name=index_name,
    dimension=len(chunk_embeddings[0]),
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)
index = pc.Index(index_name)

vectors = [{
    "id": f"affidavit-chunk-{i}",
    "values": chunk_embeddings[i],
    "metadata": {"text": chunks[i], "source": pdf_path, "chunk_num": i}
} for i in range(len(chunks))]

if summary_text:
    vectors.append({
        "id": "summary-0",
        "values": summary_embedding,
        "metadata": {"text": summary_text, "source": pdf_path}
    })

index.upsert(vectors=vectors, namespace="default")
print(f"✅ Upserted {len(vectors)} vectors (including summary).")


# -------------------------- QUERY APP (summary-aware) --------------------------
print("\n🔹 Ready to query the OCR’d document. Type 'exit' to quit.\n")

def retrieve_chunks(query: str, k: int = 5):
    qv = vo.embed([query], model="voyage-law-2", input_type="query").embeddings[0]
    res = index.query(vector=qv, top_k=k, include_metadata=True, namespace="default")
    return res.get("matches", [])

def answer(query: str, max_chars: int = 8000):
    matches = retrieve_chunks(query)
    
    context_parts = []
    references = []
    used_length = 0
    
    # Fetch and add summary to context first
    if summary_text:
        try:
            summary_vec_response = index.fetch(ids=["summary-0"], namespace="default")
            summary_vec = summary_vec_response.vectors.get('summary-0')
            if summary_vec:
                fetched_summary = (summary_vec.metadata or {}).get('text', '')
                if fetched_summary:
                    summary_context = f"Document Summary:\n{fetched_summary}"
                    context_parts.append(summary_context)
                    references.append({"id": "summary-0", "score": 1.0})
                    used_length += len(summary_context)
        except Exception as e:
            print(f"⚠️ Could not fetch summary: {e}")

    # Add other chunks
    if matches:
        context_parts.append("\nRelevant Chunks:")
        for m in matches:
            text = (m.get("metadata") or {}).get("text", "")
            if not text or m['id'] == 'summary-0':
                continue
            
            if used_length + len(text) > max_chars:
                break
            context_parts.append(text)
            references.append({"id": m["id"], "score": m.get("score", 0)})
            used_length += len(text)

    if len(context_parts) < 2 and not summary_text:
        return "Insufficient context to answer the question.", []

    context = "\n---\n".join(context_parts)
    prompt = f"""
You are a precise legal assistant. Answer the user's question based on the context below.
The context may include a 'Document Summary' and 'Relevant Chunks'.

- First, review the 'Document Summary' (if available) to understand the overall context.
- Then, use the detailed 'Relevant Chunks' to answer the specific question.
- If the question is general (e.g., "what is this document about?"), the summary should be your primary source.
- For specific details, rely on the text chunks.
- Cite your sources using the chunk IDs, like [affidavit-chunk-3] or [summary-0].
- If the answer isn't in the context, state "Insufficient context."

Context:
{context}

Question:
{query}
"""
    resp = gmodel.generate_content(prompt)
    return resp.text.strip(), references

while True:
    q = input("Query: ").strip()
    if q.lower() in {"exit", "quit"}:
        print("👋 Bye!")
        break
    try:
        ans, refs = answer(q)
        print("\n💬 Answer:\n", ans, "\n")
        if refs:
            print("🔎 References:")
            for r in refs:
                print(f"  - [{r['id']}] (score={r['score']:.3f})")
        print("\n" + "-" * 60 + "\n")
    except Exception as e:
        print("⚠️ Error:", e)