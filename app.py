# zen_ba_app.py
"""
Zen BA — AI Business Analyst application (full functionality + Dummy Confluence)
- Embeddings: HuggingFaceEmbeddings (BAAI/bge-small-en-v1.5)
- Vector DB: Chroma (persistent)
- LLM: Perplexity (via langchain_perplexity.ChatPerplexity)
- UI: Gradio
- Transcription: whisper fallback + placeholder for cloud transcription
- JIRA / Confluence integration: optional (requires credentials)
- Features: multi-source ingestion (files, confluence, jira, meeting audio/transcripts),
  requirement extraction, decision tracking, export to PDF, dummy confluence ingestion for local testing.
"""
from dotenv import load_dotenv
import os
import time
import json
import traceback
from typing import List, Dict, Optional, Tuple

import gradio as gr

# ==== Optional libs: try-imports with graceful fallback ====
try:
    from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
    from langchain_community.docstore.document import Document
    from langchain_huggingface import HuggingFaceEmbeddings
    import chromadb
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain_perplexity import ChatPerplexity
except Exception as e:
    print("❗ Required langchain / embedding libs not installed or import failed.")
    raise

# Optional features
try:
    import whisper  # optional local transcription
    WHISPER_AVAILABLE = True
except Exception:
    WHISPER_AVAILABLE = False

try:
    from jira import JIRA
    JIRA_AVAILABLE = True
except Exception:
    JIRA_AVAILABLE = False

try:
    from atlassian import Confluence
    CONFLUENCE_AVAILABLE = True
except Exception:
    CONFLUENCE_AVAILABLE = False

# reportlab for PDF export
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# ==== Load env ====
load_dotenv()
PPLX_API_KEY = os.getenv("PPLX_API_KEY")
if not PPLX_API_KEY:
    raise RuntimeError("PPLX_API_KEY not set in .env")
os.environ["OPENAI_API_KEY"] = PPLX_API_KEY
os.environ["OPENAI_API_BASE"] = "https://api.perplexity.ai"

# optional JIRA / Confluence environment vars
JIRA_BASE = os.getenv("JIRA_BASE")
JIRA_USER = os.getenv("JIRA_USER")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")

CONFLUENCE_BASE = os.getenv("CONFLUENCE_BASE")
CONFLUENCE_USER = os.getenv("CONFLUENCE_USER")
CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")

# ==== Chroma persistent setup (multiple collections) ====
CHROMA_PATH = "./chroma_zenba"
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

COL_POLICY = "policy_docs"
COL_MEETINGS = "meetings"
COL_JIRA = "jira_issues"
COL_CONFLUENCE = "confluence_pages"
COL_REQUIREMENTS = "extracted_requirements"

class EmbeddingFunctionWrapper:
    def __init__(self, embedder):
        self.embedder = embedder
    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.embedder.embed_documents(input)
    def name(self): return "wrapped_hf_embeddings"

emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
embedding_function = EmbeddingFunctionWrapper(emb)

def get_or_create_collection(name: str):
    return chroma_client.get_or_create_collection(name=name, embedding_function=embedding_function)

collection_policy = get_or_create_collection(COL_POLICY)
collection_meetings = get_or_create_collection(COL_MEETINGS)
collection_jira = get_or_create_collection(COL_JIRA)
collection_confluence = get_or_create_collection(COL_CONFLUENCE)
collection_requirements = get_or_create_collection(COL_REQUIREMENTS)

# ==== RAG / LLM helper ====
class ZenRAG:
    def __init__(self, primary_col):
        self.collection = primary_col
        self.prompt = ChatPromptTemplate.from_template(
            "You are an assistant Business Analyst. Use ONLY the provided context to answer. "
            "If the answer is not present, say: 'I couldn't find that in the provided sources.'\n\n"
            "Context:\n{context}\n\nQuestion: {question}\nAnswer (include source):"
        )
        self.llm = ChatPerplexity(model="sonar", temperature=0)

    def retrieve(self, question: str, k: int = 6):
        results = self.collection.query(query_texts=[question], n_results=k)
        docs = []
        try:
            for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                docs.append(Document(page_content=doc, metadata=meta))
        except Exception:
            pass
        return docs

    def format_docs(self, docs: List[Document]):
        formatted = []
        for d in docs:
            src = d.metadata.get("source", "unknown")
            formatted.append(f"[Source: {src}]\n{d.page_content}")
        return "\n\n".join(formatted)

    def answer(self, question: str, k: int = 6):
        docs = self.retrieve(question, k=k)
        if not docs:
            return "I couldn't find that in the provided sources."
        context = self.format_docs(docs)
        chain = ({"context": lambda _: context, "question": RunnablePassthrough()} | self.prompt | self.llm | StrOutputParser())
        return chain.invoke(question)

zen_rag_primary = ZenRAG(collection_policy)

# ==== Utilities: splitting & adding to collections ====
def split_and_add(docs: List[Document], collection, source_name: str, overwrite: bool = False) -> int:
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=80)
    split_docs = splitter.split_documents(docs)
    texts = [d.page_content for d in split_docs]
    ids = [f"{source_name}_{int(time.time())}_{i}" for i in range(len(split_docs))]
    metadatas = []
    for d in split_docs:
        md = dict(d.metadata or {})
        md["source"] = source_name
        md.setdefault("ingested_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        metadatas.append(md)

    if overwrite:
        try:
            collection.delete(where={"source": source_name})
        except Exception:
            pass

    if texts:
        collection.add(documents=texts, ids=ids, metadatas=metadatas)
    return len(split_docs)

# ==== Load documents from files or raw text ====
def load_documents(files, raw_text, source_name: str="manual_input") -> List[Document]:
    docs = []
    if files:
        for file in files:
            ext = os.path.splitext(file.name)[-1].lower()
            try:
                if ext == ".pdf":
                    loader = PyPDFLoader(file.name)
                    loaded = loader.load()
                elif ext == ".docx":
                    loader = Docx2txtLoader(file.name)
                    loaded = loader.load()
                elif ext == ".txt":
                    loader = TextLoader(file.name, encoding="utf-8")
                    loaded = loader.load()
                else:
                    # fallback read
                    try:
                        with open(file.name, "r", encoding="utf-8", errors="ignore") as fh:
                            content = fh.read()
                        loaded = [Document(page_content=content, metadata={"source": file.name})]
                    except Exception:
                        continue
                for d in loaded:
                    d.metadata["source"] = os.path.basename(file.name)
                docs.extend(loaded)
            except Exception as e:
                print(f"Failed to load {file.name}: {e}")
    if raw_text and raw_text.strip():
        docs.append(Document(page_content=raw_text.strip(), metadata={"source": source_name}))
    return docs

# ==== Meeting transcription & processing ====
def transcribe_audio_file(filepath: str) -> str:
    """
    Try local whisper; otherwise return empty string (placeholder: integrate cloud transcription).
    """
    if WHISPER_AVAILABLE:
        try:
            model = whisper.load_model("base")
            result = model.transcribe(filepath)
            return result.get("text", "")
        except Exception as e:
            print("Whisper transcription failed:", e)
            return ""
    else:
        # If you have a cloud provider, integrate here. For now return empty to indicate not available.
        return ""

def quick_extract_requirements_and_decisions(transcript_text: str, meeting_title: str) -> Tuple[List[str], List[Dict]]:
    if not transcript_text.strip():
        return [], []
    prompt = (
        "You are a Business Analyst tool. Given the meeting transcript below, extract:\n"
        "1) A list of discrete functional or non-functional requirements (short bullets).\n"
        "2) A list of decisions made in the meeting. For each decision output: decision summary, owner (who), and any due date (if mentioned) as ISO date or text.\n\n"
        "Transcript:\n" + transcript_text + "\n\nRespond with JSON: {\"requirements\": [...], \"decisions\": [{\"decision\":\"...\",\"owner\":\"...\",\"due\":\"...\"}, ...]}"
    )
    llm = ChatPerplexity(model="sonar", temperature=0)
    try:
        raw = llm.invoke(prompt)
        import re
        m = re.search(r"\{.*\}", raw, flags=re.S)
        json_text = m.group(0) if m else raw
        parsed = json.loads(json_text)
        requirements = parsed.get("requirements", [])
        decisions = parsed.get("decisions", [])
        requirements = [str(r).strip() for r in requirements]
        decisions = [dict(d) for d in decisions]
        return requirements, decisions
    except Exception as e:
        print("Extraction via LLM failed:", e)
        # naive fallback
        reqs = []
        decisions = []
        for ln in transcript_text.splitlines():
            lower = ln.lower()
            if any(k in lower for k in ["shall", "should", "as a ", "acceptance criteria", "requirement"]):
                reqs.append(ln.strip())
            if any(k in lower for k in ["decid", "action:", "we will", "agreed to"]):
                decisions.append({"decision": ln.strip(), "owner": "", "due": ""})
        return reqs[:50], decisions[:50]

def process_meeting_file(uploaded_file, meeting_title="meeting"):
    """
    Transcribe (if possible), store transcript in meetings collection, extract requirements & decisions and index them.
    Returns a two-value tuple for Gradio (status message, startup_status).
    """
    if not uploaded_file:
        return "⚠️ No file uploaded.", startup_status()
    try:
        # attempt transcription
        transcript_text = transcribe_audio_file(uploaded_file.name)
        # if user uploaded a .txt transcript, read it
        if not transcript_text:
            ext = os.path.splitext(uploaded_file.name)[-1].lower()
            if ext == ".txt":
                with open(uploaded_file.name, "r", encoding="utf-8") as fh:
                    transcript_text = fh.read()
        if not transcript_text:
            return "⚠️ Transcription not available locally. Upload a .txt transcript or configure a transcription provider.", startup_status()

        # store transcript into meetings collection
        doc = Document(page_content=transcript_text, metadata={"source": meeting_title, "type": "meeting_transcript"})
        count = split_and_add([doc], collection_meetings, source_name=meeting_title, overwrite=False)

        # extract requirements and decisions
        reqs, decisions = quick_extract_requirements_and_decisions(transcript_text, meeting_title)
        extracted_reqs = 0
        for i, r in enumerate(reqs):
            d = Document(page_content=r, metadata={"source": f"{meeting_title}_requirement_{i}", "origin": meeting_title, "type":"requirement"})
            extracted_reqs += split_and_add([d], collection_requirements, source_name=f"{meeting_title}_requirement_{i}", overwrite=False)
        if decisions:
            dec_doc = Document(page_content=json.dumps(decisions), metadata={"source": f"{meeting_title}_decisions", "type": "decisions"})
            split_and_add([dec_doc], collection_meetings, source_name=f"{meeting_title}_decisions", overwrite=False)

        return f"✅ Transcribed & ingested meeting ({count} chunks). Extracted {len(reqs)} requirements and {len(decisions)} decisions.", startup_status()
    except Exception as e:
        traceback.print_exc()
        return f"❌ Failed to process meeting file: {e}", startup_status()

# ==== JIRA integration helpers ====
def get_jira_client():
    if not JIRA_AVAILABLE:
        raise RuntimeError("jira package not installed.")
    if not (JIRA_BASE and JIRA_USER and JIRA_API_TOKEN):
        raise RuntimeError("JIRA credentials not found in environment.")
    options = {"server": JIRA_BASE}
    jira = JIRA(options, basic_auth=(JIRA_USER, JIRA_API_TOKEN))
    return jira

def ingest_jira_issues(jira_jql: str = "project = TEST ORDER BY updated DESC", max_issues: int = 50):
    if not JIRA_AVAILABLE:
        return "JIRA integration not available (jira package missing).", startup_status()
    try:
        jira = get_jira_client()
        issues = jira.search_issues(jira_jql, maxResults=max_issues)
        docs = []
        for issue in issues:
            body = f"Issue: {issue.key}\nSummary: {issue.fields.summary}\nDescription:\n{issue.fields.description or ''}\nStatus: {getattr(issue.fields, 'status', '')}"
            docs.append(Document(page_content=body, metadata={"source": f"jira_{issue.key}", "issue_key": issue.key, "type":"jira"}))
        count = split_and_add(docs, collection_jira, source_name="jira_bulk_ingest", overwrite=False)
        return f"✅ Ingested {len(docs)} JIRA issues ({count} chunks).", startup_status()
    except Exception as e:
        traceback.print_exc()
        return f"❌ JIRA ingestion failed: {e}", startup_status()

def push_decision_to_jira(issue_key: str, comment: str):
    if not JIRA_AVAILABLE:
        return "JIRA integration not available."
    try:
        jira = get_jira_client()
        jira.add_comment(issue_key, comment)
        return f"✅ Pushed decision as comment to {issue_key}"
    except Exception as e:
        traceback.print_exc()
        return f"❌ Failed to push decision: {e}"

# ==== Confluence ingestion (real) ====
def ingest_confluence_pages(space_key: Optional[str] = None, cql: Optional[str] = None, max_pages: int = 50):
    if not CONFLUENCE_AVAILABLE:
        return "Confluence integration not available (atlassian package missing).", startup_status()
    if not (CONFLUENCE_BASE and CONFLUENCE_USER and CONFLUENCE_API_TOKEN):
        return "Confluence credentials missing in environment.", startup_status()
    try:
        confluence = Confluence(url=CONFLUENCE_BASE, username=CONFLUENCE_USER, password=CONFLUENCE_API_TOKEN)
        pages = []
        if space_key:
            res = confluence.get_all_pages_from_space(space=space_key, start=0, limit=max_pages)
            pages = res
        elif cql:
            res = confluence.cql(cql + f" ORDER BY lastmodified DESC", start=0, limit=max_pages)
            pages = res.get("results", [])
        else:
            return "Provide space_key or cql.", startup_status()
        docs = []
        for p in pages:
            title = p.get("title") or p.get("pageTitle") or "confluence_page"
            body = confluence.get_page_by_id(p.get("id"), expand='body.storage') if p.get("id") else None
            content = ""
            if body and "body" in body and "storage" in body["body"] and "value" in body["body"]["storage"]:
                content = body["body"]["storage"]["value"]
            docs.append(Document(page_content=f"Title: {title}\n\n{content}", metadata={"source": f"confluence_{title}", "type":"confluence"}))
        count = split_and_add(docs, collection_confluence, source_name="confluence_bulk_ingest", overwrite=False)
        return f"✅ Ingested {len(docs)} Confluence pages ({count} chunks).", startup_status()
    except Exception as e:
        traceback.print_exc()
        return f"❌ Confluence ingestion error: {e}", startup_status()

# ==== Dummy Confluence ingestion (local testing) ====
def ingest_dummy_confluence_page(title: str, body_text: str):
    if not title.strip():
        title = f"DummyPage_{int(time.time())}"
    if not body_text.strip():
        return "⚠️ No content provided.", startup_status()
    doc = Document(page_content=f"Title: {title}\n\n{body_text.strip()}", metadata={"source": f"dummy_confluence_{title}", "type":"dummy_confluence"})
    count = split_and_add([doc], collection_confluence, source_name=f"dummy_confluence_{title}", overwrite=False)
    return f"✅ Ingested dummy Confluence page '{title}' ({count} chunks).", startup_status()

# ==== Decision tracking (local JSON + index) ====
DECISIONS_FILE = "zenba_decisions.json"
def load_decisions():
    if os.path.exists(DECISIONS_FILE):
        with open(DECISIONS_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return []

def save_decision(decision: Dict):
    decisions = load_decisions()
    decisions.append(decision)
    with open(DECISIONS_FILE, "w", encoding="utf-8") as fh:
        json.dump(decisions, fh, indent=2)
    # index decision for search
    doc = Document(page_content=json.dumps(decision), metadata={"source": "decision", "title": decision.get("title","decision"), "type":"decision"})
    split_and_add([doc], collection_meetings, source_name=f"decision_{int(time.time())}", overwrite=False)
    return True

# ==== Multi-source question answering ====
def multi_source_query(question: str, k_per_col: int = 4):
    all_docs = []
    for col in [collection_policy, collection_meetings, collection_jira, collection_confluence, collection_requirements]:
        try:
            res = col.query(query_texts=[question], n_results=k_per_col)
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            for d, m in zip(docs, metas):
                all_docs.append(Document(page_content=d, metadata=m))
        except Exception:
            pass
    if not all_docs:
        return "I couldn't find relevant information in the ingested sources."
    # rudimentary top selection (could be improved with distances)
    top_docs = all_docs[:20]
    context = "\n\n".join([f"[{d.metadata.get('source','unknown')}]\n{d.page_content}" for d in top_docs])
    synthesis_prompt = ChatPromptTemplate.from_template(
        "You are Zen BA, an AI Business Analyst. Use only the context to answer the user's question. "
        "Be concise and explicitly list sources used.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    llm = ChatPerplexity(model="sonar", temperature=0)
    chain = ({"context": lambda _: context, "question": RunnablePassthrough()} | synthesis_prompt | llm | StrOutputParser())
    return chain.invoke(question)

# ==== Requirements extraction and store ====
def extract_requirements_and_decisions_from_sources():
    """
    Use LLM to scan policy/meetings/confluence and produce consolidated requirements & decisions JSON,
    then index it into collection_requirements.
    """
    collected_texts = []
    for col in [collection_policy, collection_meetings, collection_confluence]:
        try:
            res = col.query(query_texts=["requirements", "decisions"], n_results=20)
            for d in res.get("documents",[[]])[0]:
                collected_texts.append(d)
        except Exception:
            pass
    if not collected_texts:
        return "⚠️ No documents to extract from.", startup_status()
    context = "\n\n".join(collected_texts)
    prompt = (
        "You are a Business Analyst assistant. From the context below, extract a JSON object with keys:\n"
        "\"requirements\": [\"...\"],\n\"decisions\": [{\"decision\":\"...\",\"owner\":\"...\",\"due\":\"...\"}, ...]\n\nContext:\n" + context
    )
    llm = ChatPerplexity(model="sonar", temperature=0)
    chain = ({"context": lambda _: context} | ChatPromptTemplate.from_template(prompt) | llm | StrOutputParser())
    try:
        raw = chain.invoke("")
        import re
        m = re.search(r"\{.*\}", raw, flags=re.S)
        json_text = m.group(0) if m else raw
        parsed = json.loads(json_text)
    except Exception as e:
        print("Extraction failed:", e)
        # fallback: store raw context as one doc
        parsed = {"requirements": [], "decisions": []}
    # index parsed output
    doc = Document(page_content=json.dumps(parsed), metadata={"source":"requirements_extraction", "type":"requirements"})
    split_and_add([doc], collection_requirements, source_name="requirements_extraction", overwrite=True)
    return "✅ Requirements & decisions extracted and indexed.", startup_status()

# ==== Export to PDF ====
def export_report_pdf():
    if not REPORTLAB_AVAILABLE:
        return None
    decisions = load_decisions()
    file_path = "zenba_report.pdf"
    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("Zen BA Report", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Decisions", styles["Heading2"]))
    for d in decisions:
        story.append(Paragraph(json.dumps(d, ensure_ascii=False), styles["Normal"]))
        story.append(Spacer(1, 8))
    doc.build(story)
    return file_path

# ==== Reset knowledge base ====
def reset_all_collections():
    try:
        for c in [collection_policy, collection_meetings, collection_jira, collection_confluence, collection_requirements]:
            c.delete(where={})
        if os.path.exists(DECISIONS_FILE):
            os.remove(DECISIONS_FILE)
        return "✅ Reset all collections and decisions file.", startup_status()
    except Exception as e:
        return f"❌ Reset failed: {e}", startup_status()

# ==== Startup status helper ====
def startup_status():
    counts = {
        "policy": collection_policy.count(),
        "meetings": collection_meetings.count(),
        "jira": collection_jira.count(),
        "confluence": collection_confluence.count(),
        "requirements": collection_requirements.count(),
    }
    return f"Collections counts: {counts}"

# ==== QA history (local) ====
qa_history: List[Tuple[str,str]] = []

def ask_question(q: str):
    # this handler is used for single-output Markdown in UI
    answer = multi_source_query(q)
    qa_history.append((q, answer))
    return answer

# ==== Gradio UI ====
with gr.Blocks(title="Zen BA — AI Business Analyst") as demo:
    gr.Markdown("# Zen BA — AI Business Analyst\nIngest files, meetings, JIRA, Confluence; extract requirements & decisions; query across all sources.\n\n**Note:** Live meeting participation requires external webhooks/SDKs (we provide safe placeholders).")

    status_box = gr.Textbox(value=startup_status(), label="Startup Status", interactive=False)

    with gr.Tabs():
        # Ingest Files
        with gr.TabItem("Ingest Files"):
            gr.Markdown("Upload documents (PDF/DOCX/TXT) or paste text to ingest into the policy collection.")
            files = gr.File(file_types=[".pdf", ".docx", ".txt"], file_count="multiple", type="filepath")
            text_box = gr.Textbox(label="Or paste text", lines=6)
            process_btn = gr.Button("Process / Update Policies")
            process_status = gr.Textbox(label="Status", interactive=False)
            def process_files(files, text):
                docs = load_documents(files, text)
                if not docs:
                    return "⚠️ No valid documents or text provided.", startup_status()
                count = split_and_add(docs, collection_policy, source_name=f"user_upload_{int(time.time())}", overwrite=False)
                return f"✅ Indexed {count} chunks from {len(docs)} document(s).", startup_status()
            process_btn.click(process_files, inputs=[files, text_box], outputs=[process_status, status_box])

        # Meetings
        with gr.TabItem("Meetings"):
            gr.Markdown("Upload meeting recording (mp3/wav/mp4) or a .txt transcript. Local Whisper transcription attempted if installed.")
            meeting_file = gr.File(file_types=[".mp3", ".wav", ".m4a", ".mp4", ".txt"], type="filepath")
            meeting_title = gr.Textbox(label="Meeting title", value="Client Sync")
            meeting_proc_btn = gr.Button("Process Meeting")
            meeting_out = gr.Textbox(interactive=False)
            meeting_proc_btn.click(process_meeting_file, inputs=[meeting_file, meeting_title], outputs=[meeting_out, status_box])

        # JIRA
        with gr.TabItem("JIRA"):
            gr.Markdown("Ingest JIRA issues via JQL or push decisions back as comments (requires JIRA credentials in .env).")
            jira_jql = gr.Textbox(label="JQL (default: project = TEST ORDER BY updated DESC)", value="project = TEST ORDER BY updated DESC")
            jira_ingest_btn = gr.Button("Ingest JIRA Issues")
            jira_ingest_out = gr.Textbox(interactive=False)
            jira_issue_key = gr.Textbox(label="JIRA Issue Key (for pushing decision)")
            jira_comment = gr.Textbox(label="Decision/Comment to push")
            jira_push_btn = gr.Button("Push Decision to JIRA")
            jira_push_out = gr.Textbox(interactive=False)
            jira_ingest_btn.click(ingest_jira_issues, inputs=[jira_jql], outputs=[jira_ingest_out, status_box])
            jira_push_btn.click(push_decision_to_jira, inputs=[jira_issue_key, jira_comment], outputs=jira_push_out)

        # Confluence (real)
        with gr.TabItem("Confluence"):
            gr.Markdown("Ingest Confluence pages by space key or CQL (requires Confluence credentials).")
            conf_space = gr.Textbox(label="Space key (optional)")
            conf_cql = gr.Textbox(label="CQL (optional)")
            conf_btn = gr.Button("Ingest Confluence")
            conf_out = gr.Textbox(interactive=False)
            conf_btn.click(ingest_confluence_pages, inputs=[conf_space, conf_cql], outputs=[conf_out, status_box])

        # Dummy Confluence (local testing)
        with gr.TabItem("Ingest Dummy Confluence Page"):
            gr.Markdown("Paste any text to simulate a Confluence page (no real Confluence required).")
            dummy_title = gr.Textbox(label="Page Title", value="Sample Dummy Page")
            dummy_body = gr.Textbox(label="Page Body", lines=8, placeholder="Paste requirements, decisions, notes...")
            dummy_ingest_btn = gr.Button("Ingest Dummy Confluence Page")
            dummy_status = gr.Textbox(interactive=False)
            dummy_ingest_btn.click(ingest_dummy_confluence_page, inputs=[dummy_title, dummy_body], outputs=[dummy_status, status_box])

        # Requirements & Decisions
        with gr.TabItem("Requirements & Decisions"):
            gr.Markdown("Extract requirements & decisions from ingested sources and track decisions.")
            req_text = gr.Textbox(label="Or paste meeting notes/transcript for extraction", lines=6)
            extract_btn = gr.Button("Extract from Pasted Text")
            extract_out = gr.Textbox(interactive=False)
            def extract_from_text(text):
                reqs, decisions = quick_extract_requirements_and_decisions(text or "", meeting_title="manual_extract")
                return json.dumps({"requirements": reqs, "decisions": decisions}, indent=2)
            extract_btn.click(extract_from_text, inputs=[req_text], outputs=[extract_out])

            save_decision_title = gr.Textbox(label="Decision Title")
            save_decision_owner = gr.Textbox(label="Owner")
            save_decision_due = gr.Textbox(label="Due date")
            save_decision_btn = gr.Button("Save Decision")
            save_decision_out = gr.Textbox(interactive=False)
            def save_decision_action(title, owner, due):
                if not title:
                    return "Decision title required."
                dec = {"title": title, "owner": owner, "due": due, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                save_decision(dec)
                return "✅ Saved decision."
            save_decision_btn.click(save_decision_action, inputs=[save_decision_title, save_decision_owner, save_decision_due], outputs=save_decision_out)

            extract_all_btn = gr.Button("Extract from All Ingested Sources (LLM)")
            extract_all_out = gr.Textbox(interactive=False)
            extract_all_btn.click(extract_requirements_and_decisions_from_sources, inputs=None, outputs=[extract_all_out, status_box])

        # Ask / Q&A
        with gr.TabItem("Ask / Q&A"):
            gr.Markdown("Ask across all ingested sources (policy, meetings, JIRA, Confluence, extracted requirements).")
            user_q = gr.Textbox(label="Question", lines=2)
            ask_btn = gr.Button("Ask Zen BA")
            ask_out = gr.Markdown()
            ask_btn.click(ask_question, inputs=[user_q], outputs=[ask_out])

        # Export / Admin
        with gr.TabItem("Export / Admin"):
            gr.Markdown("Export decisions to PDF, reset collections, and view status.")
            export_btn = gr.Button("📥 Export Q&A & Decisions (PDF)")
            export_file = gr.File(label="Download Export")
            def export_action():
                path = export_report_pdf()
                return path
            export_btn.click(export_action, inputs=None, outputs=export_file)

            reset_btn = gr.Button("Reset All Collections (DANGEROUS)")
            reset_out = gr.Textbox(interactive=False)
            reset_btn.click(reset_all_collections, inputs=None, outputs=[reset_out, status_box])

    refresh_btn = gr.Button("Refresh status")
    refresh_btn.click(lambda: startup_status(), inputs=None, outputs=status_box)

if __name__ == "__main__":
    demo.launch(share=False, debug=True)
