# zen_ba_app.py
"""
Zen BA — AI Business Analyst application (corrected Gradio outputs)
- Embeddings: HuggingFaceEmbeddings (BAAI/bge-small-en-v1.5)
- Vector DB: Chroma (persistent)
- LLM: Perplexity (via langchain_perplexity.ChatPerplexity)
- UI: Gradio
"""

from dotenv import load_dotenv
import os
import sys
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
    print("Install packages: langchain, langchain-community, langchain-huggingface, chromadb, langchain-perplexity, langchain-text-splitters")
    raise

# Optional: whisper (local), jira, atlassian APIs
try:
    import whisper  # or faster-whisper
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

# Reportlab for PDF export
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

# JIRA / Confluence env (optional)
JIRA_BASE = os.getenv("JIRA_BASE")  # e.g. https://your-domain.atlassian.net
JIRA_USER = os.getenv("JIRA_USER")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")

CONFLUENCE_BASE = os.getenv("CONFLUENCE_BASE")
CONFLUENCE_USER = os.getenv("CONFLUENCE_USER")
CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")

# ==== Chroma persistent setup (multiple collections for clarity) ====
CHROMA_PATH = "./chroma_zenba"
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# collection naming:
COL_POLICY = "policy_docs"
COL_MEETINGS = "meetings"
COL_JIRA = "jira_issues"
COL_CONFLUENCE = "confluence_pages"
COL_REQUIREMENTS = "extracted_requirements"

# Embedding wrapper
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

# ==== RAG / LLM wrapper for answering ====
class ZenRAG:
    def __init__(self, primary_col):
        self.collection = primary_col
        self.prompt = ChatPromptTemplate.from_template(
            "You are a Business Analyst assistant. Answer strictly from the provided context and be concise.\n"
            "If not found, say: 'I couldn't find that in the provided sources.'\n\n"
            "Context:\n{context}\n\nQuestion: {question}\nAnswer (include sources):"
        )
        self.llm = ChatPerplexity(model="sonar", temperature=0)

    def retrieve(self, question: str, k: int = 6):
        results = self.collection.query(query_texts=[question], n_results=k)
        docs = []
        try:
            for doc, dist, meta in zip(results['documents'][0], results['distances'][0], results['metadatas'][0]):
                docs.append(Document(page_content=doc, metadata=meta))
        except Exception:
            pass
        return docs

    def format_docs(self, docs: List[Document]):
        formatted = []
        for doc in docs:
            src = doc.metadata.get("source", "unknown")
            formatted.append(f"[Source: **{src}**]\n{doc.page_content}")
        return "\n\n".join(formatted)

    def answer(self, question: str, k: int = 6):
        docs = self.retrieve(question, k=k)
        if not docs:
            return "I couldn't find that in the provided sources."
        context = self.format_docs(docs)
        chain = ({"context": lambda _: context, "question": RunnablePassthrough()} | self.prompt | self.llm | StrOutputParser())
        return chain.invoke(question)

zen_rag_primary = ZenRAG(collection_policy)

# ==== Utilities: document splitting & adding to collection ====
def split_and_add(docs: List[Document], collection, source_name: str, overwrite: bool = False):
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=80)
    split_docs = splitter.split_documents(docs)
    texts = [d.page_content for d in split_docs]
    ids = [f"{source_name}_{int(time.time())}_{i}" for i in range(len(split_docs))]
    metadatas = [dict(d.metadata or {}) for d in split_docs]
    for md in metadatas:
        md["source"] = source_name
        md.setdefault("ingested_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    if overwrite:
        try:
            collection.delete(where={"source": source_name})
        except Exception:
            pass

    if texts:
        collection.add(documents=texts, ids=ids, metadatas=metadatas)
    return len(split_docs)

# ==== Load files & text (similar to original) ====
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
    if WHISPER_AVAILABLE:
        try:
            model = whisper.load_model("base")
            result = model.transcribe(filepath)
            return result.get("text", "")
        except Exception as e:
            print("Whisper transcription failed:", e)
            return ""
    else:
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
        json_text = None
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if m:
            json_text = m.group(0)
        else:
            json_text = raw

        parsed = json.loads(json_text)
        requirements = parsed.get("requirements", [])
        decisions = parsed.get("decisions", [])
        requirements = [str(r).strip() for r in requirements]
        decisions = [dict(d) for d in decisions]
        return requirements, decisions
    except Exception as e:
        print("Extraction failed:", e)
        reqs = []
        decisions = []
        lines = transcript_text.splitlines()
        for ln in lines:
            if "require" in ln.lower() or "should " in ln.lower():
                reqs.append(ln.strip())
            if "decid" in ln.lower() or "action:" in ln.lower():
                decisions.append({"decision": ln.strip(), "owner": "", "due": ""})
        return reqs[:10], decisions[:10]

def process_meeting_file(uploaded_file, meeting_title="meeting"):
    if not uploaded_file:
        return "No file uploaded."
    try:
        transcript_text = transcribe_audio_file(uploaded_file.name)
        if not transcript_text:
            ext = os.path.splitext(uploaded_file.name)[-1].lower()
            if ext == ".txt":
                with open(uploaded_file.name, "r", encoding="utf-8") as fh:
                    transcript_text = fh.read()
        if not transcript_text:
            return "Transcription not available on server. Please use a supported transcription provider or upload a transcript (.txt)."

        doc = Document(page_content=transcript_text, metadata={"source": meeting_title, "type": "meeting_transcript"})
        count = split_and_add([doc], collection_meetings, source_name=meeting_title, overwrite=False)

        reqs, decisions = quick_extract_requirements_and_decisions(transcript_text, meeting_title)
        extracted_count = 0
        for i, r in enumerate(reqs):
            d = Document(page_content=r, metadata={"source": f"{meeting_title}_requirement_{i}", "origin": meeting_title})
            extracted_count += split_and_add([d], collection_requirements, source_name=f"{meeting_title}_requirement_{i}", overwrite=False)

        if decisions:
            dec_doc = Document(page_content=json.dumps(decisions), metadata={"source": f"{meeting_title}_decisions", "type": "decisions"})
            split_and_add([dec_doc], collection_meetings, source_name=f"{meeting_title}_decisions", overwrite=False)

        return f"✅ Transcribed & ingested meeting ({count} chunks). Extracted {len(reqs)} requirements and {len(decisions)} decisions."
    except Exception as e:
        traceback.print_exc()
        return f"❌ Failed to process meeting file: {e}"

# ==== JIRA integration helpers ====
def get_jira_client():
    if not JIRA_AVAILABLE:
        raise RuntimeError("jira package not installed. Install 'jira' to enable JIRA integration.")
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
            docs.append(Document(page_content=body, metadata={"source": f"jira_{issue.key}", "issue_key": issue.key}))
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

# ==== Confluence ingestion ====
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
            docs.append(Document(page_content=f"Title: {title}\n\n{content}", metadata={"source": f"confluence_{title}"}))
        count = split_and_add(docs, collection_confluence, source_name="confluence_bulk_ingest", overwrite=False)
        return f"✅ Ingested {len(docs)} Confluence pages ({count} chunks).", startup_status()
    except Exception as e:
        traceback.print_exc()
        return f"❌ Confluence ingestion error: {e}", startup_status()

# ==== Decision tracking (simple local JSON + chroma index) ====
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
    doc = Document(page_content=json.dumps(decision), metadata={"source": "decision", "title": decision.get("title","decision")})
    split_and_add([doc], collection_meetings, source_name=f"decision_{int(time.time())}", overwrite=False)
    return True

# ==== Multi-source question answering (queries multiple collections and merges) ====
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

    top_docs = all_docs[:20]
    context = "\n\n".join([f"[{d.metadata.get('source','unknown')}]\n{d.page_content}" for d in top_docs])

    synthesis_prompt = ChatPromptTemplate.from_template(
        "You are Zen BA, an AI Business Analyst. Use only the context to answer the user's question. "
        "Be concise and explicitly list sources used.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    llm = ChatPerplexity(model="sonar", temperature=0)
    chain = ({"context": lambda _: context, "question": RunnablePassthrough()} | synthesis_prompt | llm | StrOutputParser())
    return chain.invoke(question)

# ==== Export Q&A / decisions to PDF (if reportlab available) ====
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

# ==== Helper: startup status ====
def startup_status() -> str:
    counts = {
        "policy": collection_policy.count(),
        "meetings": collection_meetings.count(),
        "jira": collection_jira.count(),
        "confluence": collection_confluence.count(),
        "requirements": collection_requirements.count(),
    }
    return f"Collections counts: {counts}"

# ==== UI (Gradio) =====
with gr.Blocks(title="Zen BA — AI Business Analyst") as demo:
    gr.Markdown("# Zen BA — AI Business Analyst (prototype)\n"
                "Ingest files, meeting transcripts, JIRA and Confluence content. Extract requirements & decisions, query across sources.\n\n"
                "**Note on live meetings:** Zen BA doesn't automatically join calls. Configure provider webhooks/recordings to feed this app.")

    status_box = gr.Textbox(value=startup_status(), label="Startup Status", interactive=False)

    with gr.Tabs():
        with gr.TabItem("Ingest Files"):
            gr.Markdown("Upload multiple files (pdf/docx/txt) or paste text to ingest into Policy collection.")
            files = gr.File(file_types=[".pdf", ".docx", ".txt", ".md"], file_count="multiple", type="filepath")
            txt_in = gr.Textbox(label="Or paste text", lines=4)
            ingest_btn = gr.Button("Ingest into Policy")
            ingest_out = gr.Textbox(label="Result", interactive=False)

            def ingest_files_action(files, txt):
                docs = load_documents(files, txt, source_name="user_upload")
                if not docs:
                    return "⚠️ No valid documents or text provided.", startup_status()
                count = split_and_add(docs, collection_policy, source_name=f"user_upload_{int(time.time())}", overwrite=False)
                return f"Indexed {count} chunks from {len(docs)} documents.", startup_status()

            ingest_btn.click(ingest_files_action, inputs=[files, txt_in], outputs=[ingest_out, status_box], show_progress=True)

        with gr.TabItem("Meetings"):
            gr.Markdown("Upload meeting recording (.mp3/.wav/.mp4) or upload a transcript (.txt).")
            meeting_file = gr.File(file_types=[".mp3", ".wav", ".m4a", ".mp4", ".txt"], type="filepath")
            meeting_title = gr.Textbox(label="Meeting title", value="Client Sync")
            meeting_proc_btn = gr.Button("Process Meeting")
            meeting_res = gr.Textbox(interactive=False)

            meeting_proc_btn.click(process_meeting_file, inputs=[meeting_file, meeting_title], outputs=meeting_res, show_progress=True)

        with gr.TabItem("JIRA"):
            gr.Markdown("Ingest JIRA issues (requires JIRA credentials in .env). Push decisions to a JIRA issue as comments.")
            jira_jql = gr.Textbox(label="JQL (default: project = TEST ORDER BY updated DESC)", value="project = TEST ORDER BY updated DESC")
            jira_ingest_btn = gr.Button("Ingest JIRA Issues")
            jira_ingest_out = gr.Textbox(interactive=False)
            jira_issue_key = gr.Textbox(label="JIRA Issue Key (for pushing decision)")
            jira_comment = gr.Textbox(label="Decision/Comment to push")
            jira_push_btn = gr.Button("Push Decision to JIRA")
            jira_push_out = gr.Textbox(interactive=False)

            jira_ingest_btn.click(ingest_jira_issues, inputs=[jira_jql], outputs=[jira_ingest_out, status_box], show_progress=True)
            jira_push_btn.click(push_decision_to_jira, inputs=[jira_issue_key, jira_comment], outputs=jira_push_out)

        with gr.TabItem("Confluence"):
            gr.Markdown("Ingest Confluence pages by space key or CQL. Requires Confluence credentials in .env.")
            conf_space = gr.Textbox(label="Space key (optional)")
            conf_cql = gr.Textbox(label="CQL (optional)")
            conf_btn = gr.Button("Ingest Confluence")
            conf_out = gr.Textbox(interactive=False)

            conf_btn.click(ingest_confluence_pages, inputs=[conf_space, conf_cql], outputs=[conf_out, status_box], show_progress=True)

        with gr.TabItem("Requirements & Decisions"):
            gr.Markdown("Quick extract requirements/decisions from pasted text (or transcript). Save decisions to tracking.")
            req_text = gr.Textbox(label="Transcript or meeting notes", lines=8)
            extract_btn = gr.Button("Extract Requirements & Decisions")
            extract_out = gr.Textbox(interactive=False)
            save_decision_btn = gr.Button("Save Decision")
            decision_title = gr.Textbox(label="Decision Title")
            decision_owner = gr.Textbox(label="Owner")
            decision_due = gr.Textbox(label="Due date")
            save_decision_out = gr.Textbox(interactive=False)

            def extract_action(text):
                reqs, decisions = quick_extract_requirements_and_decisions(text, meeting_title="manual_extract")
                return json.dumps({"requirements": reqs, "decisions": decisions}, indent=2)

            def save_decision_action(title, owner, due):
                if not title:
                    return "Decision title required."
                dec = {"title": title, "owner": owner, "due": due, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                save_decision(dec)
                return "Saved decision."

            extract_btn.click(extract_action, inputs=[req_text], outputs=extract_out)
            save_decision_btn.click(save_decision_action, inputs=[decision_title, decision_owner, decision_due], outputs=save_decision_out)

        with gr.TabItem("Ask / Q&A"):
            gr.Markdown("Ask across all ingested sources (policy, meetings, JIRA, Confluence, extracted requirements).")
            user_q = gr.Textbox(label="Question", lines=2)
            ask_btn = gr.Button("Ask Zen BA")
            ask_out = gr.Markdown()

            ask_btn.click(multi_source_query, inputs=[user_q], outputs=[ask_out])

        with gr.TabItem("Export / Admin"):
            gr.Markdown("Export decisions to PDF, reset collections, view startup status.")
            export_btn = gr.Button("Export decisions to PDF")
            export_file = gr.File(label="Download Export")
            reset_btn = gr.Button("Reset All Collections (DANGEROUS)")
            reset_out = gr.Textbox(interactive=False)

            def export_action():
                path = export_report_pdf()
                if path:
                    return path
                return None

            export_btn.click(export_action, inputs=None, outputs=export_file)

            def reset_all():
                try:
                    for c in [collection_policy, collection_meetings, collection_jira, collection_confluence, collection_requirements]:
                        c.delete(where={})
                    if os.path.exists(DECISIONS_FILE):
                        os.remove(DECISIONS_FILE)
                    return "✅ Reset all collections and decisions file.", startup_status()
                except Exception as e:
                    return f"❌ Reset failed: {e}", startup_status()

            reset_btn.click(reset_all, inputs=None, outputs=[reset_out, status_box])

    refresh_btn = gr.Button("Refresh status")
    refresh_btn.click(lambda: startup_status(), inputs=None, outputs=status_box)

if __name__ == "__main__":
    demo.launch(share=False, debug=True)
