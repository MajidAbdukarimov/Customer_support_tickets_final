#!/usr/bin/env python3
"""
AI-Powered Customer Support Assistant
Complete solution with pre-loaded documents, OpenAI integration, GitHub Issues, and advanced RAG capabilities.
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import uuid
import re
import glob
from pathlib import Path

import streamlit as st
import pandas as pd

# ──────────────────────────────── ENV & LOGGING ────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load .env from the folder where THIS file resides (works even if CWD differs)
try:
    from dotenv import load_dotenv
    ENV_PATH = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=ENV_PATH, override=True)
    logger.info(f"Loaded .env from {ENV_PATH}")
except Exception as _e:
    logger.warning(f".env load warning: {_e}")



# ── HF Spaces persistence
BASE_DIR = Path("/data") if Path("/data").exists() else Path(__file__).parent
DOCUMENTS_DIR = str(BASE_DIR / "documents")   # было: "documents"

# Переопределим пути, где это нужно:
TICKETS_FILE_PATH = str(BASE_DIR / "support_tickets.json")
CHROMA_PATH = str(BASE_DIR / "chroma_db")

# ──────────────────────────────── OPTIONAL LIBS ────────────────────────────────
# OpenAI integration (support both legacy 0.x and new 1.x SDKs)
OPENAI_AVAILABLE = False
OPENAI_V1 = False
try:
    import openai  # legacy 0.x import
    OPENAI_AVAILABLE = True
    try:
        # New 1.x client
        from openai import OpenAI as OpenAIClient  # type: ignore
        OPENAI_V1 = True
    except Exception:
        OPENAI_V1 = False
except ImportError:
    OPENAI_AVAILABLE = False
    OPENAI_V1 = False

# Vector database and embeddings
try:
    import chromadb
    from chromadb.config import Settings  # noqa: F401 (kept for backward compat)
    CHROMADB_AVAILABLE = True
except Exception as e:  # не только ImportError — Chroma кидает RuntimeError при старом sqlite
    CHROMADB_AVAILABLE = False
    logger.warning(f"Chroma disabled: {e}")

# Sentence Transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# PDF processing
PYMUPDF_AVAILABLE = False
PYPDF2_AVAILABLE = False
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except Exception:
    try:
        import PyPDF2  # type: ignore
        PYPDF2_AVAILABLE = True
    except Exception:
        PYPDF2_AVAILABLE = False

# GitHub integration
try:
    import requests
    REQUESTS_AVAILABLE = True
except Exception:
    REQUESTS_AVAILABLE = False

# ──────────────────────────────── COMPANY CONFIG ───────────────────────────────
COMPANY_CONFIG = {
    "name": "TechSolutions Inc.",
    "email": "support@techsolutions.com",
    "phone": "+1-800-TECH-HELP",
    "website": "www.techsolutions.com",
    "description": "Leading provider of innovative technology solutions",
    "business_hours": "Monday-Friday 9AM-6PM EST"
}

DOCUMENTS_DIR = "documents"

# ──────────────────────────────── DATA CLASSES ────────────────────────────────
@dataclass
class SupportTicket:
    id: str
    user_name: str
    user_email: str
    summary: str
    description: str
    created_at: str
    status: str = "Open"
    priority: str = "Medium"
    category: str = "General"
    github_url: str = ""
    github_number: int = 0

@dataclass
class ChatMessage:
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str
    sources: List[Dict[str, Any]] = None
    confidence: float = 0.0
    model_used: str = ""
    tokens_used: int = 0

# ──────────────────────────────── DOCUMENT PROCESSOR ───────────────────────────
class DocumentProcessor:
    def __init__(self):
        self.chunk_size = 1000
        self.chunk_overlap = 200

    def load_preloaded_documents(self, documents_dir: str = DOCUMENTS_DIR) -> List[Dict[str, Any]]:
        all_chunks = []
        os.makedirs(documents_dir, exist_ok=True)
        pdf_files = glob.glob(os.path.join(documents_dir, "*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {documents_dir}")
            return []
        logger.info(f"Found {len(pdf_files)} PDF files to process")

        for pdf_path in pdf_files:
            try:
                chunks = self.extract_text_from_pdf(pdf_path)
                if chunks:
                    all_chunks.extend(chunks)
                    logger.info(f"Processed {os.path.basename(pdf_path)}: {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {e}")
        logger.info(f"Total chunks processed: {len(all_chunks)}")
        return all_chunks

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        chunks = []
        filename = os.path.basename(pdf_path)
        try:
            if PYMUPDF_AVAILABLE:
                doc = fitz.open(pdf_path)  # type: ignore
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text()
                    if text and text.strip():
                        cleaned = self._clean_text(text)
                        for i, ck in enumerate(self._chunk_text(cleaned)):
                            if len(ck.strip()) > 50:
                                chunks.append({
                                    "text": ck,
                                    "source": filename,
                                    "page": page_num + 1,
                                    "total_pages": len(doc),
                                    "chunk_id": f"{filename}_p{page_num + 1}_c{i + 1}",
                                    "word_count": len(ck.split()),
                                    "char_count": len(ck)
                                })
                doc.close()
            elif PYPDF2_AVAILABLE:
                with open(pdf_path, "rb") as f:  # type: ignore
                    pdf_reader = PyPDF2.PdfReader(f)  # type: ignore
                    for page_num, page in enumerate(pdf_reader.pages):
                        text = page.extract_text()
                        if text and text.strip():
                            cleaned = self._clean_text(text)
                            for i, ck in enumerate(self._chunk_text(cleaned)):
                                if len(ck.strip()) > 50:
                                    chunks.append({
                                        "text": ck,
                                        "source": filename,
                                        "page": page_num + 1,
                                        "total_pages": len(pdf_reader.pages),
                                        "chunk_id": f"{filename}_p{page_num + 1}_c{i + 1}",
                                        "word_count": len(ck.split()),
                                        "char_count": len(ck)
                                    })
            logger.info(f"Processed {len(chunks)} chunks from {filename}")
            return chunks
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            return []

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^Page \d+.*$', '', text, flags=re.MULTILINE)
        return text

    def _chunk_text(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks, current = [], ""
        for s in sentences:
            if len(current) + len(s) + 1 <= self.chunk_size:
                current = f"{current} {s}".strip() if current else s
            else:
                if current:
                    chunks.append(current.strip())
                if len(s) > self.chunk_size:
                    words = s.split()
                    step = max(50, self.chunk_size // 10)
                    for i in range(0, len(words), step):
                        chunks.append(" ".join(words[i:i + step]))
                    current = ""
                else:
                    current = s
        if current:
            chunks.append(current.strip())
        return [c for c in chunks if len(c.strip()) > 50]

# ──────────────────────────────── VECTOR STORE ─────────────────────────────────
class VectorStore:
    def __init__(self, collection_name: str = "customer_support_docs"):
        self.collection_name = collection_name
        self.enabled = False
        if CHROMADB_AVAILABLE and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.client = chromadb.PersistentClient(path=CHROMA_PATH)  # type: ignore
                self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
                try:
                    self.collection = self.client.get_collection(collection_name)  # type: ignore
                    logger.info(f"Loaded existing collection: {collection_name}")
                except Exception:
                    self.collection = self.client.create_collection(  # type: ignore
                        collection_name,
                        metadata={"hnsw:space": "cosine"}
                    )
                    logger.info(f"Created new collection: {collection_name}")
                self.enabled = True
            except Exception as e:
                logger.error(f"ChromaDB initialization failed: {e}")
                self.enabled = False

    def add_documents(self, chunks: List[Dict[str, Any]]):
        if not self.enabled or not chunks:
            return
        try:
            existing_ids = set()
            try:
                existing = self.collection.get()  # type: ignore
                if existing and existing.get("ids"):
                    existing_ids = set(existing["ids"])
            except Exception:
                pass

            new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]
            if not new_chunks:
                logger.info("No new documents to add")
                return

            texts = [c["text"] for c in new_chunks]
            embeddings = self.encoder.encode(texts, show_progress_bar=False).tolist()
            ids = [c["chunk_id"] for c in new_chunks]
            metadatas = [{
                "source": c["source"],
                "page": c["page"],
                "total_pages": c["total_pages"],
                "word_count": c["word_count"],
                "char_count": c["char_count"]
            } for c in new_chunks]

            batch = 100
            for i in range(0, len(new_chunks), batch):
                j = min(i + batch, len(new_chunks))
                self.collection.add(  # type: ignore
                    embeddings=embeddings[i:j],
                    documents=texts[i:j],
                    metadatas=metadatas[i:j],
                    ids=ids[i:j]
                )
            logger.info(f"Added {len(new_chunks)} chunks to vector store")
        except Exception as e:
            logger.error(f"Error adding documents: {e}")

    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        try:
            q_emb = self.encoder.encode([query]).tolist()
            res = self.collection.query(  # type: ignore
                query_embeddings=q_emb,
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            out = []
            if res and res.get("documents"):
                for i, doc in enumerate(res["documents"][0]):
                    score = 1 - res["distances"][0][i]
                    md = res["metadatas"][0][i]
                    out.append({
                        "text": doc,
                        "source": md["source"],
                        "page": md["page"],
                        "total_pages": md.get("total_pages", 0),
                        "score": score,
                        "word_count": md.get("word_count", 0)
                    })
            return out
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        if not self.enabled:
            return {"total_chunks": 0, "unique_documents": 0}
        try:
            data = self.collection.get()  # type: ignore
            total = len(data["ids"]) if data and data.get("ids") else 0
            if total > 0:
                sources, total_pages, total_words = set(), 0, 0
                for md in data["metadatas"]:
                    sources.add(md.get("source", ""))
                    total_pages = max(total_pages, md.get("total_pages", 0))
                    total_words += md.get("word_count", 0)
                return {
                    "total_chunks": total,
                    "unique_documents": len([s for s in sources if s]),
                    "total_pages": total_pages,
                    "total_words": total_words,
                    "sources": list(sources)
                }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
        return {"total_chunks": 0, "unique_documents": 0}

# ──────────────────────────────── OPENAI RESPONSES ────────────────────────────
class OpenAIResponseGenerator:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.enabled = bool(self.api_key and OPENAI_AVAILABLE)
        self.model_default = "gpt-3.5-turbo"
        if self.enabled:
            if OPENAI_V1:
                # New SDK v1.x
                try:
                    self.client = OpenAIClient(api_key=self.api_key)  # type: ignore
                    logger.info("OpenAI v1 client initialized")
                except Exception as e:
                    logger.warning(f"OpenAI v1 init failed: {e}")
                    self.enabled = False
            else:
                # Legacy 0.x style
                try:
                    openai.api_key = self.api_key  # type: ignore
                    logger.info("OpenAI legacy client initialized")
                except Exception as e:
                    logger.warning(f"OpenAI legacy init failed: {e}")
                    self.enabled = False
        else:
            logger.warning("OpenAI integration disabled - API key not found or SDK missing")

    def generate_response(
        self,
        query: str,
        context_docs: List[Dict[str, Any]],
        chat_history: List[ChatMessage],
        company_config: Dict[str, Any],
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        if not self.enabled:
            return self._fallback_response(query, context_docs, company_config)

        try:
            system_prompt = self._create_system_prompt(company_config)
            user_prompt = self._create_user_prompt(query, context_docs, chat_history)

            if OPENAI_V1:
                # New SDK
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=1500,
                    temperature=temperature,
                    top_p=0.9
                )
                text = (resp.choices[0].message.content or "").strip()
                total_tokens = getattr(resp, "usage", None).total_tokens if getattr(resp, "usage", None) else 0
            else:
                # Legacy SDK
                resp = openai.ChatCompletion.create(  # type: ignore
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=1500,
                    temperature=temperature,
                    top_p=0.9
                )
                text = resp.choices[0].message.content.strip()  # type: ignore
                total_tokens = resp.usage.total_tokens if hasattr(resp, "usage") else 0  # type: ignore

            if context_docs:
                src_lines = ["", "**Sources:**"]
                for i, d in enumerate(context_docs[:3], 1):
                    src_lines.append(f"{i}. **{d['source']}** (Page {d['page']}) - Relevance: {d.get('score', 0):.1%}")
                text += "\n".join([""] + src_lines)

            return {
                "text": text,
                "sources": self._format_sources(context_docs),
                "confidence": self._calculate_confidence(context_docs),
                "model_used": model,
                "tokens_used": total_tokens,
            }
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return self._fallback_response(query, context_docs, company_config)

    def _create_system_prompt(self, cfg: Dict[str, Any]) -> str:
        return f"""You are an expert customer support assistant for {cfg['name']}.

COMPANY INFORMATION:
- Company: {cfg['name']}
- Email: {cfg['email']}
- Phone: {cfg['phone']}
- Website: {cfg['website']}
- Description: {cfg.get('description', 'Professional service provider')}

YOUR ROLE:
- Provide comprehensive, helpful, and accurate answers
- Use a professional yet friendly tone
- Base responses strictly on the provided documentation
- Structure answers clearly with headers and bullet points when helpful
- Include step-by-step instructions for procedures
- Acknowledge limitations when information is unclear
- Suggest creating support tickets when appropriate

IMPORTANT:
- Only use information from the provided context documents
- Never invent or assume information not explicitly stated
- Always prioritize accuracy over completeness
"""

    def _create_user_prompt(self, query: str, documents: List[Dict[str, Any]], history: List[ChatMessage]) -> str:
        if not documents:
            doc_context = "No relevant documentation found."
        else:
            parts = []
            for i, d in enumerate(documents[:5], 1):
                snippet = d["text"][:1200]
                if len(d["text"]) > 1200:
                    snippet += "..."
                parts.append(
                    f"DOCUMENT {i}:\nSource: {d['source']} (Page {d['page']} of {d.get('total_pages','?')})\n"
                    f"Relevance: {d.get('score', 0):.1%}\nContent:\n{snippet}\n"
                )
            doc_context = "\n".join(parts)

        if not history:
            hist = "No previous conversation."
        else:
            last = history[-6:]
            rows = []
            for m in last:
                role = "Customer" if m.role == "user" else "Assistant"
                content = m.content
                if len(content) > 300:
                    content = content[:300] + "..."
                rows.append(f"{role}: {content}")
            hist = "\n".join(rows)

        return f"""CONTEXT FROM COMPANY DOCUMENTATION:
{doc_context}

CONVERSATION HISTORY:
{hist}

CUSTOMER QUESTION:
{query}

Please answer strictly based on the documentation above. Structure with markdown headings and steps where relevant. Be concise but complete.
"""

    def _format_sources(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [{
            "source": d["source"],
            "page": d["page"],
            "relevance": d.get("score", 0),
            "word_count": d.get("word_count", 0)
        } for d in docs]

    def _calculate_confidence(self, docs: List[Dict[str, Any]]) -> float:
        if not docs:
            return 0.0
        avg = sum(d.get("score", 0) for d in docs) / len(docs)
        bonus = min(len(docs) * 0.1, 0.3)
        return min(avg + bonus, 1.0)

    def _fallback_response(self, query: str, docs: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[str, Any]:
        if not docs:
            text = f"""I couldn't find specific information about your question in our documentation.

I'd be happy to create a support ticket for you.

**To create a support ticket, please provide:**
- Your name and email address
- A detailed description of your issue

**Contact:**
{cfg['email']} • {cfg['phone']} • {cfg['website']}"""
            sources = []
            conf = 0.0
        else:
            d = docs[0]
            body = d["text"][:600] + ("..." if len(d["text"]) > 600 else "")
            text = f"""Based on our documentation:

**From {d['source']} (Page {d['page']}):**
{body}

**Sources:**
1. **{d['source']}** (Page {d['page']}) - Relevance: {d.get('score', 0.5):.1%}
"""
            sources = self._format_sources(docs)
            conf = 0.5
        return {"text": text, "sources": sources, "confidence": conf, "model_used": "fallback", "tokens_used": 0}

# ──────────────────────────────── GITHUB ISSUES ────────────────────────────────
class GitHubIssueManager:
    def __init__(self):
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.repo_owner = os.getenv('GITHUB_REPO_OWNER', '')
        self.repo_name = os.getenv('GITHUB_REPO_NAME', '')
        self.enabled = bool(self.github_token and self.repo_owner and self.repo_name and REQUESTS_AVAILABLE)
        if self.enabled:
            self.base_url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}"
            logger.info("GitHub integration enabled")
        else:
            logger.info("GitHub integration disabled")

    def create_github_issue(self, ticket: SupportTicket) -> Dict[str, Any]:
        if not self.enabled:
            return {"success": False, "error": "GitHub integration not configured"}
        issue_data = {
            "title": f"Support Ticket #{ticket.id}: {ticket.summary}",
            "body": self._format_issue_body(ticket),
            "labels": self._get_issue_labels(ticket)
        }
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        try:
            r = requests.post(f"{self.base_url}/issues", headers=headers, json=issue_data)
            if r.status_code == 201:
                jd = r.json()
                return {"success": True, "github_url": jd["html_url"], "github_number": jd["number"]}
            return {"success": False, "error": f"GitHub API error: {r.status_code}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _format_issue_body(self, t: SupportTicket) -> str:
        return f"""## Support Ticket Information

**Ticket ID:** {t.id}
**Customer:** {t.user_name}
**Email:** {t.user_email}
**Priority:** {t.priority}
**Category:** {t.category}
**Created:** {t.created_at}

## Issue Summary
{t.summary}

## Detailed Description
{t.description}

---

**Company:** {COMPANY_CONFIG['name']}
**Contact:** {COMPANY_CONFIG['email']} | {COMPANY_CONFIG['phone']}

---
*This issue was automatically created from a customer support ticket.*
"""

    def _get_issue_labels(self, t: SupportTicket) -> List[str]:
        labels = ["customer-support"]
        pr = {"Low": "priority: low", "Medium": "priority: medium", "High": "priority: high", "Urgent": "priority: urgent"}
        cat = {
            "Technical Issue": "bug", "Feature Request": "enhancement", "Account": "account",
            "Billing": "billing", "General": "question", "Bug Report": "bug", "Documentation": "documentation"
        }
        if t.priority in pr:
            labels.append(pr[t.priority])
        if t.category in cat:
            labels.append(cat[t.category])
        return labels

# ──────────────────────────────── TICKETS ─────────────────────────────────────
class TicketManager:
    def __init__(self):
        self.tickets_file = TICKETS_FILE_PATH
        self.tickets = self._load_tickets()
        self.github_manager = GitHubIssueManager()

    def _load_tickets(self) -> List[SupportTicket]:
        if os.path.exists(self.tickets_file):
            try:
                with open(self.tickets_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return [SupportTicket(**x) for x in data]
            except Exception as e:
                logger.error(f"Error loading tickets: {e}")
        return []

    def _save_tickets(self):
        try:
            with open(self.tickets_file, 'w', encoding='utf-8') as f:
                json.dump([asdict(t) for t in self.tickets], f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving tickets: {e}")

    def create_ticket(
        self, user_name: str, user_email: str, summary: str, description: str,
        priority: str = "Medium", category: str = "General", create_github_issue: bool = False
    ) -> Dict[str, Any]:
        stamp = datetime.now().strftime("%Y%m%d")
        num_today = len([t for t in self.tickets if t.created_at.startswith(stamp)]) + 1
        ticket_id = f"TK{stamp}-{num_today:03d}"

        ticket = SupportTicket(
            id=ticket_id,
            user_name=user_name,
            user_email=user_email,
            summary=summary,
            description=description,
            created_at=datetime.now().isoformat(),
            priority=priority,
            category=category
        )

        result = {
            "ticket": ticket,
            "github_created": False,
            "github_url": None,
            "github_number": None,
            "github_error": None
        }

        if create_github_issue and self.github_manager.enabled:
            gh = self.github_manager.create_github_issue(ticket)
            if gh["success"]:
                ticket.github_url = gh["github_url"]  # type: ignore
                ticket.github_number = gh["github_number"]  # type: ignore
                result["github_created"] = True
                result["github_url"] = gh["github_url"]
                result["github_number"] = gh["github_number"]
            else:
                result["github_error"] = gh["error"]

        self.tickets.append(ticket)
        self._save_tickets()
        logger.info(f"Created ticket {ticket.id} for {user_email}")
        return result

    def get_all_tickets(self) -> List[SupportTicket]:
        return self.tickets

    def get_stats(self) -> Dict[str, Any]:
        if not self.tickets:
            return {"total": 0, "open": 0, "closed": 0, "recent_week": 0, "priority_distribution": {}, "resolution_rate": 0}
        total = len(self.tickets)
        open_cnt = len([t for t in self.tickets if t.status == "Open"])

        # last 7 days
        from datetime import timedelta
        week_ago = datetime.now() - timedelta(days=7)
        recent = len([t for t in self.tickets if datetime.fromisoformat(t.created_at) > week_ago])

        pr = {}
        for t in self.tickets:
            pr[t.priority] = pr.get(t.priority, 0) + 1

        return {
            "total": total,
            "open": open_cnt,
            "closed": total - open_cnt,
            "recent_week": recent,
            "priority_distribution": pr,
            "resolution_rate": ((total - open_cnt) / total * 100) if total else 0
        }

# ──────────────────────────────── APP WRAPPER ─────────────────────────────────
class CustomerSupportApp:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.response_generator = OpenAIResponseGenerator()
        self.ticket_manager = TicketManager()

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "documents_loaded" not in st.session_state:
            st.session_state.documents_loaded = False
        if "doc_stats" not in st.session_state:
            st.session_state.doc_stats = {"total_chunks": 0}
        if "total_tokens_used" not in st.session_state:
            st.session_state.total_tokens_used = 0
        if "initialization_done" not in st.session_state:
            st.session_state.initialization_done = False

    def initialize_preloaded_documents(self):
        if st.session_state.initialization_done:
            return True
        try:
            chunks = self.doc_processor.load_preloaded_documents()
            if chunks:
                self.vector_store.add_documents(chunks)
                st.session_state.documents_loaded = True
                st.session_state.doc_stats = self.vector_store.get_stats()
                st.session_state.initialization_done = True
                logger.info(f"Initialized with {len(chunks)} document chunks")
                return True
            else:
                logger.warning("No documents found to load")
                return False
        except Exception as e:
            logger.error(f"Error initializing documents: {e}")
            return False

    def process_query(self, query: str, model: str = "gpt-3.5-turbo", temperature: float = 0.7) -> Dict[str, Any]:
        relevant = self.vector_store.search(query, n_results=5)
        data = self.response_generator.generate_response(
            query, relevant, st.session_state.chat_history, COMPANY_CONFIG, model, temperature
        )

        user_msg = ChatMessage(role="user", content=query, timestamp=datetime.now().isoformat())
        asst_msg = ChatMessage(
            role="assistant",
            content=data["text"],
            timestamp=datetime.now().isoformat(),
            sources=data["sources"],
            confidence=data["confidence"],
            model_used=data["model_used"],
            tokens_used=data["tokens_used"]
        )
        st.session_state.chat_history.extend([user_msg, asst_msg])
        if len(st.session_state.chat_history) > 20:
            st.session_state.chat_history = st.session_state.chat_history[-20:]
        st.session_state.total_tokens_used += data["tokens_used"]
        return data

# ──────────────────────────────── UI HELPERS ──────────────────────────────────
def setup_ai_configuration():
    with st.sidebar:
        st.markdown("---")
        st.header("AI Configuration")

        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key and OPENAI_AVAILABLE:
            st.success("OpenAI integration active")
            model = st.selectbox("AI Model", ["gpt-3.5-turbo", "gpt-4-turbo-preview"])
            temperature = st.slider("Response Creativity", 0.0, 1.0, 0.7, 0.1)
            if st.session_state.get('total_tokens_used', 0) > 0:
                st.metric("Session Tokens", st.session_state.total_tokens_used)
            return model, temperature
        else:
            st.warning("OpenAI not configured")
            if not OPENAI_AVAILABLE:
                st.error("OpenAI SDK not installed or incompatible")
            else:
                st.info("Set OPENAI_API_KEY in .env")
            return "gpt-3.5-turbo", 0.7

def setup_github_integration():
    with st.sidebar:
        st.markdown("---")
        st.header("GitHub Integration")
        github_token = os.getenv('GITHUB_TOKEN')
        github_owner = os.getenv('GITHUB_REPO_OWNER')
        github_repo = os.getenv('GITHUB_REPO_NAME')

        if github_token and github_owner and github_repo and REQUESTS_AVAILABLE:
            st.success("GitHub integration active")
            st.info(f"Repository: {github_owner}/{github_repo}")
            return True
        else:
            st.warning("GitHub integration disabled")
            missing = []
            if not github_token: missing.append("GITHUB_TOKEN")
            if not github_owner: missing.append("GITHUB_REPO_OWNER")
            if not github_repo: missing.append("GITHUB_REPO_NAME")
            if not REQUESTS_AVAILABLE: missing.append("requests library")
            if missing: st.info(f"Missing: {', '.join(missing)}")
            return False

def display_knowledge_base_stats():
    with st.sidebar:
        st.markdown("---")
        st.header("Knowledge Base")
        stats = st.session_state.doc_stats
        if stats.get("total_chunks", 0) > 0:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", stats.get("unique_documents", 0))
                st.metric("Pages", stats.get("total_pages", 0))
            with col2:
                st.metric("Text Chunks", stats.get("total_chunks", 0))
                st.metric("Total Words", f"{stats.get('total_words', 0):,}")
            if stats.get("sources"):
                with st.expander("Loaded Documents"):
                    for source in stats["sources"]:
                        st.write(f"• {source}")
            if stats.get("total_words", 0) and stats.get("total_chunks", 0):
                avg = stats["total_words"] / stats["total_chunks"]
                st.metric("Avg Words/Chunk", f"{avg:.0f}")
        else:
            st.info("No documents loaded yet")

def display_company_info():
    with st.sidebar:
        st.markdown("---")
        st.header("Company Information")
        st.write(f"**{COMPANY_CONFIG['name']}**")
        st.write(f"{COMPANY_CONFIG['email']}")
        st.write(f"{COMPANY_CONFIG['phone']}")
        st.write(f"{COMPANY_CONFIG['website']}")
        if COMPANY_CONFIG.get('description'):
            st.write(f"*{COMPANY_CONFIG['description']}*")
        if COMPANY_CONFIG.get('business_hours'):
            st.write(f"{COMPANY_CONFIG['business_hours']}")

def display_support_analytics():
    with st.sidebar:
        st.markdown("---")
        st.header("Support Analytics")
        app = st.session_state.get('app')
        if app:
            stats = app.ticket_manager.get_stats()
            if stats["total"] > 0:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Tickets", stats["total"])
                    st.metric("This Week", stats["recent_week"])
                with col2:
                    st.metric("Open Tickets", stats["open"])
                    st.metric("Resolution Rate", f"{stats['resolution_rate']:.0f}%")
                if stats.get("priority_distribution"):
                    with st.expander("Priority Breakdown"):
                        for p, c in stats["priority_distribution"].items():
                            st.write(f"• {p}: {c}")
            else:
                st.info("No tickets created yet")

def create_sample_documents_notice() -> bool:
    documents_dir = DOCUMENTS_DIR
    os.makedirs(documents_dir, exist_ok=True)
    existing_pdfs = glob.glob(os.path.join(documents_dir, "*.pdf"))
    if not existing_pdfs:
        st.warning(f"No PDF documents found in '{documents_dir}' directory.")
        st.info("""
**To use this application:**
1) Create a 'documents' folder next to app.py
2) Add at least 3 PDF documents
3) Ensure ≥2 PDFs and one has 400+ pages
4) Restart the app
""")
        return False
    return True

# ──────────────────────────────── MAIN ────────────────────────────────────────
def main():
    st.set_page_config(page_title="AI-Powered Customer Support", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem; border-radius: 15px; color: white;
        text-align: center; margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .success-message { background:#d4edda; color:#155724; padding:1rem; border-radius:10px; border-left:4px solid #28a745;}
    .info-box { background:#e3f2fd; color:#0d47a1; padding:1rem; border-radius:10px; border-left:4px solid #2196f3; margin:1rem 0;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="main-header">
        <h1>🤖 AI-Powered Customer Support Assistant</h1>
        <p>Intelligent responses powered by pre-loaded documentation and advanced RAG</p>
        <p><strong>{COMPANY_CONFIG['name']}</strong> • Advanced RAG • GitHub Integration</p>
    </div>
    """, unsafe_allow_html=True)

    missing = []
    if not (PYMUPDF_AVAILABLE or PYPDF2_AVAILABLE):
        missing.append("PyMuPDF or PyPDF2")
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        missing.append("sentence-transformers")
    if not CHROMADB_AVAILABLE:
        logger.info("Chroma is disabled — vector search will be unavailable.")
    if missing:
        st.error(f"Missing dependencies: {', '.join(missing)}")
        st.info("Please install missing dependencies and restart the application.")
        st.stop()

    if not create_sample_documents_notice():
        st.stop()

    if 'app' not in st.session_state:
        st.session_state.app = CustomerSupportApp()

    app = st.session_state.app

    if not st.session_state.get('initialization_done', False):
        with st.spinner("Loading pre-configured documents... Please wait."):
            if app.initialize_preloaded_documents():
                st.success("Documents loaded successfully! AI is ready to answer questions.")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error("Failed to load documents. Please check the documents directory.")

    model, temperature = setup_ai_configuration()
    github_enabled = setup_github_integration()
    display_knowledge_base_stats()
    display_company_info()
    display_support_analytics()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("💬 AI Chat Assistant")
        if st.session_state.documents_loaded:
            st.markdown('<div class="info-box"><strong>🎯 Ready to help!</strong><br>Ask me anything about our documentation. I can provide detailed answers with source citations.</div>', unsafe_allow_html=True)

        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message.role):
                    st.write(message.content)
                    if message.role == "assistant":
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            conf = getattr(message, "confidence", 0.0)
                            color = "🟢" if conf > 0.8 else "🟡" if conf > 0.5 else "🔴"
                            st.metric("Confidence", f"{color} {conf:.1%}")
                        with c2:
                            mdl = getattr(message, "model_used", "fallback")
                            st.metric("AI Model", mdl.replace("-", " ").title())
                        with c3:
                            toks = getattr(message, "tokens_used", 0)
                            if toks:
                                st.metric("Tokens Used", f"{toks:,}")
                        if message.sources:
                            with st.expander(f"📚 Sources ({len(message.sources)} documents)"):
                                for i, s in enumerate(message.sources, 1):
                                    rel = s.get("relevance", 0)
                                    icon = "🎯" if rel > 0.8 else "📍" if rel > 0.5 else "📌"
                                    st.write(f"{icon} **{s['source']}** (Page {s['page']}) - {rel:.1%} relevance")

        if st.session_state.documents_loaded:
            if prompt := st.chat_input("Ask me anything about our documentation..."):
                with st.chat_message("user"):
                    st.write(prompt)
                with st.chat_message("assistant"):
                    with st.spinner("🧠 Analyzing documentation and generating response..."):
                        resp = app.process_query(prompt, model, temperature)
                        st.write(resp["text"])
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            conf = resp['confidence']
                            color = "🟢" if conf > 0.8 else "🟡" if conf > 0.5 else "🔴"
                            st.metric("Confidence", f"{color} {conf:.1%}")
                        with c2:
                            st.metric("AI Model", resp['model_used'].replace("-", " ").title() if resp['model_used'] != "fallback" else "Basic Search")
                        with c3:
                            if resp['tokens_used'] > 0:
                                st.metric("Tokens Used", f"{resp['tokens_used']:,}")
                        if resp['confidence'] < 0.3:
                            st.info("💡 **Need more help?** Consider creating a support ticket for personalized assistance.")
                        st.success("✅ Response generated successfully!")
                        time.sleep(0.3)
                        st.rerun()
        else:
            st.info("⚠️ **Documents are loading...** Please wait for initialization to complete.")

    with col2:
        st.header("🎫 Create Support Ticket")
        with st.form("enhanced_ticket_form"):
            st.subheader("👤 Contact Information")
            user_name = st.text_input("Full Name*", placeholder="Enter your full name")
            user_email = st.text_input("Email Address*", placeholder="your.email@company.com")

            st.subheader("🎯 Issue Details")
            category = st.selectbox("Category", ["General", "Technical Issue", "Account", "Billing", "Feature Request", "Bug Report", "Documentation"])
            priority = st.selectbox("Priority", ["Low", "Medium", "High", "Urgent"])
            summary = st.text_input("Issue Summary*", placeholder="Brief, clear description of your issue")
            description = st.text_area(
                "Detailed Description*",
                placeholder="Please provide detailed information including:\n• Steps to reproduce\n• Expected vs actual behavior\n• Error messages\n• System information",
                height=150
            )
            create_github = False
            if github_enabled:
                create_github = st.checkbox("🔗 Create GitHub Issue", value=True, help="Also create a public GitHub issue for tracking")

            submitted = st.form_submit_button("🎫 Create Ticket", type="primary")

            if submitted:
                if user_name and user_email and summary and description:
                    # ✅ FIXED: closed string + anchored end
                    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                    if re.match(email_pattern, user_email):
                        result = app.ticket_manager.create_ticket(
                            user_name, user_email, summary, description, priority, category, create_github
                        )
                        ticket = result["ticket"]
                        st.markdown(f"""
                        <div class="success-message">
                            <h4>✅ Ticket Created Successfully!</h4>
                            <p><strong>Ticket ID:</strong> {ticket.id}</p>
                            <p><strong>Priority:</strong> {ticket.priority}</p>
                            <p><strong>Category:</strong> {ticket.category}</p>
                            {"<p><strong>GitHub:</strong> " + result['github_url'] + "</p>" if result.get("github_created") else ""}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error("Please enter a valid email address.")
                else:
                    st.error("Please fill all required fields (*).")

if __name__ == "__main__":
    main()
