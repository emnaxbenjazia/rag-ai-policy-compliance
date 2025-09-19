# rag.py — Stateless RAG app over a fixed FAISS index built by ingest.py

from dotenv import load_dotenv
import os
from pathlib import Path
import warnings
import streamlit as st
from PyPDF2.errors import PdfReadWarning

#  noisy but harmless warnings
warnings.filterwarnings("ignore", category=PdfReadWarning)
warnings.filterwarnings("ignore", message="`resume_download` is deprecated", category=FutureWarning)

# LangChain vectorstore only (retriever)
from langchain_community.vectorstores import FAISS  # used implicitly via ingest.load_index()
# Load the persisted FAISS index via our ingestion module
from ingest import load_index, rebuild_index, CORPUS_DIR

from openai import OpenAI

# UI templates
from htmlTemplates import css, bot_template, user_template

#  Prompt template (plain string) 
QA_TEMPLATE = """You are an expert explainer of EU AI & data protection policy (AI Act, GDPR, EDPB/CNIL).
First, answer in your own words (no legalese). Then, you MAY include brief quotes from the context to support key points.
Always add inline citations like [S1], [S2] right after the statements they support.

Format:
Answer: 2–4 sentences in your own words, clear and direct.
Evidence (optional):
- "<short quote>" [S#]
- "<short quote>" [S#]

If the information is not present in the context, say you don't know.

Question: {question}

Context (excerpts from the corpus):
{context}
"""


#  LLM client
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Missing OPENAI_API_KEY in environment (.env).")
        return None
    base_url = os.getenv("OPENAI_BASE_URL") 
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)

def llm_answer(client: OpenAI, model: str, prompt: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content

# Build a stateless RAG “chain” 
def make_retriever(vectorstore):
    return vectorstore.as_retriever(search_kwargs={"k": 8})

def answer_question(retriever, client, model, question: str):
    # Retrieve source docs
    docs = retriever.get_relevant_documents(question)
    # Build context
    def clip(txt, max_chars=4000): 
        return txt[:max_chars]
    context = "\n\n".join(clip(d.page_content) for d in docs)
    # Compose prompt
    prompt = QA_TEMPLATE.format(question=question, context=context)
    # Call LLM
    answer = llm_answer(client, model, prompt)
    return answer, docs

# Q/A rendering 
def render_answer(question: str, answer: str, docs):
    # chat bubbles
    st.write(user_template.replace("{{MSG}}", question), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", answer), unsafe_allow_html=True)

    # sources
    if docs:
        st.markdown("#### Sources")
        cols = st.columns(3)
        for i, d in enumerate(docs):
            meta = d.metadata or {}
            title = meta.get("source", "Document")
            page = meta.get("page")
            with cols[i % 3]:
                st.markdown(f"**[S{i+1}] {title}**" + (f" — p.{page}" if page else ""))
                snippet = (d.page_content or "").strip()
                st.caption(snippet[:300] + ("..." if len(snippet) > 300 else ""))

# App
def main():
    load_dotenv(override=True)
    st.set_page_config(page_title="EU AI & Data Policy RAG", page_icon="📘", layout="wide")
    st.write(css, unsafe_allow_html=True)

    # Prepare session objects
    if "vs" not in st.session_state:
        st.session_state.vs = None
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "client" not in st.session_state:
        st.session_state.client = None

    st.header("EU Regulations - AI & Data Policy RAG 📘")
    st.caption("Answers are grounded ONLY in our curated legal corpus.")

    # Load FAISS index & client once
    if st.session_state.vs is None:
        with st.spinner("Opening index…"):
            st.session_state.vs = load_index()  # from ingest.py 
            st.session_state.retriever = make_retriever(st.session_state.vs)
        st.success("Index loaded.")

    if st.session_state.client is None:
        st.session_state.client = get_openai_client()
        if st.session_state.client is None:
            return 

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Ask box
    question = st.text_input("Ask a question about EU AI regulation & data policy:")
    if question:
        with st.spinner("Thinking…"):
            answer, docs = answer_question(st.session_state.retriever, st.session_state.client, model, question)
        render_answer(question, answer, docs)

    # Sidebar: corpus info & maintenance
    with st.sidebar:
        st.subheader("Corpus")
        st.write(f"Folder: `{CORPUS_DIR}`")
        pdfs = sorted([p.name for p in CORPUS_DIR.glob("*.pdf")])
        if pdfs:
            st.markdown("**Included PDFs:**")
            for name in pdfs:
                st.markdown(f"- {name}")
        else:
            st.info("Add your legal PDFs to the folder above.")

        if st.button("Rebuild Index"):
            with st.spinner("Rebuilding index from folder…"):
                st.session_state.vs = rebuild_index()
                st.session_state.retriever = make_retriever(st.session_state.vs)
            st.success("Index rebuilt. Ask a question above.")

if __name__ == '__main__':
    main()
