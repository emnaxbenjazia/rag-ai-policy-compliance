# ingest.py  — offline indexing for EU AI/Data policy corpus (FAISS)

from pathlib import Path
import shutil
import warnings
from typing import List, Tuple

from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadWarning
warnings.filterwarnings("ignore", category=PdfReadWarning)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---- Paths (edit if you like)
CORPUS_DIR = Path("corpus")                 # put your 7 PDFs here
FAISS_DIR  = Path("index/eu_ai_policies")   # where the index is saved

CORPUS_DIR.mkdir(parents=True, exist_ok=True)
FAISS_DIR.parent.mkdir(parents=True, exist_ok=True)

# ---- Config
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def _load_pages(corpus_dir: Path) -> List[Tuple[str, str, int]]:
    """Return [(page_text, filename, page_no), ...] from all PDFs in corpus/."""
    pages: List[Tuple[str, str, int]] = []
    for p in sorted(corpus_dir.glob("*.pdf")):
        try:
            with open(p, "rb") as fh:
                reader = PdfReader(fh)
                for i, page in enumerate(reader.pages, start=1):
                    txt = page.extract_text() or ""
                    if txt.strip():
                        pages.append((txt, p.name, i))
        except Exception as e:
            print(f"[warn] Could not read {p.name}: {e}")
    return pages

def _chunk_pages(pages: List[Tuple[str, str, int]]):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    texts, metas = [], []
    for txt, source, page_no in pages:
        for chunk in splitter.split_text(txt):
            texts.append(chunk)
            metas.append({"source": source, "page": page_no})
    return texts, metas

def _embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )

def build_index():
    """Build FAISS index from PDFs in CORPUS_DIR (no delete)."""
    pages = _load_pages(CORPUS_DIR)
    texts, metas = _chunk_pages(pages)
    if not texts:
        raise SystemExit("No extractable text found. If PDFs are scanned, run OCR and retry.")
    vs = FAISS.from_texts(texts=texts, metadatas=metas, embedding=_embeddings())
    vs.save_local(str(FAISS_DIR))
    print(f"[ok] FAISS index built at {FAISS_DIR}")

def rebuild_index():
    """Delete existing FAISS index and rebuild from CORPUS_DIR."""
    if FAISS_DIR.exists():
        shutil.rmtree(FAISS_DIR)
    FAISS_DIR.parent.mkdir(parents=True, exist_ok=True)
    build_index()
    return load_index()

def load_index():
    """Open the already-built FAISS index."""
    return FAISS.load_local(str(FAISS_DIR), _embeddings(), allow_dangerous_deserialization=True)

if __name__ == "__main__":
    # CLI:
    #   python ingest.py         -> build (or overwrite existing files)
    #   python ingest.py rebuild -> force rebuild
    import sys
    if len(sys.argv) > 1 and sys.argv[1].lower() == "rebuild":
        rebuild_index()
    else:
        build_index()
