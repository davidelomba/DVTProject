"""
Prepares the retrieval side of the pipeline: the embedding model, the two
vector stores, the loaders that feed them and the retriever tool the agentic
extractor calls.

STORES
1. Brighton KB: the guideline paper, identical for every patient, so it is
   built once and reloaded from disk unless force_rebuild.
2. EHR KB: one patient's clinical record, wiped and rebuilt on every run.
   Built only for the "rag" and "agentic_graph" extraction modes; "full_text"
   passes the record in the prompt instead.

Both source texts are cleaned before reaching Agent 2: the paper's reference
list is dropped at load time, and bibliography lines surviving into a
retrieved chunk are stripped from the context.
"""

import os
import re
import shutil

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from langchain_core.tools.retriever import create_retriever_tool
import config


def get_embeddings():
    """Builds the embedding model shared by every vector store.

    Multilingual by necessity: the retrieval queries are in English while the
    records are in Italian.

    intfloat/multilingual-e5-small expects a role prefix on each input
    ("query: " for a search query, "passage: " for a stored chunk) and its
    model card reports degraded retrieval without it. langchain-huggingface
    routes the two separately: embed_documents() applies encode_kwargs,
    embed_query() applies query_encode_kwargs (falling back to encode_kwargs
    when empty), both forwarded to sentence-transformers' encode(prompt=...).
    """

    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        # Applied when embedding stored chunks (embed_documents()).
        encode_kwargs={"prompt": "passage: "},
        # Applied when embedding a search query (embed_query()).
        query_encode_kwargs={"prompt": "query: "},
    )


def build_brighton_kb(brighton_pdf_text: str, embeddings=None, force_rebuild: bool = False) -> Chroma:
    """
    Static KB of DVT synonyms (Brighton Collaboration paper).
    Should be built once and reused for all patients: if it already exists
    on disk and force_rebuild=False, it is reloaded instead of recomputing
    embeddings.
    """

    embeddings = embeddings or get_embeddings()

    # Reuse the existing index instead of recomputing embeddings: the
    # Brighton paper is static and identical across every patient/run.
    if os.path.isdir(config.BRIGHTON_KB_PERSIST_DIR) and not force_rebuild:
        return Chroma(
            persist_directory=config.BRIGHTON_KB_PERSIST_DIR,
            embedding_function=embeddings,
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.BRIGHTON_CHUNK_SIZE,
        chunk_overlap=config.BRIGHTON_CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(brighton_pdf_text)

    # Builds the index from scratch and persists it to disk.
    return Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=config.BRIGHTON_KB_PERSIST_DIR,
        metadatas=[{"source": "brighton_dvt_synonyms"} for _ in chunks],
    )


def build_ehr_kb(patient_record_text: str, patient_id: str, embeddings=None) -> Chroma:
    """
    Dynamic KB: the single patient's clinical record, chunked and embedded.
    Only needed when config.EXTRACTOR_MODE is "rag" (agents.extract_evidence)
    or "agentic_graph" (agents.extract_evidence_agentic, via the retriever
    tool built by make_ehr_retriever_tool below); not used when
    EXTRACTOR_MODE is "full_text" (the default), which passes the record
    directly instead.
    Persisted in a dedicated per-patient subfolder, so different runs
    don't overwrite one another.
    """

    embeddings = embeddings or get_embeddings()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.EHR_CHUNK_SIZE,
        chunk_overlap=config.EHR_CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(patient_record_text)

    # Suffixed per patient_id so concurrent/different runs don't clash.
    persist_dir = f"{config.EHR_KB_PERSIST_DIR}_{patient_id}"

    # This KB is meant to be rebuilt fresh on every run (see module docstring).
    # Without removing the old directory first, Chroma.from_texts() appends
    # to whatever is already persisted there, so re-running the pipeline on
    # the same patient_id (e.g. during debugging) silently accumulates
    # duplicate chunks across runs, diluting retrieval relevance over time.
    if os.path.isdir(persist_dir):
        shutil.rmtree(persist_dir)

    return Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        metadatas=[{"source": "ehr", "patient_id": patient_id} for _ in chunks],
    )


# Lines of the Brighton paper that carry no clinical meaning: numbered
# bibliography entries, DOIs, URLs and journal citations. The PDF is chunked
# whole, so a retrieved chunk routinely contains a run of references and they
# reach Agent 2 as if they were reference terminology.
_BIBLIOGRAPHY_LINE = re.compile(
    # A citation marker starting the line counts only when an author name
    # follows it. Without that condition the pattern also matched body text
    # wrapped after a closing citation ("[69]. TTS is characterized by
    # thrombosis..."), deleting a clinically meaningful line.
    r"^\s*\[\d+\]\s+[A-Z]"  # "[83] Goodman LR, Stein PD, ..."
    r"|https?://"           # bare URLs
    r"|doi\.org"            # DOI links
    r"|\bdoi:\s*10\."       # inline DOIs
    r"|^\s*10\.\d{4}/"      # a DOI on a line of its own
    r"|\b\d{4};\s*\d+"      # journal volume citations: "2007;189(5):1071-6"
    r"|Last accessed",
    re.IGNORECASE,
)


def clean_brighton_context(context: str) -> str:
    """Strips bibliographic noise from retrieved guideline context.

    A second line of defence: load_brighton_pdf_text already drops the whole
    reference list before indexing, so on this paper only a handful of stray
    DOI lines remain for this filter to catch. It matters more for a PDF whose
    reference section has no heading to truncate at.

    Args:
        context: the concatenated page content of the retrieved chunks.

    Returns:
        The same text without bibliography lines. Falls back to the original
        if filtering would remove everything, so a chunk that happens to be
        all references still yields something rather than an empty context.
    """

    kept = [ln for ln in context.splitlines() if not _BIBLIOGRAPHY_LINE.search(ln)]
    cleaned = "\n".join(kept).strip()
    return cleaned if cleaned else context


def make_ehr_retriever_tool(ehr_vectorstore: Chroma):
    """
    Wraps the EHR retriever as a LangChain Tool, used by the agentic
    extractor (agents.extract_evidence_agentic) when config.EXTRACTOR_MODE
    is "agentic_graph".

    Requires the base `langchain` package (not just langchain-core/
    langchain-community/langchain-ollama).
    """

    ehr_retriever = ehr_vectorstore.as_retriever(search_kwargs={"k": config.EHR_RETRIEVER_K})

    # Wraps the retriever as a LangChain Tool object the agent can call by
    # name ("search_patient_record") with a free-text query of its choosing.

    return create_retriever_tool(
        ehr_retriever,
        "search_patient_record",

        # Deliberately spells out every clinical domain touched by the 10
        # questionnaire criteria (A1, A2, A3_1/A3_2, B1_1/B1_2/B2, C, F, X).
        # The same description is reused unchanged across all sections, so a
        # narrower one risks the model not thinking to search for a domain
        # (e.g. autopsy, imaging modality, specialist diagnosis) it doesn't
        # explicitly mention.
        "Use this tool to search the patient's clinical record for any "
        "information relevant to a DVT (deep vein thrombosis) diagnosis: "
        "reported symptoms or signs (e.g. calf pain, swelling, redness), "
        "imaging studies and their outcomes (ultrasound, Doppler, CT/MR "
        "venography, contrast venography), autopsy or pathology findings, "
        "surgical procedures (e.g. thrombectomy), laboratory results (e.g. "
        "D-dimer values and reference ranges), diagnoses reported by a "
        "specialist, and any alternative diagnosis that could explain the "
        "symptoms.",
    )


# The reference list's heading: the word alone on a line, which in this paper
# occurs exactly once. load_brighton_pdf_text truncates the text here.
_REFERENCES_HEADING = re.compile(r"(?im)^[ \t]*references[ \t]*$")


def load_brighton_pdf_text(pdf_path: str, drop_references: bool = True) -> str:
    """Extracts the guideline text from the PDF, without its reference list.

    The reference list is roughly the last 40% of this paper and is pure noise
    for the pipeline: chunks falling in it are indexed and retrieved like any
    other and reach Agent 2 as if they were reference terminology. Dropping it
    before chunking removes it wholesale (including entries split across
    several lines, which a line-level filter cannot fully catch) and keeps
    the vector store limited to clinical content, so the retrieved chunks are
    chosen among useful text only.

    Args:
        pdf_path: path to the guideline PDF.
        drop_references: set False to keep the full text.

    Returns:
        The extracted text, truncated at the reference heading when one is
        found. A paper without such a heading yields the full text, keeping
        the reference list rather than losing content.
    """

    reader = PdfReader(pdf_path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)

    if drop_references:
        match = _REFERENCES_HEADING.search(text)
        if match:
            text = text[:match.start()].rstrip()

    return text


def load_ehr_text(txt_path: str) -> str:
    """Reads the patient's clinical record from a plain .txt file."""

    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read()
