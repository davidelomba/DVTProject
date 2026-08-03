"""
Builds the two vector stores required by the plan:
1. Static KB: Brighton paper (DVT synonyms) -- never changes between patients.
2. Dynamic KB: the single patient's clinical record (EHR) -- rebuilt each run.
"""

import os
import shutil

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from langchain_core.tools.retriever import create_retriever_tool
import config


def get_embeddings():
    """Lightweight, local embedding model; does not impact the LLM's RAM budget.

    intfloat/multilingual-e5-small is trained to expect a "query: "/"passage: "
    prefix on every input depending on its role (its own model card warns of a
    performance degradation without it) -- verified against the installed
    langchain-huggingface: embed_documents() uses encode_kwargs, embed_query()
    uses query_encode_kwargs (falling back to encode_kwargs if empty), both
    passed straight through to sentence-transformers' encode(prompt=...).
    """
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        encode_kwargs={"prompt": "passage: "},
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
    or "agentic" (agents.extract_evidence_agentic, via the retriever tool
    built by make_ehr_retriever_tool below) -- not used when EXTRACTOR_MODE
    is "full_text" (the default), which passes the record directly instead.
    Persisted in a dedicated per-patient subfolder, so different runs
    don't overwrite one another.
    """
    embeddings = embeddings or get_embeddings()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.EHR_CHUNK_SIZE,
        chunk_overlap=config.EHR_CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(patient_record_text)

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


def make_ehr_retriever_tool(ehr_vectorstore: Chroma):
    """
    Wraps the EHR retriever as a LangChain Tool, used by the agentic
    extractor (agents.extract_evidence_agentic) when config.EXTRACTOR_MODE
    is "agentic" -- see pipeline.py for where this is called.

    Requires the base `langchain` package (not just langchain-core/
    langchain-community/langchain-ollama).
    """

    ehr_retriever = ehr_vectorstore.as_retriever(search_kwargs={"k": config.EHR_RETRIEVER_K})
    return create_retriever_tool(
        ehr_retriever,
        "search_patient_record",
        # Deliberately spells out every clinical domain touched by the 10
        # questionnaire criteria (A1, A2, A3_1/A3_2, B1_1/B1_2/B2, C, F, X) --
        # the same description is reused unchanged across all sections, so a
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


def load_brighton_pdf_text(pdf_path: str) -> str:
    """
    Extracts text from the Brighton paper PDF using pypdf.
    """
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_ehr_text(txt_path: str) -> str:
    """
    Reads the patient's clinical record from a plain .txt file.
    If you later switch to PDF or DOCX clinical records, add an equivalent
    loader here (e.g. reuse load_brighton_pdf_text's approach for PDF, or
    python-docx for Word files) and call the right one from pipeline.py.
    """
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read()
