"""
Costruzione dei due vector store richiesti dal piano:
1. KB statica: paper Brighton (sinonimi DVT) -> non cambia mai tra i pazienti.
2. KB dinamica: cartella clinica (EHR) del singolo paziente -> ricreata ad ogni run.
"""

import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool

import config


def get_embeddings():
    """Modello di embedding leggero, locale, non impatta il budget RAM del LLM."""
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)


def build_brighton_kb(brighton_pdf_text: str, embeddings=None, force_rebuild: bool = False) -> Chroma:
    """
    KB statica dei sinonimi DVT (Brighton Collaboration paper).
    Va costruita una sola volta e riusata per tutti i pazienti: se esiste gia'
    su disco e force_rebuild=False, la ricarica invece di ricalcolare gli embedding.
    """
    embeddings = embeddings or get_embeddings()

    if os.path.isdir(config.BRIGHTON_KB_PERSIST_DIR) and not force_rebuild:
        return Chroma(
            persist_directory=config.BRIGHTON_KB_PERSIST_DIR,
            embedding_function=embeddings,
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_text(brighton_pdf_text)

    return Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=config.BRIGHTON_KB_PERSIST_DIR,
        metadatas=[{"source": "brighton_dvt_synonyms"} for _ in chunks],
    )


def build_ehr_kb(patient_record_text: str, patient_id: str, embeddings=None) -> Chroma:
    """
    KB dinamica: cartella clinica del singolo paziente.
    Persistita in una sottocartella dedicata per paziente, cosi' run diversi
    non si sovrascrivono a vicenda.
    """
    embeddings = embeddings or get_embeddings()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.EHR_CHUNK_SIZE,
        chunk_overlap=config.EHR_CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(patient_record_text)

    persist_dir = f"{config.EHR_KB_PERSIST_DIR}_{patient_id}"

    return Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        metadatas=[{"source": "ehr", "patient_id": patient_id} for _ in chunks],
    )


def make_ehr_retriever_tool(ehr_vectorstore: Chroma):
    """
    Espone il retriever EHR come Tool LangChain, per l'eventuale Agente 1 agentico
    (vedi agents.py — USE_AGENTIC_EXTRACTOR in config.py).
    Se si usa il retrieval diretto (default), questo tool non e' necessario:
    si chiama direttamente ehr_vectorstore.as_retriever().invoke(query).
    """
    ehr_retriever = ehr_vectorstore.as_retriever(search_kwargs={"k": config.EHR_RETRIEVER_K})
    return create_retriever_tool(
        ehr_retriever,
        "search_patient_record",
        "Usa questo tool per cercare sintomi, referti chirurgici, date ed esami di "
        "laboratorio nella cartella clinica del paziente.",
    )


def load_brighton_pdf_text(pdf_path: str) -> str:
    """
    Estrae il testo dal PDF del paper Brighton. Usa pypdf (libreria standard,
    non richiede setup aggiuntivo oltre pip install pypdf).
    """
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)
