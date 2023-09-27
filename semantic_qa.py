"""Implements basic semantic Q&A bot on a document collection"""

import sys
from enum import Enum
import datetime
import pathlib
import hashlib
from typing import Any
from pprint import pprint

import tqdm

from langchain.schema import Document
from langchain.document_loaders import DirectoryLoader

from langchain.text_splitter import (
    TextSplitter,
    RecursiveCharacterTextSplitter,
)

from langchain.embeddings.base import Embeddings
from langchain.embeddings import (
    OpenAIEmbeddings,
    HuggingFaceEmbeddings,
    HuggingFaceInstructEmbeddings,
)

from langchain.vectorstores.base import VectorStore
from langchain.vectorstores import (
    Chroma,
    Redis,
    PGVector,
    Pinecone,
    MongoDBAtlasVectorSearch,
)

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA

from pymongo.collection import Collection as MongodbCollection

from pydantic.utils import deep_update # pylint: disable=no-name-in-module

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


class VectordbProviders(Enum):
    """Types of vector db providers we currently support"""

    CHROMADB = "chromadb"
    REDIS = "redis"
    PGVECTOR = "pgvector"
    PINECONE = "pinecone"
    MONGODB_ATLAS = "mongodb_atlas"


class EmbeddingsProviders(Enum):
    """Types of embeddings providers we currently support"""

    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    INSTRUCTOR = "instructor"


toml_config: dict = {}

with open("config.toml", "rb") as f:
    toml_config = tomllib.load(f)
with open("secrets.toml", "rb") as g:
    toml_secrets = tomllib.load(g)

toml_config = deep_update(toml_config, toml_secrets)


def load_docs(directories: list[str], glob: str) -> list[Document]:
    """Iterates recursively over directories to find files that match the glob pattern

    Args:
        directories (list[str]): root folders containing our source documents
        glob (str): glob pattern

    Returns:
        list[Document]: list of fully qualified file names
    """
    documents = []
    for directory in directories:
        loader = DirectoryLoader(
            directory,
            recursive=True,
            show_progress=True,
            use_multithreading=True,
            glob=glob,
            # sample_size=20,
            # randomize_sample=True,
            # sample_seed=42,
        )
        documents += loader.load()
    return documents


def enrich_source_documents(documents: list[Document]) -> list[Document]:
    """Iterates over documents, enriching metadata and content with useful information

    Args:
        documents (list[Document]): original docs

    Returns:
        list[Document]: enriched docs
    """
    for document in tqdm.tqdm(documents):
        # Generic enrichment, adding modification time and file size
        path = pathlib.Path(document.metadata["source"])
        m_time = datetime.datetime.fromtimestamp(path.stat().st_mtime)
        document.metadata["file_mod_time"] = m_time.strftime("%Y-%m-%d %H:%M:%S")
        document.metadata["file_size"] = str(path.stat().st_size)
        document.metadata["file_sha1"] = hashlib.sha1(path.read_bytes()).hexdigest()
        # Additional application-specific enrichment, if needed
    return documents


def split_docs(
    documents: list[Document],
) -> list[Document]:
    """Splits the documents into chunks, ensuring that all pieces retain the original metadata

    Args:
        documents (list[Document]): original docs

    Returns:
        list[Document]: chunked docs
    """
    text_splitter: TextSplitter = RecursiveCharacterTextSplitter(
        chunk_size=toml_config["splitter"]["chunk_size"],
        chunk_overlap=toml_config["splitter"]["chunk_overlap"],
        add_start_index=True,
    )
    docs: list[Document] = text_splitter.split_documents(documents)
    return docs


def create_embeddings_function(
    provider: EmbeddingsProviders, show_progress: bool
) -> Embeddings:
    """Creates an embeddings generator function

    Args:
        provider (EmbeddingsProviders): one of OPENAI, HUGGINGFACE, INSTRUCTOR
        show_progress (bool): whether to show a progress bar when generating embeddings

    Returns:
        Embeddings: the embeddings generator function
    """
    embeddings: Embeddings
    match provider:
        case EmbeddingsProviders.OPENAI:
            embeddings = OpenAIEmbeddings(
                model="text-embedding-ada-002",
                openai_api_key=toml_config["openai"]["api_key"],
                show_progress_bar=show_progress,
            )

        case EmbeddingsProviders.HUGGINGFACE:
            embeddings = HuggingFaceEmbeddings(
                # model_name="gtr-t5-large",
                # model_name="all-mpnet-base-v2",
                model_name="all-MiniLM-L6-v2",
                model_kwargs={"device": toml_config["general"]["default_device"]},
                encode_kwargs={"show_progress_bar": show_progress},
            )

        case EmbeddingsProviders.INSTRUCTOR:
            embeddings = HuggingFaceInstructEmbeddings(
                # model_name="hkunlp/instructor-base",
                model_name="hkunlp/instructor-large",
                model_kwargs={"device": toml_config["general"]["default_device"]},
                encode_kwargs={"show_progress_bar": show_progress},
            )

    return embeddings


def _fix_vector_db(provider: VectordbProviders) -> None:
    """Takes care of specific initialisation and fixing

    Args:
        provider (VectordbProviders): one of OPENAI, HUGGINGFACE, INSTRUCTOR
    """
    # pylint: disable=import-outside-toplevel
    match provider:
        case VectordbProviders.PGVECTOR:
            import sqlalchemy

            with sqlalchemy.create_engine(
                toml_config["pgvector"]["connection_string"]
            ).begin() as conn:
                conn.execute(sqlalchemy.text("create extension if not exists vector;"))
        case VectordbProviders.PINECONE:
            import pinecone  # type: ignore

            pinecone.init(
                api_key=toml_config["pinecone"]["api_key"],
                environment=toml_config["pinecone"]["environment"],
            )


def _create_mongodb_connection(
    connection_string: str, db_name: str, collection_name: str
) -> MongodbCollection:
    """Creates a MongoDB collection object, needed to create an instance of MongoDBAtlasVectorSearch

    Args:
        connection_string (str): MongoDB Atlas connection string
        db_name (str): database name
        collection_name (str): collection name

    Returns:
        MongodbCollection: collection object, ready to use in the MongoDBAtlasVectorSearch __init__
    """
    # pylint: disable=import-outside-toplevel
    from pymongo import MongoClient

    mongo_client: MongoClient = MongoClient(connection_string)
    collection = mongo_client[db_name][collection_name]
    return collection


def create_vector_db_from_docs(
    provider: VectordbProviders,
    collection_name: str,
    documents: list[Document],
    embed_function: Embeddings,
) -> VectorStore:
    """Creates or rebuilds an embeddings vector store, populated with document and embeddings

    Args:
        provider (VectordbProviders): one of CHROMADB, REDIS, PGVECTOR, PINECONE
        collection_name (str): name given to the document collection
        documents (list[Document]): list of documents to store and embed
        embed_function (Embeddings): embeddings generator function

    Returns:
        VectorStore: populated vector store, with documents and embeddings
    """
    _fix_vector_db(provider=provider)
    vectordb: VectorStore
    match provider:
        case VectordbProviders.CHROMADB:
            vectordb = Chroma.from_documents(
                documents=documents,
                embedding=embed_function,
                persist_directory=toml_config["chromadb"]["persistence_root_dir"],
                collection_metadata={"hnsw:space": "cosine"},
                collection_name=collection_name,
            )

        case VectordbProviders.REDIS:
            vectordb = Redis.from_documents(
                documents=documents,
                embedding=embed_function,
                index_name=collection_name,
                redis_url=toml_config["redis"]["url"],
            )

        case VectordbProviders.PGVECTOR:
            vectordb = PGVector.from_documents(
                documents=documents,
                embedding=embed_function,
                collection_name=collection_name,
                connection_string=toml_config["pgvector"]["connection_string"],
            )

        case VectordbProviders.PINECONE:
            vectordb = Pinecone.from_documents(
                documents=documents,
                embedding=embed_function,
                index_name=collection_name,
            )

        case VectordbProviders.MONGODB_ATLAS:
            vectordb = MongoDBAtlasVectorSearch.from_documents(
                documents=documents,
                embedding=embed_function,
                collection=_create_mongodb_connection(
                    toml_config["mongodb_atlas"]["connection_string"],
                    toml_config["mongodb_atlas"]["db_name"],
                    collection_name,
                ),
            )

    return vectordb


def open_vector_db_for_querying(
    provider: VectordbProviders, collection_name: str, embed_function: Embeddings
) -> VectorStore:
    """Opens an existing embeddings vector store

    Args:
        provider (VectordbProviders): one of CHROMADB, REDIS, PGVECTOR, PINECONE
        collection_name (str): name of the document collection, same as when it was created
        embed_function (Embeddings): embeddings generator function

    Returns:
        VectorStore: the vector store
    """
    _fix_vector_db(provider=provider)
    vectordb: VectorStore
    match provider:
        case VectordbProviders.CHROMADB:
            vectordb = Chroma(
                embedding_function=embed_function,
                persist_directory=toml_config["chromadb"]["persistence_root_dir"],
                collection_metadata={"hnsw:space": "cosine"},
                collection_name=collection_name,
            )

        case VectordbProviders.REDIS:
            vectordb = Redis(
                embedding=embed_function,
                index_name=collection_name,
                redis_url=toml_config["redis"]["url"],
            )

        case VectordbProviders.PGVECTOR:
            vectordb = PGVector(
                embedding_function=embed_function,
                collection_name=collection_name,
                connection_string=toml_config["pgvector"]["connection_string"],
            )

        case VectordbProviders.PINECONE:
            vectordb = Pinecone.from_existing_index(
                index_name=collection_name,
                embedding=embed_function,
                text_key="text",
            )

        case VectordbProviders.MONGODB_ATLAS:
            vectordb = MongoDBAtlasVectorSearch(
                embedding=embed_function,
                collection=_create_mongodb_connection(
                    toml_config["mongodb_atlas"]["connection_string"],
                    toml_config["mongodb_atlas"]["db_name"],
                    collection_name,
                ),
            )

    return vectordb


def output_relevant_docs(
    vector_db: VectorStore, query_str: str, k: int
) -> list[tuple[Document, float]]:
    """Runs a similarity search in a vector store

    Args:
        vector_db (VectorStore): vector store to query into
        query_str (str): query string
        k (int): number of results to return

    Returns:
        list[tuple[Document, float]]: list of matches of the form [matching_doc, matching_score]
    """
    return vector_db.similarity_search_with_score(query_str, k)


def run_qa_chain(
    llm: ChatOpenAI, vector_db: VectorStore, query_str: str, k: int
) -> Any:
    """Runs a basic QA chain using lanchain and openai, with a default prompt

    Args:
        llm (ChatOpenAI): an OpenAI LLM instance
        vector_db (VectorStore): vector store to query into
        query_str (str): query string
        k (int): number of results to return

    Returns:
        a verbose response, showing the chain steps
    """
    return load_qa_chain(
        llm,
        chain_type="stuff",
        verbose=True,
    ).run(
        input_documents=vector_db.similarity_search(query_str, k=k), question=query_str
    )


def run_custom_retrieval_chain(
    llm: ChatOpenAI, vector_db: VectorStore, query_str: str
) -> Any:
    """Runs a custom QA chain using lanchain and openai, allowing us to tweak parameters

    Args:
        llm (ChatOpenAI): an OpenAI LLM instance
        vector_db (VectorStore): vector store to query into
        query_str (str): query string

    Returns:
        a response that is heavily dependent on the prompt and search_kwargs
    """
    return RetrievalQA.from_chain_type(
        llm,
        chain_type="stuff",
        # verbose=True,
        retriever=vector_db.as_retriever(
            # search_type="similarity",
            search_kwargs={
                "k": 6,
                # "score_threshold": 0.2,
                "fetch_k": 50,
                "lambda_mult": 0.25,
            },
        ),
        chain_type_kwargs={
            "prompt": PromptTemplate(
                template=toml_config["general"]["prompt_template"],
                input_variables=["context", "question"],
            )
        },
    ).run(query_str)


if __name__ == "__main__":
    REBUILD = True  #  Rebuild doc embeddings?
    vectordb_provider = VectordbProviders.MONGODB_ATLAS
    embeddings_provider = EmbeddingsProviders.INSTRUCTOR

    if REBUILD:
        create_vector_db_from_docs(
            provider=vectordb_provider,
            collection_name=toml_config["general"]["collection_name"],
            documents=split_docs(
                enrich_source_documents(
                    load_docs(
                        toml_config["docs"]["source_dirs"], toml_config["docs"]["glob"]
                    )
                )
            ),
            embed_function=create_embeddings_function(
                provider=embeddings_provider,
                show_progress=True,
            ),
        )

    QUERY_STR = "Shall I open an email with an attachment I just received from an unknown sender?"
    query_db: VectorStore = open_vector_db_for_querying(
        provider=vectordb_provider,
        collection_name=toml_config["general"]["collection_name"],
        embed_function=create_embeddings_function(
            provider=embeddings_provider, show_progress=False
        ),
    )
    openai_model = ChatOpenAI(
        model=toml_config["openai"]["chat_model"],
        openai_api_key=toml_config["openai"]["api_key"],
        temperature=toml_config["openai"]["temperature"],
    )

    # Return the top k relevant docs
    # matches = output_relevant_docs(query_db, QUERY_STR, 3)
    # pprint(matches)
    # print("-" * 70)

    # Answer the question using default langchain prompt
    # answer = run_qa_chain(openai_model, query_db, QUERY_STR, 3)
    # print(answer)
    # print("-" * 70)

    # Customise the prompt and retrieval parameters
    answer = run_custom_retrieval_chain(openai_model, query_db, QUERY_STR)
    print(answer)
    print("-" * 70)
