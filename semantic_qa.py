"""Implements basic semantic Q&A bot on a document collection"""

import sys
import os
from enum import Enum
import datetime
import pathlib
import hashlib
from typing import cast, Any, Optional
from pprint import pprint
import argparse
import numpy as np
import joblib  # type: ignore
import tqdm

from langchain.schema import Document

from langchain.text_splitter import (
    TextSplitter,
    RecursiveCharacterTextSplitter,
)

from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore
from langchain.base_language import BaseLanguageModel
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.retrieval_qa.base import RetrievalQA

from langchain_community.embeddings.huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceInstructEmbeddings,
)
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.vectorstores.redis import Redis
from langchain_community.vectorstores.myscale import MyScale, MyScaleSettings
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores.milvus import Milvus
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_community.vectorstores.azure_cosmos_db import AzureCosmosDBVectorSearch
from langchain_community.llms.replicate import Replicate
from langchain_community.document_loaders.directory import DirectoryLoader

from langchain_postgres import PGVector
from langchain_elasticsearch import ElasticsearchStore
from langchain_pinecone import PineconeVectorStore
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_fireworks import ChatFireworks
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from pydantic.v1.utils import deep_update

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
    MONGODB = "mongodb"
    ELASTICSEARCH = "elasticsearch"
    NEO4J = "neo4j"
    MYSCALE = "myscale"
    FAISS = "faiss"
    MILVUS = "milvus"


class EmbeddingsProviders(Enum):
    """Types of embeddings providers we currently support"""

    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    INSTRUCTOR = "instructor"
    OLLAMA = "ollama"


class ChatModelProviders(Enum):
    """Chat model providers we currently support"""

    OPENAI = "openai"
    FIREWORKS = "fireworks"
    ANTHROPIC = "anthropic"
    LMSTUDIO = "lmstudio"
    REPLICATE = "replicate"


MONGO_IMPLEMENTATION_CLASSES: dict[str, Any] = {
    "atlas": MongoDBAtlasVectorSearch,
    "cosmosdb": AzureCosmosDBVectorSearch,
}


def read_config() -> dict:
    """Reads the TOML config files and builds the merged config dict by reading config.toml
    (Git version controlled), then merging secrets.toml (non-version controlled) on top

    Returns:
        dict: merged config dictionary
    """
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)
    with open("secrets.toml", "rb") as g:
        secrets = tomllib.load(g)
    return deep_update(config, secrets)


toml_config: dict = read_config()


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
            # debug settings
            sample_size=toml_config["debug"]["corpus_sample_size"],
            randomize_sample=toml_config["debug"]["corpus_randomize_sample"],
            sample_seed=toml_config["debug"]["corpus_sample_seed"],
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
        # Generic enrichment
        path = pathlib.Path(document.metadata["source"])
        m_time = datetime.datetime.fromtimestamp(path.stat().st_mtime)
        document.metadata["file_mod_time"] = m_time.strftime("%Y-%m-%d %H:%M:%S")
        document.metadata["file_size"] = str(path.stat().st_size)
        document.metadata["file_sha1"] = hashlib.sha1(path.read_bytes()).hexdigest()
        document.metadata["file_name"] = path.stem
        document.metadata["file_ext"] = path.suffix
        # Additional application-specific enrichment, if needed
    return documents


def enrich_chunked_documents(documents: list[Document]) -> list[Document]:
    """Iterates over documents, enriching metadata and content with useful information

    Args:
        documents (list[Document]): chunked docs

    Returns:
        list[Document]: enriched, chunked docs
    """
    for document in tqdm.tqdm(documents):
        # Generic enrichment
        document.metadata["chunk_length"] = len(document.page_content)
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
        provider (EmbeddingsProviders): one of OPENAI, HUGGINGFACE, INSTRUCTOR, OLLAMA
        show_progress (bool): whether to show a progress bar when generating embeddings

    Returns:
        Embeddings: the embeddings generator function
    """
    embeddings: Embeddings
    match provider:
        case EmbeddingsProviders.OPENAI:
            os.environ["OPENAI_API_KEY"] = toml_config["openai"]["api_key"]
            embeddings = OpenAIEmbeddings(
                # model="text-embedding-ada-002",  # Dimensions = 1536
                model="text-embedding-3-small",  # Dimensions = 1536
                # model="text-embedding-3-large",  # Dimensions = 3072
                show_progress_bar=show_progress,
                timeout=toml_config["general"]["llm_request_timeout"],
            )

        case EmbeddingsProviders.HUGGINGFACE:
            embeddings = HuggingFaceEmbeddings(
                # model_name="gtr-t5-large",  # Dimensions = 768
                # model_name="all-mpnet-base-v2",  # Dimensions = 768
                # model_name="all-MiniLM-L12-v2",  # Dimensions = 384
                model_name="all-MiniLM-L6-v2",  # Dimensions = 384
                model_kwargs={"device": toml_config["general"]["default_device"]},
                show_progress=show_progress,
            )

        case EmbeddingsProviders.INSTRUCTOR:
            embeddings = HuggingFaceInstructEmbeddings(
                # model_name="hkunlp/instructor-base",  # Dimensions = 768
                model_name="hkunlp/instructor-large",  # Dimensions = 768
                # model_name="hkunlp/instructor-xl",  # Dimensions = 768
                model_kwargs={"device": toml_config["general"]["default_device"]},
                encode_kwargs={"show_progress_bar": show_progress},
            )

        case EmbeddingsProviders.OLLAMA:
            embeddings = OllamaEmbeddings(
                base_url="http://localhost:11434",
                # model="mistral",  # Dimensions = 4096
                # model="llama2",  # Dimensions = 4096
                # model="orca-mini",  # Dimensions = 3200
                # model="phi3",  # Dimensions = 3072
                model="llama3",  # Dimensions = 4096
                show_progress=show_progress,
            )

    return embeddings


def _fix_chat_model(provider: ChatModelProviders) -> None:
    """Takes care of specific initialisation and fixing

    Args:
        provider (ChatModelProviders): One of OPENAI, FIREWORKS, ANTHROPIC, LMSTUDIO, REPLICATE
    """
    if provider in [
        ChatModelProviders.OPENAI,
        ChatModelProviders.FIREWORKS,
        ChatModelProviders.ANTHROPIC,
    ]:
        os.environ[f"{provider.value.upper()}_API_KEY"] = toml_config[provider.value][
            "api_key"
        ]
    if provider in [
        ChatModelProviders.REPLICATE,
    ]:
        os.environ[f"{provider.value.upper()}_API_TOKEN"] = toml_config[provider.value][
            "api_key"
        ]


def _fix_vector_db(provider: VectordbProviders) -> Optional[Any]:
    """Takes care of specific initialisation and fixing

    Args:
        provider (VectordbProviders): One of OPENAI, HUGGINGFACE, INSTRUCTOR, MONGODB,
            ELASTICSEARCH, NEO4J, MYSCALE, FAISS, MILVUS

    Returns:
        Optional[Any]: in most cases None, but MongoDBAtlasVectorSearch needs a MongoClient
    """
    # pylint: disable=import-outside-toplevel
    ret = None
    match provider:
        case VectordbProviders.PGVECTOR:
            import sqlalchemy

            with sqlalchemy.create_engine(
                toml_config["pgvector"]["connection_string"]
            ).begin() as conn:
                conn.execute(sqlalchemy.text("create extension if not exists vector;"))

        case VectordbProviders.PINECONE:
            os.environ["PINECONE_API_KEY"] = toml_config["pinecone"]["api_key"]

        case VectordbProviders.MONGODB:
            from pymongo import MongoClient

            ret = MongoClient(toml_config["mongodb"]["connection_string"])

    return ret


def create_vector_db_from_docs(
    provider: VectordbProviders,
    collection_name: str,
    documents: list[Document],
    embed_function: Embeddings,
) -> VectorStore:
    """Creates or rebuilds an embeddings vector store, populated with document and embeddings

    Args:
        provider (VectordbProviders): One of OPENAI, HUGGINGFACE, INSTRUCTOR, MONGODB,
            ELASTICSEARCH, NEO4J, MYSCALE, FAISS, MILVUS
        collection_name (str): name given to the document collection
        documents (list[Document]): list of documents to store and embed
        embed_function (Embeddings): embeddings generator function

    Returns:
        VectorStore: populated vector store, with documents and embeddings
    """
    _fix = _fix_vector_db(provider=provider)
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
                connection=toml_config["pgvector"]["connection_string"],
                use_jsonb=True,
            )

        case VectordbProviders.PINECONE:
            vectordb = PineconeVectorStore.from_documents(
                documents=documents,
                embedding=embed_function,
                index_name=collection_name,
            )

        case VectordbProviders.MONGODB:
            from pymongo import MongoClient  # pylint: disable=import-outside-toplevel

            vectordb = MONGO_IMPLEMENTATION_CLASSES[
                toml_config["mongodb"]["implementation"]
            ].from_documents(
                documents=documents,
                embedding=embed_function,
                collection=cast(MongoClient, _fix)[toml_config["mongodb"]["db_name"]][
                    collection_name
                ],
            )
            if (
                isinstance(vectordb, AzureCosmosDBVectorSearch)
                and not vectordb.index_exists()
            ):
                vectordb.create_index(dimensions=len(embed_function.embed_query("foo")))

        case VectordbProviders.ELASTICSEARCH:
            vectordb = ElasticsearchStore.from_documents(
                documents=documents,
                embedding=embed_function,
                index_name=collection_name,
                es_url=toml_config["elasticsearch"]["url"],
                distance_strategy="COSINE",
            )

        case VectordbProviders.NEO4J:
            vectordb = Neo4jVector.from_documents(
                documents=documents,
                embedding=embed_function,
                url=toml_config["neo4j"]["url"],
                username=toml_config["neo4j"]["username"],
                password=toml_config["neo4j"]["password"],
            )

        case VectordbProviders.MYSCALE:
            vectordb = MyScale.from_documents(
                documents=documents,
                embedding=embed_function,
                config=MyScaleSettings(
                    host=toml_config["myscale"]["host"],
                    port=toml_config["myscale"]["port"],
                    username=toml_config["myscale"]["username"],
                    password=toml_config["myscale"]["password"],
                ),
            )

        case VectordbProviders.FAISS:
            vectordb = FAISS.from_documents(
                documents=documents,
                embedding=embed_function,
                distance_strategy=DistanceStrategy.COSINE,
            )
            vectordb.save_local(
                folder_path=toml_config["faiss"]["persistence_root_dir"],
                index_name=collection_name,
            )

        case VectordbProviders.MILVUS:
            vectordb = Milvus.from_documents(
                documents=documents,
                embedding=embed_function,
                collection_name=collection_name,
                index_params={
                    "metric_type": DistanceStrategy.COSINE.value,
                    "index_type": toml_config["milvus"]["index_type"],
                },
                connection_args={
                    "uri": toml_config["milvus"]["uri"],
                    "user": toml_config["milvus"]["user"],
                    "password": toml_config["milvus"]["password"],
                },
            )

    return vectordb


def open_vector_db_for_querying(
    provider: VectordbProviders, collection_name: str, embed_function: Embeddings
) -> VectorStore:
    """Opens an existing embeddings vector store

    Args:
        provider (VectordbProviders): One of OPENAI, HUGGINGFACE, INSTRUCTOR, MONGODB,
            ELASTICSEARCH, NEO4J, MYSCALE, FAISS, MILVUS
        collection_name (str): name of the document collection, same as when it was created
        embed_function (Embeddings): embeddings generator function

    Returns:
        VectorStore: the vector store
    """
    _fix = _fix_vector_db(provider=provider)
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
                embeddings=embed_function,
                collection_name=collection_name,
                connection=toml_config["pgvector"]["connection_string"],
                use_jsonb=True,
            )

        case VectordbProviders.PINECONE:
            vectordb = PineconeVectorStore.from_existing_index(
                index_name=collection_name,
                embedding=embed_function,
                text_key="text",
            )

        case VectordbProviders.MONGODB:
            from pymongo import MongoClient  # pylint: disable=import-outside-toplevel

            vectordb = MONGO_IMPLEMENTATION_CLASSES[
                toml_config["mongodb"]["implementation"]
            ](
                embedding=embed_function,
                collection=cast(MongoClient, _fix)[toml_config["mongodb"]["db_name"]][
                    collection_name
                ],
            )

        case VectordbProviders.ELASTICSEARCH:
            vectordb = ElasticsearchStore(
                embedding=embed_function,
                index_name=collection_name,
                es_url=toml_config["elasticsearch"]["url"],
            )

        case VectordbProviders.NEO4J:
            vectordb = Neo4jVector(
                embedding=embed_function,
                url=toml_config["neo4j"]["url"],
                username=toml_config["neo4j"]["username"],
                password=toml_config["neo4j"]["password"],
            )

        case VectordbProviders.MYSCALE:
            vectordb = MyScale(
                embedding=embed_function,
                config=MyScaleSettings(
                    host=toml_config["myscale"]["host"],
                    port=toml_config["myscale"]["port"],
                    username=toml_config["myscale"]["username"],
                    password=toml_config["myscale"]["password"],
                ),
            )

        case VectordbProviders.FAISS:
            vectordb = FAISS.load_local(
                embeddings=embed_function,
                folder_path=toml_config["faiss"]["persistence_root_dir"],
                index_name=collection_name,
                allow_dangerous_deserialization=True,
                distance_strategy=DistanceStrategy.COSINE,
            )

        case VectordbProviders.MILVUS:
            vectordb = Milvus(
                embedding_function=embed_function,
                collection_name=collection_name,
                index_params={
                    "metric_type": DistanceStrategy.COSINE.value,
                    "index_type": toml_config["milvus"]["index_type"],
                },
                connection_args={
                    "uri": toml_config["milvus"]["uri"],
                    "user": toml_config["milvus"]["user"],
                    "password": toml_config["milvus"]["password"],
                },
            )

    return vectordb


def create_chat_model(provider: ChatModelProviders) -> BaseLanguageModel:
    """Creates the chat model to elaborate the response based on the search results

    Args:
        provider (ChatModelProviders): One of OPENAI, FIREWORKS, ANTHROPIC, LMSTUDIO, REPLICATE

    Returns:
        BaseChatModel: The chat model
    """
    _fix_chat_model(provider=provider)
    model: BaseLanguageModel
    match provider:
        case ChatModelProviders.OPENAI:
            model = ChatOpenAI(
                model=toml_config["openai"]["chat_model"],
                temperature=toml_config["openai"]["temperature"],
                timeout=toml_config["general"]["llm_request_timeout"],
            )

        case ChatModelProviders.FIREWORKS:
            model = ChatFireworks(
                model=toml_config["fireworks"]["chat_model"],
                temperature=toml_config["fireworks"]["temperature"],
                max_tokens=toml_config["fireworks"]["max_tokens"],
                timeout=toml_config["general"]["llm_request_timeout"],
            )

        case ChatModelProviders.ANTHROPIC:
            model = ChatAnthropic(  # type: ignore[call-arg]
                model_name=toml_config["anthropic"]["chat_model"],
                temperature=toml_config["anthropic"]["temperature"],
                timeout=toml_config["general"]["llm_request_timeout"],
            )

        case ChatModelProviders.LMSTUDIO:
            model = ChatOpenAI(
                base_url=toml_config["lmstudio"]["base_url"],
                api_key=toml_config["lmstudio"]["api_key"],
                model=toml_config["lmstudio"]["chat_model"],
                temperature=toml_config["lmstudio"]["temperature"],
                timeout=toml_config["general"]["llm_request_timeout"],
            )

        case ChatModelProviders.REPLICATE:
            replicate_model_kwargs = dict(toml_config["replicate"])
            _ = replicate_model_kwargs.pop("api_key")
            model = Replicate(
                model=replicate_model_kwargs.pop("chat_model"),
                input=replicate_model_kwargs,
            )

    return model


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
    try:
        return vector_db.similarity_search_with_score(query_str, k=k)
    except NotImplementedError:
        return vector_db.similarity_search_with_relevance_scores(query_str, k=k)


def run_qa_chain(
    llm: BaseLanguageModel, vector_db: VectorStore, query_str: str, k: int
) -> Any:
    """Runs a basic QA chain using lanchain and openai, with a default prompt

    Args:
        llm (BaseChatModel): an LLM instance
        vector_db (VectorStore): vector store to query into
        query_str (str): query string
        k (int): number of results to return

    Returns:
        a verbose response, showing the chain steps
    """
    return (
        load_qa_chain(
            llm,
            chain_type="stuff",
            verbose=True,
        )
        .invoke(
            {
                "input_documents": vector_db.similarity_search(query_str, k=k),
                "question": query_str,
            }
        )
        .get("output_text", "")
    )


def run_custom_retrieval_chain(
    llm: BaseLanguageModel, vector_db: VectorStore, query_str: str
) -> Any:
    """Runs a custom QA chain using lanchain and openai, allowing us to tweak parameters

    Args:
        llm (BaseChatModel): an LLM instance
        vector_db (VectorStore): vector store to query into
        query_str (str): query string

    Returns:
        a response that is heavily dependent on the prompt and search_kwargs
    """
    search_params = dict(toml_config["search_params"])
    return (
        RetrievalQA.from_chain_type(
            llm,
            chain_type="stuff",
            # verbose=True,
            retriever=vector_db.as_retriever(
                search_type=search_params.pop("algorithm"),
                search_kwargs=search_params,
            ),
            chain_type_kwargs={
                "prompt": PromptTemplate(
                    template=toml_config["chat"]["prompt_template"],
                    input_variables=["context", "question"],
                )
            },
        )
        .invoke({"query": query_str})
        .get("result", "")
    )


def get_cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculates the cosine similarity between two vectors:
        cos(theta) = dot_product(a, b) / (norm(a) * norm(b))

    Args:
        a (list[float]): vector a
        b (list[float]): vector b

    Returns:
        float: cosine of the "hyper-angle" between the vectors
    """
    array_a = np.array(a)
    array_b = np.array(b)
    return float(
        np.dot(array_a, array_b) / (np.linalg.norm(array_a) * np.linalg.norm(array_b))
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the vector DB from the source documents",
    )
    args = parser.parse_args()

    vectordb_provider = VectordbProviders(toml_config["general"]["vectordb_provider"])
    embeddings_provider = EmbeddingsProviders(
        toml_config["general"]["embeddings_provider"]
    )
    chat_model_provider = ChatModelProviders(
        toml_config["general"]["chat_model_provider"]
    )

    if toml_config["debug"]["force_rebuild"] or args.rebuild:
        chunked_documents: list[Document] = []
        if toml_config["debug"]["persist_chunked_docs"]:
            try:
                chunked_documents = joblib.load("chunked_docs.pkl")
            except Exception:  # pylint: disable=broad-exception-caught
                chunked_documents = []

        if not chunked_documents:
            chunked_documents = enrich_chunked_documents(
                split_docs(
                    enrich_source_documents(
                        load_docs(
                            toml_config["docs"]["source_dirs"],
                            toml_config["docs"]["glob"],
                        )
                    )
                )
            )
            if toml_config["debug"]["persist_chunked_docs"]:
                joblib.dump(chunked_documents, "chunked_docs.pkl", compress=True)

        create_vector_db_from_docs(
            provider=vectordb_provider,
            collection_name=toml_config["docs"]["collection_name"],
            documents=chunked_documents,
            embed_function=create_embeddings_function(
                provider=embeddings_provider,
                show_progress=True,
            ),
        )

    QUERY_STR = "Shall I open an email with an attachment I just received from an unknown sender?"
    QUERY_ALT = "I just got an email from someone I don't know containing a file. Should I open it?"
    QUERY_ESP = (
        "¿Debo abrir un correo electrónico con un archivo adjunto "
        "que acabo de recibir de un remitente desconocido?"
    )

    embed_creator = create_embeddings_function(
        provider=embeddings_provider, show_progress=False
    )

    query_db: VectorStore = open_vector_db_for_querying(
        provider=vectordb_provider,
        collection_name=toml_config["docs"]["collection_name"],
        embed_function=embed_creator,
    )

    chat_model = create_chat_model(chat_model_provider)

    embeddings_english = embed_creator.embed_query(QUERY_STR)
    embeddings_spanish = embed_creator.embed_query(QUERY_ESP)
    embeddings_alternative = embed_creator.embed_query(QUERY_ALT)

    # Print out the QUERY_STR embeddings, truncating their decimal digits
    print(f"Embeddings for query '{QUERY_STR}':")
    pprint(
        [f"{embed:+.8f}" for embed in embeddings_english],
        compact=True,
        width=os.get_terminal_size().columns,
    )
    print(f"Dimensions of embeddings: {len(embeddings_english)}")
    print("-" * 70)

    # Show the semantic similarity between the sentences,
    # calculated as the cosine distance between their embeddings
    similarity = get_cosine_similarity(embeddings_english, embeddings_alternative)
    print(f"Query 1 '{QUERY_STR}'")
    print(f"Query 2 '{QUERY_ALT}'")
    print(f"similarity: {similarity}")
    print("-" * 70)
    similarity = get_cosine_similarity(embeddings_english, embeddings_spanish)
    print(f"Query 1 '{QUERY_STR}'")
    print(f"Query 2 '{QUERY_ESP}'")
    print(f"similarity: {similarity}")
    print("-" * 70)

    # Return the top k relevant docs
    print(f"Most relevant documents and similarity scores for query '{QUERY_STR}':")
    matches = output_relevant_docs(query_db, QUERY_STR, 3)
    pprint(matches)
    print("-" * 70)

    # Answer the question using default langchain prompt
    print("Answer to query using default chat prompt:")
    answer = run_qa_chain(chat_model, query_db, QUERY_STR, 3)
    print(answer)
    print("-" * 70)

    # Customise the prompt and retrieval parameters
    print("Answer with custom prompt and similarity search parameters in config.toml:")
    answer = run_custom_retrieval_chain(chat_model, query_db, QUERY_STR)
    print(answer)
    print("-" * 70)
