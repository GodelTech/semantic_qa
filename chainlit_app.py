"""
Implements a basic Web UI for our Q&A bot using chainlit

Run the following on the command line:
  > chainlit run ./chainlit_app.py -w

"""


from chainlit import (
    on_chat_start,
    on_message,
    user_session,
    Message,
)
from langchain.chat_models import ChatOpenAI
from semantic_qa import (
    toml_config,
    VectordbProviders,
    EmbeddingsProviders,
    open_vector_db_for_querying,
    create_embeddings_function,
    run_custom_retrieval_chain,
)


@on_chat_start
async def chat_start() -> None:
    """This method runs when a user opens a new chat session"""

    query_db = open_vector_db_for_querying(
        provider=VectordbProviders.MONGODB_ATLAS,
        collection_name=toml_config["general"]["collection_name"],
        embed_function=create_embeddings_function(
            provider=EmbeddingsProviders.OPENAI, show_progress=False
        ),
    )
    openai = ChatOpenAI(
        model=toml_config["openai"]["chat_model"],
        openai_api_key=toml_config["openai"]["api_key"],
        temperature=toml_config["openai"]["temperature"],
    )
    user_session.set("query_db", query_db)
    user_session.set("openai_model", openai)
    await Message("Welcome to our chat. How can I help you?").send()


@on_message
async def message_received(message_content: str, _message_id: str) -> None:
    """This method runs every time a user sends a chat message

    Args:
        message_content (str): message content received from the user
        _message_id (str): message id
    """
    answer = run_custom_retrieval_chain(
        user_session.get("openai_model"),
        user_session.get("query_db"),
        message_content,
    )
    await Message(content=answer).send()
