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
from semantic_qa import (
    toml_config,
    VectordbProviders,
    EmbeddingsProviders,
    ChatModelProviders,
    open_vector_db_for_querying,
    create_embeddings_function,
    run_custom_retrieval_chain,
    create_chat_model,
)


@on_chat_start
async def chat_start() -> None:
    """This method runs when a user opens a new chat session"""

    query_db = open_vector_db_for_querying(
        provider=VectordbProviders(toml_config["general"]["vectordb_provider"]),
        collection_name=toml_config["general"]["collection_name"],
        embed_function=create_embeddings_function(
            provider=EmbeddingsProviders(toml_config["general"]["embeddings_provider"]),
            show_progress=False,
        ),
    )
    model = create_chat_model(
        ChatModelProviders(toml_config["general"]["chat_model_provider"])
    )
    user_session.set("query_db", query_db)
    user_session.set("chat_model", model)
    await Message("Welcome to our chat. How can I help you?").send()


@on_message
async def message_received(message_content: str, _message_id: str) -> None:
    """This method runs every time a user sends a chat message

    Args:
        message_content (str): message content received from the user
        _message_id (str): message id
    """
    answer = run_custom_retrieval_chain(
        user_session.get("chat_model"),
        user_session.get("query_db"),
        message_content,
    )
    await Message(content=answer).send()
