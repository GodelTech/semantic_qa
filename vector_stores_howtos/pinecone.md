# How to use pinecone as vector store

1. Create an account on Pinecone (https://pinecone.io/), or log on to an existing account

1. Create a new project and give it a suitable name. The "Cloud provider" and "Environment" options might be limited depending on whether you are on the Free Tier

1. Make a note of the API key and environment name for your new project, which you will need to store in the config file

1. Within the new project, create a new index with a suitable name (same as "collection_name" in the config file), same dimension number as one generated by embeddings model, and "cosine" metric
