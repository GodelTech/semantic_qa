# semantic_qa
A small semantic Q&amp;A demo using langchain and openai

## Prerequisites

The only hard requirement is Python 3.10+ with pip and virtualenv. A CUDA-compatible GPU is strongly advised to generate text embeddings locally using larger models.

## Dependencies

After cloning this repo, create and activate a Python virtual environment, then install the required Python packages using pip:

```Powershell
PS > virtualenv venv
PS > venv\scripts\activate.ps1
(venv) PS > pip install -r requirements_dev.txt
(venv) PS > pip install -r requirements.txt
```

```sh
$ virtualenv venv
$ source venv/bin/activate
(venv) $ pip install -r requirements_dev.txt
(venv) $ pip install -r requirements.txt
```

If you have a CUDA-capable GPU, install the right torch packages as described at https://pytorch.org/get-started/locally/ 

```
pip install -U --force-reinstall --no-deps torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Document corpus

Choose whatever document folder you fancy. Why not try a local copy of Godel's security policies from Sharepoint? 

## Vector database

The demo currently supports the following vector stores:

 * [ChromaDB](https://www.trychroma.com/)
 * [Pgvector](https://github.com/pgvector/pgvector). [Set-up instructions](vector_stores_howtos/pgvector.md)
 * [Redis Stack](https://redis.io/docs/about/about-stack/). [Set-up instructions](vector_stores_howtos/redis-stack.md)
 * [Pinecone](https://www.pinecone.io/). [Set-up instructions](vector_stores_howtos/pinecone.md)

The first one uses file-based SQLite for storage and does not require any work. The other three need some set up, detailed in the links above.

## Embeddings generator models

The demo currently supports:

 * Calling the [OpenAI API](https://platform.openai.com/docs/api-reference/embeddings), which requires an API key and a payment plan, using the model ["text-embedding-ada-002"](https://openai.com/blog/new-and-improved-embedding-model) by default
 * Generating embeddings locally using torch and a pre-trained model downloaded from [Hugging Face](https://huggingface.co/models). The default model is ["all-MiniLM-L6-v2"](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
 * Generating embeddings locally using torch and one of the pre-trained [Instructor](https://github.com/HKUNLP/instructor-embedding) models. The default used is ["hkunlp/instructor-large"](https://github.com/HKUNLP/instructor-embedding#model-list)


