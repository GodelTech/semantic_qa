# How to use pgvector as vector store

1. Create pgvector container with persistent volume for Postgres data

    ```sh
    docker volume create postgresql-data
    docker run --name pgvector -p 5432:5432 -e POSTGRES_PASSWORD=mysecretpassword -v postgresql-data:/var/lib/postgresql/data -d --restart=unless-stopped ankane/pgvector:latest
    ``` 

1. Connect to postgres db as postgres:mysecretpassword and run:

    ```sql
    create database vectors;
    create user vector_user with encrypted password 'vector_pass';
    grant all privileges on database vectors to vector_user;
    alter user vector_user with superuser;
    ```

1. Connect to new database vectors as vector_user:vector_pass and run:

    ```sql
    create extension if not exists vector;
    ```

1. Use this connection string
 
    postgresql+psycopg2://vector_user:vector_pass@localhost:5432/vectors
