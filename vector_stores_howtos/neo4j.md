# How to use Neo4j as vector store

1. Create redis-stack container with persistent volume for redis data 

```sh
docker volume create neo4j-data
docker run --name neo4j -p7474:7474 -p7687:7687 -v neo4j-data:/data --env NEO4J_AUTH=neo4j/my_secret_pwd -d --restart=unless-stopped neo4j:5-community
``` 
