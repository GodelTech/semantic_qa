# How to use redis-stack as vector store

1. Create redis-stack container with persistent volume for redis data 

```sh
docker volume create redis-stack-data
docker run --name redis-stack -p 6379:6379 -p 8001:8001 -v redis-stack-data:/data -d --restart=unless-stopped redis/redis-stack:latest
``` 
