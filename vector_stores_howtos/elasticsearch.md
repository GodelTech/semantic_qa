# How to use elasticsearch as vector store

1. Create elasticsearch container with persistent data volume, disabling SSL for development:

    ```sh
    docker network create elasticnetwork
    docker volume create elasticsearch-data
    docker run -d --name elasticsearch --net elasticnetwork \
      -v elasticsearch-data:/usr/share/elasticsearch/data   \
      -p 9200:9200 -p 9300:9300                             \
      -e "xpack.security.http.ssl.enabled=false"            \
      -e "http.cors.enabled=true"                           \
      -e "http.cors.allow-origin=/.*/"                      \
      -e "discovery.type=single-node"                       \
      -e "ELASTIC_PASSWORD=my_secret_pwd"                   \
      --restart=unless-stopped elasticsearch:8.10.2
    ```
