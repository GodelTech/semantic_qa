# How to use MongoDB Atlas as vector store

1. Create a new account at https://www.mongodb.com/atlas/database ("Try Free") or log on to an existing one.

1. Create a new Cluster. You can choose a free of charge "Shared" configuration at a cloud location near you.

1. Set up an authentication mechanism. Username and password is probably the simplest, and the website will generate a strong pwd for you. Make sure to write it down for later.

1. Make sure you add your IP address to the access list to the new cluster.

1. Once the cluster is provisioned, create a new database and collection with suitable names, and write both to the TOML config file

1. Obtain the connection string which will look like the one below, and save it to secrets.toml

    ```
    mongodb+srv://username:password@cluster.xxxxxxx.mongodb.net/?retryWrites=true&w=majority
    ```

1. Create a new "Search" index, using the JSON editor, for the database and collection you created earlier, and the following JSON content (replace XXX with the dimension number of your embeddings model):

    ```
    {
      "mappings": {
        "dynamic": true,
        "fields": {
          "embedding": {
            "dimensions": XXX,
            "similarity": "cosine",
            "type": "knnVector"
          }
        }
      }
    }
    ```
