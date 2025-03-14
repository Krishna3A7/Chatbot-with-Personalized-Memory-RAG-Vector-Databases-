# Initialize Vector Database (ChromaDB or FAISS)
import chromadb
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Initialize embedding model
embedding_model = OpenAIEmbeddings()

# Initialize ChromaDB (or FAISS for local storage)
vector_db = chromadb.PersistentClient(path="./chatbot_memory")
collection = vector_db.get_or_create_collection("chat_memory")

print("Vector database initialized.")
# This sets up a vector database for storing conversation embeddings. The embeddings are stored in a collection called "chat_memory".

# Create a Function to Store Chat History
import json

def store_message(user_id, message, response):
    """Stores the user message and response in the vector database"""
    data = f"User: {message} | Chatbot: {response}"
    embedding = embedding_model.embed_query(data)  # Convert to vector

    # Store in vector database
    collection.add(ids=[user_id], embeddings=[embedding], metadatas=[{"text": data}])

    print("Message stored in memory.")
# This function takes the user ID, user message, and chatbot response as input, combines them into a single string, converts the string to a vector embedding, and stores the embedding in the vector database.
# The metadata associated with the embedding includes the text data for reference.
# The function also prints a message to indicate that the message has been stored in memory.
