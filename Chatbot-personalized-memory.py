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

# Implement Retrieval-Augmented Generation (RAG) Chatbot
from langchain.llms import OpenAI

# Initialize OpenAI LLM
llm = OpenAI(model_name="gpt-4")

def retrieve_memory(user_id, query):
    """Retrieves past conversation history and generates context-aware responses"""
    query_embedding = embedding_model.embed_query(query)

    # Search for relevant memory in vector database
    results = collection.query(query_embeddings=[query_embedding], n_results=3)

    memory_context = ""
    for item in results["metadatas"]:
        memory_context += item["text"] + "\n"

    # Combine memory with the new query
    prompt = f"User's Past Context: {memory_context}\nUser Query: {query}\nChatbot Response: "
    response = llm(prompt)

    # Store the new interaction
    store_message(user_id, query, response)

    return response
# This function retrieves past conversation history from the vector database, generates context-aware responses using the retrieved memory, and stores the new interaction in memory.
# The function takes the user ID and user query as input, embeds the query, searches for relevant memory in the vector database, and combines the memory with the new query.
# The combined context is used as a prompt to generate a response from the OpenAI language model.
# The function then stores the user query and generated response in the vector database and returns the response to be sent back to the user.

# Build the Chat Interface
def chat():
    print("\n🤖 Chatbot with Memory (Type 'exit' to quit)")
    user_id = "user_123"  # Assign user ID for personalization

    while True:
        query = input("You: ")
        if query.lower() == "exit":
            print("Goodbye!")
            break
        
        response = retrieve_memory(user_id, query)
        print(f"Chatbot: {response}")

# Run chatbot
chat()
# This chat interface allows users to interact with the chatbot, which retrieves past conversation history and generates context-aware responses using the RAG approach.
# The chatbot stores the conversation history in the vector database to personalize responses based on the user's past interactions.
# Users can type "exit" to end the conversation with the chatbot.
# The chatbot responds to user queries with context-aware responses based on the retrieved memory and the user's current query.

# Export and Visualize Memory Data
import pandas as pd

def export_memory():
    """Exports stored chat history from vector database"""
    results = collection.get()
    chat_history = [{"text": item["text"]} for item in results["metadatas"]]

    df = pd.DataFrame(chat_history)
    df.to_csv("chat_memory.csv", index=False)

    print("📁 Chat memory exported to chat_memory.csv")

# Example usage
export_memory()
# This function exports the stored chat history from the vector database to a CSV file for visualization and analysis.
# The exported data includes the text data associated with each stored conversation.
# The exported CSV file is saved as "chat_memory.csv" in the current directory.
