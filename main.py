from data.sources import combine_all_data
from embeddings.faiss_index import create_index
from llm.groq_chat import call_groq
from sentence_transformers import SentenceTransformer

# === Step 1: Load data ===
print("📦 Loading data...")
texts, metadata = combine_all_data(
    news_json_path="data/classified_disaster_news.json",
    knowledge_path="data/disaster_knowledge.csv"
)

# === Step 2: Build FAISS index ===
print("🔍 Building FAISS index...")
index, _ = create_index(texts)
model = SentenceTransformer("all-MiniLM-L6-v2")

print("💬 Chatbot ready! Type your question (or 'quit' to exit)\n")

# === Step 3: Chat Loop ===
while True:
    query = input("🧑 You: ").strip()
    if query.lower() in ["quit", "exit"]:
        print("👋 Bye! Stay safe.")
        break

    # Step 4: Embed + Retrieve
    query_vec = model.encode([query])
    D, I = index.search(query_vec, k=5)
    
    chunks = []
    for idx in I[0]:
        source = metadata[idx]
        text = texts[idx]
        url = source.get("url", "")
        if url:
            chunks.append(f"{text}\n(Source: {url})")
        else:
            chunks.append(text)

    context = "\n\n".join(chunks)

    # Step 5: Prompt LLM
    prompt = f"""Use the following information to answer the user's question:\n\n{context}\n\nUser Question: {query}\nAnswer:"""
    try:
        response = call_groq(prompt)
        print(f"🤖 Bot: {response}\n")
    except Exception as e:
        print("❌ Error while querying Groq:", e)
