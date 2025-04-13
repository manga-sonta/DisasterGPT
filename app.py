import streamlit as st
from data.sources import combine_all_data, load_live_news
from embeddings.faiss_index import create_index
from llm.groq_chat import call_groq
from sentence_transformers import SentenceTransformer
import pandas as pd
from PIL import Image

# === Load and embed all data ===
st.cache_data(show_spinner=False)
def load_and_embed():
    texts, metadata = combine_all_data(
        news_json_path="data/classified_disaster_news.json",
        knowledge_path="data/disaster_knowledge.csv"
    )
    index, _ = create_index(texts)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return texts, metadata, index, model

texts, metadata, index, model = load_and_embed()

# === UI Setup ===
st.set_page_config(page_title="DisasterGPT", layout="wide")
st.title("🚨 DisasterGPT")
st.markdown("<h4 style='font-weight:bold; font-style:italic;'>AI for Disaster Awareness</h4>", unsafe_allow_html=True)

# === Load news data for bulletin panel ===
news_data = load_live_news("data/classified_disaster_news.json", include_uncategorized=False)
df = pd.DataFrame(news_data)
disaster_types = sorted(df["matched_keyword"].dropna().unique())

# === Define color map for disaster types ===
color_map = {
    "flood": "#0074D9",
    "earthquake": "#FF4136",
    "cyclone": "#2ECC40",
    "wildfire": "#FF851B",
    "landslide": "#B10DC9",
    "tsunami": "#39CCCC",
    "storm": "#FFDC00"
}

# === Sidebar Bulletin ===
st.sidebar.image(Image.open("logo.jpg"), use_container_width=True)
st.sidebar.markdown("<h3 style='font-size:22px;'>📰 Bulletin</h3>", unsafe_allow_html=True)
selected_disaster = st.sidebar.selectbox("Select a disaster type:", disaster_types)

filtered = df[df["matched_keyword"] == selected_disaster]
if filtered.empty:
    st.sidebar.warning("No articles found for this category.")
else:
    color = color_map.get(selected_disaster.lower(), "#DDDDDD")
    for _, row in filtered.iterrows():
        preview = row['text'][:80] + "..." if len(row['text']) > 80 else row['text']
        st.sidebar.markdown(
            f"""<div style='border-left: 5px solid {color}; padding-left: 10px; margin-bottom: 1rem; font-size:15px;'>
            <strong>{row['title']}</strong><br>
            <em>{row.get('date', '')}</em><br>
            <p style='margin: 0.2rem 0;'>{preview}</p>
            <a href="{row.get('url', '')}" target="_blank">Read more</a>
            </div><hr style='margin:0.5rem 0;'>""",
            unsafe_allow_html=True
        )

# === Chatbot Column ===
st.subheader("🤖 Chatbot")
# st.image("logo.jpg", width=150)
user_query = st.text_input("Ask a question about disasters:", placeholder="e.g. What should I do during a cyclone?")

if user_query:
    query_vec = model.encode([user_query])
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
    prompt = f"""Use the following information to answer the user's question:\n\n{context}\n\nUser Question: {user_query}\nAnswer:"""

    with st.spinner("Thinking..."):
        try:
            answer = call_groq(prompt)
            st.markdown(f"**🤖 Response:**\n\n{answer}")
        except Exception as e:
            st.error(f"Error: {e}")
