import json
import pandas as pd
import os
from utils.preprocessing import preprocess_text

# Load classified disaster news with matched_keyword
def load_live_news(path="data/classified_disaster_news.json", include_uncategorized=False):
    if not os.path.exists(path):
        raise FileNotFoundError(f"News file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        news_data = json.load(f)

    entries = []
    for article in news_data:
        if not include_uncategorized and article.get("matched_keyword") == "no_category":
            continue

        entries.append({
            "text": article.get("content") or article.get("title", ""),
            "title": article.get("title", ""),
            "url": article.get("url", ""),  # ✅ ADD THIS LINE
            "author": article.get("author"),
            "date": article.get("date"),
            "matched_keyword": article.get("matched_keyword", ""),
            "similarity_score": article.get("similarity_score", 0.0),
            "type": "news"
        })

    return entries


# Load structured disaster knowledge/tips
def load_disaster_knowledge(path="data/disaster_knowledge.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Knowledge file not found: {path}")
    
    df = pd.read_csv(path)
    return [
        {
            "text": row["Information"],
            "type": "tip",
            "category": row["DisasterType_InfoType"]
        }
        for _, row in df.iterrows()
        if pd.notna(row["Information"])
    ]

# Combine both news and tips into texts + metadata
def combine_all_data(news_json_path="data/classified_disaster_news.json", knowledge_path="data/disaster_knowledge.csv"):
    live_news = load_live_news(news_json_path)
    disaster_knowledge = load_disaster_knowledge(knowledge_path)

    all_entries = live_news + disaster_knowledge

    # 👇 Preprocess each text
    for entry in all_entries:
        entry["text"] = preprocess_text(entry["text"])

    all_texts = [entry["text"] for entry in all_entries]
    metadata = all_entries

    return all_texts, metadata

