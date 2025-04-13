import requests
from config import GROQ_API_KEY

def call_groq(prompt, model="llama3-8b-8192"):  # ✅ Use working model
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful disaster response assistant."},
            {"role": "user", "content": prompt}
        ]
    }

    res = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)

    try:
        data = res.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        print("❌ Groq API error:", res.status_code, res.text)
        raise
