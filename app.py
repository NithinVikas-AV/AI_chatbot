import pandas as pd
import numpy as np
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

genai.configure(api_key="AIzaSyAzbxmAPyzcR595QvjWsTHoIlprJmaKlJQ")

model = genai.GenerativeModel("gemini-2.0-flash")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

df = pd.read_csv("City_Info.csv")
df['combined_text'] = df[['Location', 'Country', 'Parking Available', 'Weather', 'Temperature', 'Rain', 'Wind Speed', 'Description']].astype(str).agg(' | '.join, axis=1)

df['embedding'] = df['combined_text'].apply(lambda x: embedding_model.encode(x))

app = Flask(__name__)
CORS(app) 

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat_v3", methods=["POST"])
def chat_v3():
    data = request.json
    prompt = data.get("message", "")
    print(f"Received prompt: {prompt}")

    if not prompt:
        return jsonify({"reply": "Please enter a valid question."}), 400

    # Embedding + RAG
    prompt_embedding = embedding_model.encode(prompt)
    embeddings_matrix = np.vstack(df['embedding'].values)
    similarities = cosine_similarity([prompt_embedding], embeddings_matrix)[0]
    df['similarity'] = similarities
    top_matches = df.sort_values('similarity', ascending=False).head(5)

    # Context + Prompt
    context_head = "Location | Country | Parking Available | Weather | Temperature | Rain | Wind Speed | Description"
    context_row = "\n\n".join(top_matches['combined_text'].values)
    context_text = f"{context_head}\n\n{context_row}"
    final_prompt = f"Use the following information to answer the question:\n\n{context_text}\n\nQuestion: {prompt}"

    response = model.generate_content(final_prompt)
    print(response.text)
    return jsonify({"reply": response.text})

if __name__ == "__main__":
    app.run(debug=True)