import pandas as pd
import numpy as np
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key
API_KEY = os.getenv("GEMINI_API_KEY")

# API key
genai.configure(api_key=API_KEY)

# Models
model = genai.GenerativeModel("gemini-2.0-flash")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Creating a Dataframe using pandas.
# Creating a column by combining every column in a row.
df = pd.read_csv("City_Info.csv")
df['combined_text'] = df[['Location', 'Country', 'Parking Available', 'Weather', 'Temperature', 'Rain', 'Wind Speed', 'Description']].astype(str).agg(' | '.join, axis=1)

# Embedding caching
EMBEDDING_CACHE = "city_embeddings.pkl"

def load_or_compute_embeddings(df):
    if os.path.exists(EMBEDDING_CACHE):
        with open(EMBEDDING_CACHE, 'rb') as f:
            df['embedding'] = pickle.load(f)
    else:
        df['embedding'] = df['combined_text'].apply(lambda x: embedding_model.encode(x))
        with open(EMBEDDING_CACHE, 'wb') as f:
            pickle.dump(df['embedding'].values, f)
    return df

df = load_or_compute_embeddings(df)

embeddings_matrix = np.vstack(df['embedding'].values)

app = Flask(__name__, static_folder='static')
CORS(app)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        prompt = data.get("message", "")
        print(f"Received prompt: {prompt}")

        if not prompt:
            return jsonify({"reply": "Please enter a valid question."}), 400

        # Prompt Embedding + RAG
        prompt_embedding = embedding_model.encode(prompt)
        similarities = cosine_similarity([prompt_embedding], embeddings_matrix)[0]
        df['similarity'] = similarities
        top_matches = df.sort_values('similarity', ascending=False).head(5)

        # Context Creation
        context_head = "Location | Country | Parking Available | Weather | Temperature | Rain | Wind Speed | Description"
        context_row = "\n\n".join(top_matches['combined_text'].values)
        context_text = f"{context_head}\n\n{context_row}"
        
        #Prompt Engineering
        instruction = """
                        You are a polite and concise assistant designed to help users based on the context provided.
                            - Use the context below to answer the user's question **only if it is relevant**.
                            - If the context is not relevant or does not contain the answer, **do not mention this to the user**.
                            - Instead, use your own general knowledge to answer the question politely and informatively.
                            - Keep your answers short, helpful, and to the point.
                      """

        #Final Prompt
        final_prompt = f"""Instructions before answering:
                            {instruction}
                            
                        Context:
                            {context_text}
                        
                        User Question: 
                            {prompt}
                        """

        response = model.generate_content(final_prompt)
        print(response.text)
        return jsonify({"reply": response.text})
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"reply": "An error occurred. Please try again later."}), 500

if __name__ == "__main__":
    app.run(debug=True)  # Removed debug=True; use gunicorn for production (e.g., gunicorn -w 4 -b 0.0.0.0:5000 app:app)