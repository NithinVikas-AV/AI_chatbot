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
from werkzeug.utils import secure_filename
import json

# Load environment variables from .env file
load_dotenv()

# Get the API key
API_KEY = os.getenv("GEMINI_API_KEY")

# API key
genai.configure(api_key=API_KEY)

# Models
model = genai.GenerativeModel("gemini-2.0-flash")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

#Shared file
SHARED_FILE = os.path.join("uploads", "shared_file.csv")
SHARED_TEXT_FILE = os.path.join("uploads", "shared_text.txt")

def load_or_compute_embeddings(input_df):
    # Build combined_text with column names
    input_df['combined_text'] = input_df.apply(
        lambda row: ' | '.join([f"{col}: {row[col]}" for col in input_df.columns]), axis=1
    )
    input_df['embedding'] = input_df['combined_text'].apply(lambda x: embedding_model.encode(x).tolist())
    return input_df[['combined_text', 'embedding']]


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

        # Check if shared_file exists
        if not os.path.exists(SHARED_FILE) or os.path.getsize(SHARED_FILE) == 0:
            context_text = None
        else:
            df_shared = pd.read_csv(SHARED_FILE)
            df_shared['embedding'] = df_shared['embedding'].apply(lambda x: json.loads(x))
            prompt_embedding = embedding_model.encode(prompt)

            embeddings_matrix = np.vstack(df_shared['embedding'].values)
            similarities = cosine_similarity([prompt_embedding], embeddings_matrix)[0]
            df_shared['similarity'] = similarities

            top_matches = df_shared.sort_values('similarity', ascending=False).head(5)
            context_text = "\n\n".join(top_matches['combined_text'].values)

        instruction = """
        You are a polite and concise assistant designed to help users based on the context provided.
        - Use the context below to answer the user's question **only if it is relevant**.
        - If the context is not relevant or does not contain the answer, **do not mention this to the user**.
        - Instead, use your own general knowledge to answer the question politely and informatively.
        - Keep your answers short, helpful, and to the point.
        """

        final_prompt = f"""
        Instructions before answering:
        {instruction}

        Context:
        {context_text if context_text else "None"}

        User Question:
        {prompt}
        """

        response = model.generate_content(final_prompt)
        return jsonify({"reply": response.text})

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"reply": "An error occurred. Please try again later."}), 500


UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files.get("file")
    if not file or not file.filename.endswith(".csv"):
        return jsonify({"error": "Invalid file"}), 400

    # Read and validate CSV
    df_new = pd.read_csv(file)
    if df_new.empty:
        return jsonify({"error": "Empty CSV file."}), 400

    # Generate embeddings
    processed_df = load_or_compute_embeddings(df_new)

    # Save to shared_file.csv
    if os.path.exists(SHARED_FILE):
        processed_df.to_csv(SHARED_FILE, mode='a', header=False, index=False)
    else:
        processed_df.to_csv(SHARED_FILE, index=False)

    return jsonify({"message": "File uploaded and data saved to shared_file.csv."}), 200


if __name__ == "__main__":
    app.run(debug=True)