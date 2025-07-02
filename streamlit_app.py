# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load model and data
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

@st.cache_data
def load_data():
    df = pd.read_pickle("bankfaqs_with_local_embeddings.pkl")
    return df

model = load_model()
df = load_data()

# Similarity function
def get_best_match(user_query: str, df: pd.DataFrame, model: SentenceTransformer, use_combined=False):
    query_embedding = model.encode([user_query])[0].reshape(1, -1)
    embeddings_col = 'combined_embedding' if use_combined else 'question_embedding'
    embeddings = np.vstack(df[embeddings_col].values)
    
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_idx = np.argmax(similarities)
    
    matched_question = df.iloc[top_idx]['Question']
    matched_answer = df.iloc[top_idx]['Answer']
    score = similarities[top_idx]
    
    return matched_question, matched_answer, score, use_combined

# Streamlit UI
st.title("Bank FAQ Chatbot")

user_query = st.text_input("Ask a question:")

if user_query:
    with st.spinner("Searching..."):
        q, a, score, combined_used = get_best_match(user_query, df, model, use_combined=False)
        
        if score < 0.5:
            q, a, score, combined_used = get_best_match(user_query, df, model, use_combined=True)

    st.subheader("🔍 Best Match Found:")
    st.markdown(f"**Matched Question:** {q}")
    st.markdown(f"**Answer:** {a}")
    st.markdown(f"**Similarity Score:** `{score:.4f}`")