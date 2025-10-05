import os
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import streamlit as st

#load dataset
csv_path = "Dataset.csv"

df = pd.read_csv(csv_path)

#embedding 
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embed_model = SentenceTransformer(model_name)

texts = (df['title'] + ". " + df['description']).tolist()
embeddings = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
embeddings = normalize(embeddings, axis=1)

emb = embeddings.shape[1]
index = faiss.IndexFlatIP(emb)
index.add(embeddings)

id_to_course = df['course_id'].tolist()

#recommendation
def recommend_courses(profile, completed_ids=None, top_k=5):
    if completed_ids is None:
        completed_ids = []

    q_emb = embed_model.encode([profile], convert_to_numpy=True)
    q_emb = normalize(q_emb, axis=1)
    scores, indices = index.search(q_emb, top_k + len(completed_ids) + 10)
    scores, indices = scores[0], indices[0]

    results = []
    for score, idx in zip(scores, indices):
        if idx < 0 or idx >= len(id_to_course):
            continue
        cid = id_to_course[idx]
        if cid in completed_ids:
            continue
        row = df.loc[df['course_id'] == cid].iloc[0]
        results.append((cid, row['title'], row['description'], float(score)))
        if len(results) >= top_k:
            break
    return results

#streamlit ui
st.set_page_config(page_title="Course Recommendation engine", page_icon="", layout="centered")
st.title("Personalized Course Recommendation Engine")

st.write("Enter background, courses completed , and interests to get top recommendations.")

profile = st.text_area("Describe your background & interests:", height=100)
completed = st.text_input("Completed course IDs (comma separated, e.g. C005,C006):")

if st.button("Recommend"):
    if not profile.strip():
        st.warning("Please enter your profile description.")
    else:
        completed_ids = [c.strip() for c in completed.split(",") if c.strip()]
        recs = recommend_courses(profile, completed_ids)
        if not recs:
            st.error("No recommendations found.")
        else:
            st.subheader("Top Recommendations:")
            for cid, title, desc, score in recs:
                with st.expander(f"{cid} {title} (score={score:.4f})"):
                    st.write(desc)

