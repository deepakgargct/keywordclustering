import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import base64
import io

# Set page configuration
st.set_page_config(page_title="Semantic Keyword Clustering", layout="wide")

st.title("🔍 Semantic Keyword Clustering Tool")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file containing a 'keyword' column", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file
    df = pd.read_csv(uploaded_file)

    # Check if 'keyword' column exists
    if 'keyword' not in df.columns:
        st.error("The uploaded CSV must contain a 'keyword' column.")
    else:
        # Display the first few keywords
        st.subheader("Sample Keywords")
        st.write(df['keyword'].head())

        # Load the SentenceTransformer model
        @st.cache_resource
        def load_model():
            return SentenceTransformer('all-MiniLM-L6-v2')

        model = load_model()

        # Encode the keywords
        st.subheader("Encoding Keywords...")
        embeddings = model.encode(df['keyword'].tolist(), show_progress_bar=True)

        # Perform clustering
        st.subheader("Clustering Keywords...")
        # You can adjust the number of clusters as needed
        num_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=20, value=5)
        clustering_model = AgglomerativeClustering(n_clusters=num_clusters)
        clustering_model.fit(embeddings)
        cluster_assignment = clustering_model.labels_

        # Assign clusters to keywords
        df['cluster'] = cluster_assignment

        # Display clustered keywords
        st.subheader("Clustered Keywords")
        for cluster_num in range(num_clusters):
            st.write(f"**Cluster {cluster_num}**")
            cluster_keywords = df[df['cluster'] == cluster_num]['keyword'].tolist()
            st.write(cluster_keywords)

        # Download clustered data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

        csv_data = convert_df_to_csv(df)
        st.download_button(
            label="📥 Download Clustered Keywords as CSV",
            data=csv_data,
            file_name='clustered_keywords.csv',
            mime='text/csv',
        )
