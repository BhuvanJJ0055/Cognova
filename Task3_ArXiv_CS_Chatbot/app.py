"""
Task 3 - app.py (Standalone Streamlit UI)
Author: Bhuvan J J

Dedicated scientific paper chatbot dashboard.
Provides paper search, NLP taggers, interactive LLM chat, live arXiv API 
ingestion, and 2D PCA paper cluster visualizations.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Import local modules
from arxiv_loader import get_local_papers, search_arxiv_api, import_papers_to_local
from build_arxiv_index import ArXivRetriever, CSV_PATH
from nlp_utils import extract_concepts, summarize_text
from llm_explainer import generate_explanation

# Page Config
st.set_page_config(
    page_title="Cognova ArXiv Expert",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Scientific Deep Indigo/Gold theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .arxiv-header {
        background: linear-gradient(135deg, #1f1c2c 0%, #928dab 100%);
        padding: 2.2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .arxiv-header h1 {
        font-weight: 700;
        font-size: 2.6rem;
        margin: 0;
    }
    
    .arxiv-header p {
        font-size: 1.05rem;
        opacity: 0.9;
        margin: 0.5rem 0 0 0;
    }
    
    /* Concept Tags */
    .concept-pill {
        display: inline-block;
        padding: 0.3rem 0.7rem;
        margin: 0.15rem;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: rgba(146, 141, 171, 0.15);
        color: #4b3f72;
        border: 1px solid rgba(146, 141, 171, 0.3);
    }

    /* Message Bubbles */
    .chat-bubble {
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
    }
    
    .user-bubble {
        background-color: #f3f0f7;
        border-left: 5px solid #4b3f72;
    }
    
    .bot-bubble {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #d4af37;
    }
    
    .score-badge {
        float: right;
        background-color: #d4af37;
        color: #1f1c2c;
        padding: 0.2rem 0.5rem;
        font-size: 0.75rem;
        font-weight: bold;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session States
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
if "last_retrieved" not in st.session_state:
    st.session_state.last_retrieved = []

# Load local papers
df_papers = get_local_papers()
total_papers = len(df_papers)

# Initialize Retriever (triggers auto DB build if missing)
@st.cache_resource
def get_retriever():
    return ArXivRetriever()

retriever = get_retriever()

# Sidebar Navigation
with st.sidebar:
    st.image("https://img.icons8.com/clouds/100/book.png", width=100)
    st.title("ArXiv Expert Hub")
    st.write("Academic Search & Chat controls.")
    
    st.subheader("Database Scope")
    st.metric(label="Total Indexed Papers", value=f"{total_papers} papers")
    st.metric(label="Domains Covered", value=f"{df_papers['primary_category'].nunique()} fields")
    
    st.subheader("Retrieval Tuning")
    conf_threshold = st.slider(
        "Semantic Similarity Cutoff",
        min_value=0.0,
        max_value=0.5,
        value=0.08,
        step=0.02,
        help="Lower values yield broader matches. Higher values require tighter word intersections."
    )
    
    max_results = st.slider(
        "Max Reference Papers",
        min_value=1,
        max_value=5,
        value=3
    )
    
    st.subheader("LLM Configuration")
    llm_option = st.selectbox(
        "Explanation Engine",
        ["Local Fallback (Deterministic)", "Hugging Face Inference API", "Google Gemini API"]
    )
    
    hf_token = ""
    gemini_key = ""
    
    if llm_option == "Hugging Face Inference API":
        has_system_hf = bool(os.environ.get("HF_API_TOKEN"))
        placeholder = "System default active" if has_system_hf else "Enter HF API Token"
        hf_token_input = st.text_input("HF API Token", type="password", placeholder=placeholder, help="Input Hugging Face Bearer Token")
        hf_token = hf_token_input if hf_token_input else os.environ.get("HF_API_TOKEN", "")
        st.info("Uses Meta Llama 3 8B. If token empty, queries public endpoints (rate-limited).")
    elif llm_option == "Google Gemini API":
        has_system_gemini = bool(os.environ.get("GEMINI_API_KEY"))
        placeholder = "System default active" if has_system_gemini else "Enter Gemini API Key"
        gemini_key_input = st.text_input("Gemini API Key", type="password", placeholder=placeholder, help="Input Google Gemini API Key")
        gemini_key = gemini_key_input if gemini_key_input else os.environ.get("GEMINI_API_KEY", "")
        
    if st.button("Clear Chat Memory"):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")

# Title Header
st.markdown("""
    <div class="arxiv-header">
        <h1>📚 Cognova ArXiv Scientific Expert</h1>
        <p>Explore Seminal Machine Learning Research, Summarise Papers, and Discuss Advanced Concepts</p>
    </div>
""", unsafe_allow_html=True)

# Main Application Tabs
tab_chat, tab_explore, tab_visualize = st.tabs([
    "💬 Expert Advisor Chat", 
    "🔍 Search & Ingest Papers", 
    "📊 Category Similarity Map"
])

# TABS 1: CHAT PANEL
with tab_chat:
    st.markdown("### Ask the AI Expert")
    st.write("Ask a research question. The advisor will match it against seminal papers and generate a conceptual explanation.")
    
    # Display Chat Log
    for msg in st.session_state.chat_history:
        bubble_class = "user-bubble" if msg["role"] == "user" else "bot-bubble"
        speaker = "Student" if msg["role"] == "user" else "ArXiv Expert"
        st.markdown(f"""
        <div class="chat-bubble {bubble_class}">
            <b>{speaker}:</b><br>
            {msg['content']}
        </div>
        """, unsafe_allow_html=True)
        
    # Input box
    user_query = st.text_input("Enter query or follow-up question:", key="arxiv_user_query")
    
    if st.button("Send", type="primary") and user_query:
        # User message
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # 1. Semantic search
        with st.spinner("Searching indexing vector space..."):
            hits = retriever.retrieve(user_query, threshold=conf_threshold, top_k=max_results)
            st.session_state.last_retrieved = hits
            
        # 2. Extract NER technical concepts
        concepts = extract_concepts(user_query)
        
        # 3. Generate answer
        with st.spinner("Thinking (generating response)..."):
            explanation = generate_explanation(
                query=user_query,
                retrieved_papers=hits,
                chat_history=st.session_state.chat_history[:-1], # pass previous turns
                hf_token=hf_token if llm_option == "Hugging Face Inference API" else None,
                gemini_key=gemini_key if llm_option == "Google Gemini API" else None
            )
            
        # Add to history
        st.session_state.chat_history.append({"role": "assistant", "content": explanation})
        
        # Rerun to update chat display
        st.rerun()

    # Sidebar reference matching
    if st.session_state.last_retrieved:
        st.markdown("---")
        st.markdown("### 🔍 Relevant Reference Context")
        
        # Extract keywords
        last_q = st.session_state.chat_history[-2]["content"] if len(st.session_state.chat_history) >= 2 else ""
        concepts = extract_concepts(last_q) if last_q else []
        
        if concepts:
            st.write("**Extracted Technical Concepts:**")
            pills_html = "".join([f'<span class="concept-pill">⚙️ {c.title()}</span>' for c in concepts])
            st.markdown(f"<div>{pills_html}</div>", unsafe_allow_html=True)
            
        st.write(f"Matched **{len(st.session_state.last_retrieved)}** research papers in local index:")
        
        for idx, hit in enumerate(st.session_state.last_retrieved):
            score_pct = int(hit["similarity_score"] * 100)
            
            with st.expander(f"📚 {hit['title']} ({score_pct}% Match Score)"):
                st.write(f"**Authors**: {hit['authors']}")
                st.write(f"**Published**: {hit['published']} | **Category**: {hit['primary_category']}")
                st.write(f"**Abstract**: {hit['summary']}")
                st.markdown(f"[Link to full paper]({hit['url']})")
                
                # Sentence extractive summary
                summary = summarize_text(hit["summary"], num_sentences=2)
                st.info(f"💡 **Extracted Key Points:** {summary}")

# TAB 2: SEARCH & INGEST
with tab_explore:
    st.subheader("Manage Research Database")
    st.write("Browse current papers or fetch new publications dynamically from the live arXiv API.")
    
    col_l, col_r = st.columns([1, 1])
    
    with col_l:
        st.markdown("#### 📂 Local Scientific Catalog")
        search_kw = st.text_input("Filter catalog by keyword:", placeholder="e.g. BERT or residual")
        
        df_display = df_papers.copy()
        if search_kw:
            df_display = df_display[
                df_display["title"].str.contains(search_kw, case=False, na=False) |
                df_display["summary"].str.contains(search_kw, case=False, na=False) |
                df_display["authors"].str.contains(search_kw, case=False, na=False)
            ]
            
        st.write(f"Displaying **{len(df_display)}** papers:")
        st.dataframe(
            df_display[["id", "title", "authors", "primary_category", "published"]],
            use_container_width=True
        )
        
    with col_r:
        st.markdown("#### 🌐 Fetch Fresh Papers from arXiv API")
        api_query = st.text_input("arXiv API Query string:", placeholder="e.g. attention model or category:cs.LG")
        max_num = st.number_input("Max papers to fetch:", min_value=1, max_value=20, value=5)
        
        if st.button("Search arXiv API"):
            if api_query:
                with st.spinner("Querying arXiv server..."):
                    fetched = search_arxiv_api(api_query, max_results=max_num)
                    
                if fetched:
                    st.success(f"Retrieved {len(fetched)} papers matching search.")
                    
                    # Show fetched papers list
                    for idx, paper in enumerate(fetched):
                        with st.container():
                            st.markdown(f"**{idx+1}. {paper['title']}**")
                            st.write(f"Authors: {paper['authors']} | Published: {paper['published']}")
                            st.write(f"Category: {paper['primary_category']}")
                            
                    # Add to local catalog button
                    if st.button("Import Fetched Papers into Local DB"):
                        added = import_papers_to_local(fetched)
                        st.success(f"Import complete! Added {added} new unique papers. Rebuilding index...")
                        # Force index rebuild
                        retriever.build_and_save_index(CSV_PATH)
                        st.rerun()
                else:
                    st.info("No matching papers found on arXiv. Check query formatting.")
            else:
                st.error("Please enter a query string.")

# TAB 3: PCA CLUSTER VISUALIZATION
with tab_explore:
    pass # Managed separately in visualizer tab to prevent duplicates
    
with tab_visualize:
    st.subheader("📊 Document Semantic Map (PCA Projection)")
    st.write(
        "This scatter plot projects the TF-IDF representation of our papers' abstracts "
        "into 2D space using Principal Component Analysis (PCA). Points that cluster "
        "closer together share similar semantic vocabulary and themes."
    )
    
    # Run PCA
    if retriever.vectorizer is not None and retriever.tfidf_matrix is not None:
        try:
            with st.spinner("Computing dimensional projection..."):
                # Get dense matrix
                dense_matrix = retriever.tfidf_matrix.toarray()
                
                # Fit PCA
                n_papers = dense_matrix.shape[0]
                n_components = min(2, n_papers)
                
                if n_components >= 2:
                    pca = PCA(n_components=2)
                    coords = pca.fit_transform(dense_matrix)
                    
                    # Create plotting dataframe
                    df_plot = pd.DataFrame(retriever.metadata)
                    df_plot["x"] = coords[:, 0]
                    df_plot["y"] = coords[:, 1]
                    
                    # Matplotlib plot
                    plt.figure(figsize=(10, 6))
                    sns.set_theme(style="whitegrid")
                    
                    # Use categories for coloring
                    categories = df_plot["primary_category"].unique()
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Color palette
                    palette = sns.color_palette("Set2", len(categories))
                    color_map = dict(zip(categories, palette))
                    
                    for cat in categories:
                        df_cat = df_plot[df_plot["primary_category"] == cat]
                        ax.scatter(
                            df_cat["x"], df_cat["y"], 
                            label=cat, 
                            s=100, 
                            alpha=0.8,
                            edgecolors='w',
                            color=color_map[cat]
                        )
                        
                    # Annotate top 5 papers
                    # Limit annotation to prevent overlap clutter
                    for idx, row in df_plot.head(8).iterrows():
                        # Truncate title
                        short_title = row['title'][:20] + "..." if len(row['title']) > 20 else row['title']
                        ax.annotate(
                            short_title, 
                            (row['x'], row['y']),
                            xytext=(5, 5),
                            textcoords='offset points',
                            fontsize=8,
                            alpha=0.7
                        )
                        
                    ax.set_title("Paper Abstract Clusters (PCA projection)", fontsize=14, fontweight="bold", pad=15)
                    ax.set_xlabel("Component 1 (Variance)", fontsize=11)
                    ax.set_ylabel("Component 2 (Variance)", fontsize=11)
                    ax.legend(title="Primary Category", bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.tight_layout()
                    
                    # Render in Streamlit
                    st.pyplot(fig)
                else:
                    st.info("Add at least 3 papers to generate a 2D PCA cluster map.")
        except Exception as e:
            st.error(f"Failed to generate visualization: {e}")
    else:
        st.warning("Retriever index not loaded. Cannot run PCA visualization.")
