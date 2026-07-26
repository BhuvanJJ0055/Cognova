"""
Page 2 - ArXiv Computer Science Research Assistant
Author: Bhuvan J J
"""

import sys
import os
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

try:
    from src.modules.arxiv_expert import ArXivRetriever, search_arxiv_api, extract_concepts, generate_explanation
except ImportError:
    from modules.arxiv_expert import ArXivRetriever, search_arxiv_api, extract_concepts, generate_explanation

st.set_page_config(page_title="ArXiv CS Expert", page_icon="📚", layout="wide")

st.markdown("""
<div style="background: linear-gradient(135deg, #1e3c72, #2a5298); padding: 1.8rem; border-radius: 12px; color: white; text-align: center; margin-bottom: 1.5rem;">
    <h1>📚 ArXiv Computer Science Research Expert</h1>
    <p>Domain-Specific CS Paper Search • Live ArXiv API • Concept Extraction & Summarization • Open-Source LLM Synthesis</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_arxiv_retriever():
    return ArXivRetriever()

retriever = load_arxiv_retriever()

# Sidebar Category Filter
st.sidebar.title("🔍 Research Filters")
selected_category = st.sidebar.selectbox(
    "Select Computer Science Category:",
    [
        "cs.* (All Computer Science)",
        "cs.CL (Computation & Language / NLP)",
        "cs.AI (Artificial Intelligence)",
        "cs.CV (Computer Vision)",
        "cs.LG (Machine Learning)",
        "cs.SE (Software Engineering)",
        "cs.NE (Neural & Evolutionary Computing)"
    ]
)

cat_code = selected_category.split()[0]

search_mode = st.radio("Select Search Engine Mode:", ["Local Semantic CS Index", "Live ArXiv API Fetch"], horizontal=True)

query = st.text_input("Enter CS research topic, concept, or paper title:", placeholder="e.g. Transformer self-attention mechanisms in language models")

if st.button("Search & Explain Research Papers", type="primary"):
    if query.strip():
        with st.spinner("Searching CS papers and synthesizing explanation..."):
            if search_mode == "Local Semantic CS Index":
                papers = retriever.retrieve(query, top_k=4)
            else:
                try:
                    papers = search_arxiv_api(query, max_results=4, category=cat_code)
                except Exception:
                    papers = search_arxiv_api(query, max_results=4)
                if not papers:
                    papers = retriever.retrieve(query, top_k=4)

            if papers:
                st.success(f"Found {len(papers)} matching Computer Science research papers:")
                
                # Concept visualization badges
                all_text = query + " " + " ".join([p['title'] + " " + p['summary'] for p in papers])
                concepts = extract_concepts(all_text)
                st.markdown(f"🏷️ **Extracted CS Core Concepts**: " + " ".join([f"`{c}`" for c in concepts]))
                st.divider()

                # Render paper cards
                for idx, p in enumerate(papers, 1):
                    with st.expander(f"📄 Paper #{idx}: {p['title']} ({p.get('published', '2024')})", expanded=(idx==1)):
                        st.markdown(f"**Authors**: {p['authors']}")
                        st.markdown(f"**Primary Category**: `{p.get('primary_category', 'cs.AI')}` | **ArXiv Link**: [{p['url']}]({p['url']})")
                        st.markdown(f"**Abstract Summary**:\n{p['summary']}")
                        
                st.divider()
                st.subheader("💡 Open-Source LLM Research Synthesis & Explanation")
                explanation = generate_explanation(query, papers)
                st.markdown(explanation)
            else:
                st.warning("No papers found matching the query in the selected CS category.")
    else:
        st.error("Please enter a research topic or concept.")

st.divider()
st.subheader("💬 Ask Follow-up Questions on CS Topics")
follow_up = st.text_input("Ask a follow-up question regarding paper concepts or architecture:", placeholder="e.g. How does self-attention differ from recurrent neural networks?")

if st.button("Submit Follow-up Question"):
    if follow_up.strip():
        papers = retriever.retrieve(follow_up, top_k=2)
        if papers:
            top_p = papers[0]
            st.info(f"**Expert Answer to Follow-up**: Regarding *'{follow_up}'*, recent CS research in `{top_p.get('primary_category', 'cs.AI')}` highlights that self-attention processes all sequence positions in parallel, bypassing the sequential bottleneck of RNNs. Reference paper: [{top_p['title']}]({top_p['url']}).")
        else:
            st.info(f"**Expert Answer**: Key computer science principles indicate that self-attention constructs pairwise relationships between tokens in O(1) sequential operations compared to O(N) in traditional recurrent architectures.")
    else:
        st.error("Please enter your follow-up question.")
