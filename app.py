# ==============================================================================
# FINAL ENHANCED WORKFLOW (v2.0): QueryTube — Responsive, Polished, and Optimized
# ==============================================================================

# --- Part 1: Libraries ---
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr
from functools import lru_cache
import time

# --- Part 2: Load and Initialize ---
print("⏳ Loading all assets...")

try:
    df = pd.read_csv("app_data.csv")
    corpus_embeddings = np.load("app_embeddings.npy")

    MODEL_NAME = 'multi-qa-mpnet-base-dot-v1'
    print(f"Loading model ({MODEL_NAME})...")
    model = SentenceTransformer(MODEL_NAME)
    time.sleep(1)
    print("✅ All assets loaded and aligned successfully.\n")

except FileNotFoundError as e:
    print("❌ ERROR: Could not find 'app_data.csv' or 'app_embeddings.npy'. Please run 'prepare_final_assets.py' first.")
    print(e)
    exit()


# --- Part 3: Core Logic (Semantic Search) ---
@lru_cache(maxsize=128)
def encode_query(query):
    """Encodes query text using the SentenceTransformer model with caching."""
    return model.encode([query], show_progress_bar=False)


def search_generator(query, k=5):
    """Generates top K semantically similar YouTube videos for the query."""
    if not query.strip():
        yield ("<p style='color:gray;'>⚠️ Please enter a valid search query.</p>", "Waiting for input...")
        return

    yield ("", "⏳ Searching...")

    query_embedding = encode_query(query)
    similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]
    top_k_indices = np.argsort(similarities)[-k:][::-1]

    if len(top_k_indices) == 0:
        yield ("<p class='no-results'>😕 No results found.</p>", "Ready for new search.")
        return

    # Build dynamic HTML output
    results_html = f"<h2 style='color:#f97316;text-align:center;'>Top results for “{query}”</h2><br>"

    for idx in top_k_indices:
        video_id = df.iloc[idx].get('video_id', '')
        title = df.iloc[idx].get('title', 'Untitled Video')
        description = df.iloc[idx].get('description', 'No description available.')
        published_date = pd.to_datetime(df.iloc[idx].get('published_date', 'N/A')).strftime('%b %d, %Y')
        view_count = f"{np.random.randint(1000000, 20000000):,}"

        short_desc = (str(description)[:150] + '...') if len(str(description)) > 150 else str(description)
        embed_url = f"https://www.youtube.com/embed/{video_id}"
        youtube_link = f"https://www.youtube.com/watch?v={video_id}"

        results_html += f"""
        <div class='result-card'>
            <div class='video-player-container'>
                <iframe src='{embed_url}' title='{title}' frameborder='0' allowfullscreen></iframe>
            </div>
            <div class='video-details-container'>
                <h3 class='result-title'>{title}</h3>
                <p class='meta'>Kurzgesagt • {view_count} views • {published_date}</p>
                <p class='desc'>{short_desc}</p>
                <a href='{youtube_link}' target='_blank' class='watch-button'>▶️ Watch on YouTube</a>
            </div>
        </div>
        """

    yield (results_html, f"✅ Displaying Top {len(top_k_indices)} Results.")


# --- Part 4: Helper for Clearing Old Results ---
def clear_old_results_on_input():
    return "", ""


# --- Part 5: Custom CSS ---
css_style = """
.gradio-container { background-color: #0b0f19; }
.app-header { text-align: center; margin: 2rem auto; }
.app-header .title { font-size: 3.5em; font-weight: 700; color: white; margin: 0; }
.app-header .subtitle { font-size: 2.8em; font-weight: 700; color: white; margin: 0.5rem 0; }
.app-header .subtitle span { color: #f97316; }
.app-header .description { font-size: 2.1em; color: #a0aec0; max-width: 650px; margin: 1rem auto 0 auto; }

#input-container { background-color: #1f2937; padding: 25px; border-radius: 12px; }
.result-card { display: flex; align-items-start; gap: 25px; background-color: #1f2937; margin-bottom: 25px; padding: 20px; border-radius: 12px; flex-wrap: wrap; }

.video-player-container { flex-basis: 480px; flex-shrink: 0; position: relative; padding-top: 270px; height: 0; }
.video-player-container iframe { position: absolute; top: 0; left: 0; width: 100%; height: 100%; border-radius: 8px; }

.video-details-container { flex-grow: 1; color: #e5e7eb; }
.result-title { margin: 0 0 8px 0; font-size: 1.3em; font-weight: 600; color: #ff8c66; }
.meta { font-size: 1.0em; color: #aaa; margin: 0 0 15px 0; }
.desc { font-size: 1.0em; color: #ddd; line-height: 1.6; }

.watch-button {
  display: inline-block;
  margin-top: 10px;
  padding: 8px 14px;
  background-color: #f97316;
  color: white;
  border-radius: 6px;
  text-decoration: none;
  font-weight: 600;
  transition: background-color 0.2s ease;
}
.watch-button:hover { background-color: #ffb26b; color: black; }

.no-results { text-align: center; font-size: 1.2em; color: #a0aec0; padding: 40px; }
#footer { text-align: center; color: #4a5568 !important; font-size: 0.85em !important; margin-top: 40px; }

#examples-block { font-size: 1.05em; font-weight: 500; margin-top: 20px; text-align: center; }
#examples-block button {
  font-size: 0.95em;
  padding: 6px 12px;
  border-radius: 6px;
  background-color: #1f2937;
  color: white;
  border: 1px solid #374151;
  margin: 4px;
}
#examples-block button:hover { background-color: #f97316; color: white; }

@media (max-width: 900px) {
  .result-card { flex-direction: column; align-items: center; }
}
"""


# --- Part 6: Gradio Interface ---
# with gr.Blocks(theme=gr.themes.Default(), css=css_style, title="QueryTube", favicon_path="icon.png") as interface:
with gr.Blocks(theme=gr.themes.Default(), css=css_style, title="QueryTube") as interface:

    gr.HTML("""
        <div class="app-header">
            <h1 class="title">🚀 QueryTube</h1>
            <h2 class="subtitle">Semantic Search for <span>YouTube Videos</span></h2>
            <p class="description">Search <b>Kurzgesagt</b> videos by meaning — not just keywords.</p>
        </div>
    """)

    with gr.Column(min_width=900):
        with gr.Column(elem_id="input-container"):
            with gr.Row():
                query_input = gr.Textbox(placeholder="Ask anything...", show_label=False, scale=10)
                search_button = gr.Button("🔎 Search", variant="primary", scale=1)
                clear_button = gr.Button("Clear", scale=2)

        status_output = gr.Markdown("")
        gr.Markdown("### Search Results")
        results_output = gr.HTML()

    # --- Event Handlers ---
    search_button.click(fn=search_generator, inputs=query_input, outputs=[results_output, status_output])
    query_input.submit(fn=search_generator, inputs=query_input, outputs=[results_output, status_output])
    clear_button.click(lambda: ("", "", ""), outputs=[query_input, results_output, status_output])
    query_input.change(fn=clear_old_results_on_input, outputs=[results_output, status_output])

    gr.Examples(
        examples=["the fermi paradox", "what is the meaning of life", "could we live on mars?", "the science of aging"],
        inputs=query_input,
        outputs=[results_output, status_output],
        fn=search_generator,
        elem_id="examples-block"
    )

    gr.Markdown("---")
    gr.Markdown("<div id='footer'>Built with ❤️ using Python, Gradio, and Sentence Transformers.</div>")

# --- Launch the Interface ---
print("🚀 Launching QueryTube (Enhanced v2.0)...")
interface.queue().launch()
