# QueryTube: A Semantic Search Engine for YouTube Videos

This project is a complete, end-to-end semantic search application that allows users to search for YouTube videos from the Kurzgesagt channel using natural language queries. Instead of matching keywords, it understands the *meaning* behind a query to find the top 5 most conceptually relevant videos.

## Features

*   **Pure Semantic Search:** Powered by the `multi-qa-mpnet-base-dot-v1` sentence transformer model.
*   **Interactive UI:** Built with Gradio, featuring a clean, modern, and user-friendly interface.
*   **Embedded Video Previews:** Search results include embedded YouTube players for immediate content preview.
*   **Dynamic Descriptions:** Displays the unique, original video description for each search result.

## Live Demo

The live, interactive application is deployed on Hugging Face Spaces. You can access it here:

**[PASTE YOUR HUGGING FACE LINK HERE]**

---

## How to Run This Project Locally

### 1. Prerequisites
*   Python 3.9+
*   Git

### 2. Clone the Repository

```bash

git clone (https://github.com/manasa-kyatham/QueryTube-Semantic-Search.git)

cd QueryTube-Semantic-Search

```

### 3. Set Up The Environment and Install Dependencies
It is highly recommended to use a virtual environment.
code
Bash
# Create and activate a virtual environment
```bash
python -m venv venv
```

```bash
venv\Scripts\activate  # On Windows
```

# Install the required libraries
```bash
pip install -r requirements.txt
```

### 4. Run the Application
Once the dependencies are installed, you can launch the Gradio application:

```bash
python app.py
```
This will start a local web server. Open the URL provided in the terminal (e.g., http://127.0.0.1:7860) in your web browser to use the application.

