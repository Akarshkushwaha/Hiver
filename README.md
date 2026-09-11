# AmazonHelp AI Support Agent

This is a take-home assignment solution for the Hiver SDE Intern position. It builds an AI support agent that classifies intents, drafts historically-grounded replies (using RAG over past Twitter resolutions), and makes escalation decisions.

## Requirements

1. Python 3.9+
2. `uv` (or `pip`)
3. A Groq API key (used for LLM inference)

## Quick Start (Under 15 Minutes)

1. Clone this repository and ensure the `twcs.csv` Kaggle dataset is in the root directory.
2. Setup the environment and install dependencies:
   ```bash
   uv venv
   uv pip install -r requirements.txt
   # OR: pip install pandas scikit-learn groq tqdm python-dotenv
   ```
3. Add your Groq API key to a `.env` file:
   ```env
   GROQ_API_KEY=your_api_key_here
   ```
4. **Preprocess the data** (extracts AmazonHelp tweets and builds the historical pairs):
   ```bash
   uv run preprocess.py
   ```
5. **Run the AI Agent** (Interactive test):
   ```bash
   uv run agent.py
   ```
6. **Run the Evaluation Harness**:
   ```bash
   uv run eval.py
   ```

## Architecture

- **Data Pipeline**: `preprocess.py` filters the 3M row Kaggle dataset to isolate customer/agent turn-pairs for `@AmazonHelp`.
- **RAG + Classifier (agent.py)**: Uses a dense embedding model (SentenceTransformers) to find historically similar customer issues. It then injects these past resolutions into the LLM prompt (Groq's Qwen model) to draft a response, classify intent, and decide on escalation.
- **Evaluation (eval.py)**: Evaluates a hand-labelled golden dataset using LLM-as-judge to compare draft replies against actual historical replies.
