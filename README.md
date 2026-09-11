# AmazonHelp AI Support Agent

Hey there! This is my submission for the Hiver SDE Intern take-home assignment. 

I decided to build an AI support agent focused on `@AmazonHelp`. The goal was to take a messy, real-world Twitter dataset and build an agent that can classify customer intents, draft replies based on how Amazon historically handled similar issues (using RAG), and figure out if a human needs to step in.

## Prerequisites

You'll need a few basics to get this running:
1. Python 3.9+
2. `uv` (or just standard `pip`)
3. A Groq API key (I used Groq because their inference speed is insanely fast for prototyping)

## How to run this (in under 15 mins)

I designed this to be super easy to reproduce.

1. Clone this repo and drop the `twcs.csv` Kaggle dataset into the root directory.
2. Set up your virtual environment and install the dependencies:
   ```bash
   uv venv
   uv pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory and drop your Groq API key in there:
   ```env
   GROQ_API_KEY=your_api_key_here
   ```
4. **Prep the Data**: Run the preprocessing script. This filters the massive 3M row dataset down to just AmazonHelp threads and creates our historical pairs.
   ```bash
   uv run preprocess.py
   ```
5. **Test the Agent**: Want to see it in action? Just run the agent script directly to see a sample interaction.
   ```bash
   uv run agent.py
   ```
6. **Run the Evals**: Run the evaluation harness to see how the agent scores against the golden dataset.
   ```bash
   uv run eval.py
   ```

## Under the Hood

- **Data Pipeline (`preprocess.py`)**: Sifts through the noise of the raw Kaggle dataset to isolate clean customer/agent turn-pairs for Amazon.
- **The Agent (`agent.py`)**: Instead of relying on basic keyword search, I used `SentenceTransformers` to build dense embeddings of past resolutions. When a new tweet comes in, it fetches the 3 most semantically similar past cases, and injects them into a prompt for a Qwen LLM to draft a response, classify the intent, and make an escalation call.
- **The Evals (`eval.py`)**: Running manual evals is tedious, so I built an LLM-as-judge pipeline. It runs the golden dataset through the agent and scores the drafted replies against the *actual* historical replies.

For a deep dive into my design decisions, failure analysis, and metrics, check out `Report.md`!
