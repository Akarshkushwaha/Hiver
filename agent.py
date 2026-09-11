import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import os
import json
from dotenv import load_dotenv

class AmazonSupportAgent:
    def __init__(self, pairs_csv_path="AmazonHelp_pairs.csv"):
        load_dotenv()
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        
        print("Loading historical data for RAG...")
        df = pd.read_csv(pairs_csv_path).dropna(subset=['customer_text', 'brand_text'])
        # Sample 5k for fast vectorization with SentenceTransformers
        self.history_df = df.sample(n=min(5000, len(df)), random_state=42).reset_index(drop=True)
        
        print("Loading dense embedding model and building Index...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        print("Encoding historical messages...")
        self.embeddings = self.embedder.encode(self.history_df['customer_text'].tolist(), show_progress_bar=True)
        
        self.intents = {
            "Delivery/Shipping Issue": "Where is my package? Late delivery, tracking questions, carrier issues.",
            "Returns/Refunds": "How do I return this? Where is my refund? Item arrived damaged.",
            "Account/Billing Issue": "Why was I charged? Prime membership questions, locked account.",
            "Product/Item Defect": "Item is broken, missing parts, or doesn't work.",
            "Digital Services (Video/Kindle)": "Prime Video not working, Kindle book won't download, Amazon Music issues.",
            "Other/General Inquiry": "General complaints, compliments, or questions that don't fit the above."
        }

    def get_similar_historical_resolutions(self, text, top_k=3):
        query_vec = self.embedder.encode([text])
        sims = cosine_similarity(query_vec, self.embeddings).flatten()
        top_indices = sims.argsort()[-top_k:][::-1]
        
        resolutions = []
        for idx in top_indices:
            resolutions.append({
                "customer": self.history_df.iloc[idx]['customer_text'],
                "brand": self.history_df.iloc[idx]['brand_text']
            })
        return resolutions

    def process_message(self, customer_text):
        similar_cases = self.get_similar_historical_resolutions(customer_text)
        cases_str = "\n".join([f"- Past Customer: {c['customer']}\n  Past Agent: {c['brand']}" for c in similar_cases])
        
        intent_str = "\n".join([f"- {k}: {v}" for k, v in self.intents.items()])
        
        prompt = f"""You are an expert AI customer support agent for Amazon.

Customer Message: "{customer_text}"

Here are 3 historical examples of how Amazon resolved similar issues:
{cases_str}

TASK:
1. Classify the intent into EXACTLY ONE of these categories:
{intent_str}

2. Decide if this needs escalation ("Yes" or "No"). 
   - Rule: Escalate ONLY if the customer is extremely angry/threatening, OR if the issue is a complex security/account lock issue.
   - Rule: Do NOT escalate routine tracking, shipping, or basic refund questions. 

3. Draft a helpful, polite reply. 
   - Rule: Ground your tone and structure in the historical examples.
   - Rule: If historical examples ask to DM a link, include "Please DM us so we can help: https://amzn.to/..."

Output ONLY valid JSON:
{{
    "intent": "exact string of the category name (e.g. Delivery/Shipping Issue)",
    "escalate": "Yes" or "No",
    "escalate_reason": "brief reason",
    "draft_reply": "your drafted response"
}}
"""
        try:
            completion = self.client.chat.completions.create(
                model="qwen/qwen3.8-27b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300,
                response_format={"type": "json_object"}
            )
            return json.loads(completion.choices[0].message.content)
        except Exception as e:
            return {"intent": "Error", "draft_reply": "Error generating reply.", "escalate": "Yes", "escalate_reason": str(e)}

if __name__ == "__main__":
    agent = AmazonSupportAgent()
    sample_msg = "My prime package was supposed to arrive yesterday but tracking still says shipped. Where is it??"
    print(f"\nTesting with message: {sample_msg}")
    result = agent.process_message(sample_msg)
    print(json.dumps(result, indent=2))
