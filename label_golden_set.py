import pandas as pd
import os
import json
import time
from groq import Groq
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

INTENTS = [
    "Delivery/Shipping Issue",
    "Returns/Refunds",
    "Account/Billing Issue",
    "Product/Item Defect",
    "Digital Services (Video/Kindle)",
    "Other/General Inquiry"
]

def label_row(text):
    prompt = f"""You are an expert customer service analyst for Amazon.
Analyze the following customer tweet and classify it into one of the following intents:
{json.dumps(INTENTS)}

Also, decide if this tweet needs to be escalated to a human agent, or if it can be auto-handled.
Escalate if: the customer is very angry, the issue is complex, or it involves a severe security/safety risk.
Do not escalate if: it's a routine tracking question, simple refund policy question, or general complaint.

Customer Tweet: "{text}"

Output your response in valid JSON format ONLY:
{{
    "intent": "exact string from the list above",
    "escalate": "Yes" or "No",
    "escalate_reason": "brief reason why"
}}
"""
    try:
        completion = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=150,
            response_format={"type": "json_object"}
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        print("Error:", e)
        return {"intent": "Other/General Inquiry", "escalate": "Yes", "escalate_reason": f"Error parsing: {e}"}

def main():
    df = pd.read_csv("AmazonHelp_golden_sample.csv")
    results = []
    print("Labeling 200 samples using Groq...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        res = label_row(row['customer_text'])
        res['customer_text'] = row['customer_text']
        res['brand_text'] = row['brand_text']
        results.append(res)
        time.sleep(1) # Prevent rate limits
        
    out_df = pd.DataFrame(results)
    out_df = out_df[['customer_text', 'intent', 'escalate', 'escalate_reason', 'brand_text']]
    out_df.to_csv("AmazonHelp_golden_labelled.csv", index=False)
    print("Saved to AmazonHelp_golden_labelled.csv")

if __name__ == "__main__":
    main()
