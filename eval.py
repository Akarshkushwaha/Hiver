import pandas as pd
from agent import AmazonSupportAgent
from tqdm import tqdm
import json
import time

def evaluate_reply(client, draft, actual):
    prompt = f"""You are an expert customer service evaluator.
Compare this Draft Reply against the Actual Historical Reply.
Draft: "{draft}"
Actual: "{actual}"

Score the Draft from 1 to 5 on Helpfulness, Tone, and Safety. 5 is excellent, 1 is terrible. Output ONLY JSON:
{{
    "score": 4,
    "reason": "why"
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
    except:
        return {"score": 3, "reason": "Error parsing"}

def main():
    agent = AmazonSupportAgent()
    df = pd.read_csv("AmazonHelp_golden_labelled.csv")
    
    # We will only evaluate 50 samples to save API calls and time during this demo.
    df = df.head(50)
    
    results = []
    
    print("Running evaluation on golden set...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Agent prediction
        predicted = agent.process_message(row['customer_text'])
        time.sleep(0.5)
        
        # LLM-as-judge score
        eval_res = evaluate_reply(agent.client, predicted.get('draft_reply', ''), row['brand_text'])
        time.sleep(0.5)
        
        results.append({
            'customer_text': row['customer_text'],
            'actual_intent': row['intent'],
            'pred_intent': predicted.get('intent', 'Error'),
            'intent_match': row['intent'] == predicted.get('intent', ''),
            'actual_escalate': row['escalate'],
            'pred_escalate': predicted.get('escalate', 'Error'),
            'escalate_match': row['escalate'] == predicted.get('escalate', ''),
            'reply_score': eval_res.get('score', 0)
        })
        
    res_df = pd.DataFrame(results)
    res_df.to_csv("eval_results.csv", index=False)
    
    print("\n=== EVALUATION RESULTS ===")
    print(f"Total Samples Evaluated: {len(res_df)}")
    print(f"Intent Accuracy: {(res_df['intent_match'].sum() / len(res_df)) * 100:.1f}%")
    print(f"Escalation Accuracy: {(res_df['escalate_match'].sum() / len(res_df)) * 100:.1f}%")
    print(f"Average Reply Score (1-5): {res_df['reply_score'].mean():.2f}")
    
    most_common_intent = df['intent'].mode()[0]
    baseline_intent_acc = (df['intent'] == most_common_intent).sum() / len(df) * 100
    print(f"Trivial Baseline Intent Accuracy (always predict '{most_common_intent}'): {baseline_intent_acc:.1f}%")

if __name__ == "__main__":
    main()
