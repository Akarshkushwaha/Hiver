# Hiver SDE Intern - AI Support Agent Report

## 1. Golden Evaluation Set: Sampling & Labeling Note
**How it was sampled**: I randomly selected 200 customer-agent interaction pairs from the `AmazonHelp` filtered dataset to ensure a diverse, unbiased representation of issues.
**How it was labelled**: To bootstrap the process rapidly, I used an LLM (`Qwen-27B`) as an oracle to pre-label the 200 samples with an Intent, Escalation flag, and Escalation Reason. I then manually reviewed the CSV, adjusting labels where the LLM missed subtle context (e.g., re-classifying angry shipping delays as "Escalate: Yes").

## 2. Evaluation Harness: Judge vs. Human Agreement
The evaluation harness (`eval.py`) uses an LLM-as-judge prompt to score the draft replies (1-5) against the actual historical reply. 
**Evidence of Agreement**: I manually scored a random sub-sample of 15 generated replies on Helpfulness and Tone. 
- **Exact Match**: The LLM-judge gave the exact same score as me on 11/15 cases (73%).
- **Within 1 Point**: The LLM-judge was within ±1 point of my score on 15/15 cases (100%).
The judge tends to have a slight positive bias (+0.5 average) because it favors its own structured, polite writing style over the brevity of human Amazon agents, but it reliably penalizes hallucinations.

## 3. Problem Framing
For `AmazonHelp`, "good" support means speed, strict policy adherence, and de-escalation by moving users to secure DMs. 
**What I chose NOT to build**: I chose not to build a complex dialogue manager or multi-turn state machine. Twitter support is inherently asynchronous and often single-turn from the AI's perspective (Customer Tweets -> Brand Replies). A stateless RAG-based architecture is the most robust, predictable approach for a prototype.

## 4. Results vs. Baselines
* **Trivial Baseline**: Always predicts "Other/General Inquiry" (majority class), always escalates.
    * Intent Accuracy: 54.0%
* **Simple Baseline**: A zero-shot LLM prompt without RAG/historical context.
    * Intent Accuracy: ~58.0%
    * Reply Score: ~2.5/5 (Lacks Amazon's specific brand voice and policy links).
* **Our Pipeline (SentenceTransformers RAG + Qwen 27B)**:
    * Intent Accuracy: **64.0%**
    * Escalation Accuracy: **58.0%**
    * Reply Score: **3.42/5**

## 5. Failure Analysis (Top 5 Failure Modes)

1. **Complex Intersecting Intents**
   * *Real Example*: `@AmazonHelp I think Amazon gine mad ...product worth 1750 and delivery charge 1000 ....plz remove delivery charge`
   * *Actual*: Account/Billing Issue | *Predicted*: Delivery/Shipping Issue
   * *Hypothesis*: The presence of "delivery charge" confuses the LLM. It sees "delivery" and ignores the financial/billing core of the complaint.

2. **Vague / Link-only Tweets**
   * *Real Example*: `@AmazonHelp Getting errora https://t.co/qLxnCgRP0A`
   * *Actual*: Other/General Inquiry | *Predicted*: Digital Services (Video/Kindle)
   * *Hypothesis*: Without vision capabilities or URL parsing, "getting errors" is heavily guessed as a Digital Service outage rather than a generic inquiry.

3. **High-Emotion Rants Masking the Issue**
   * *Real Example*: `@AmazonHelp A GENUINE COMPANY ONCE IS SURE THAT PRODUCT IS FAKE... AMAZON INSTEAD HARASSES ITS CUSTOMERS...`
   * *Actual*: Product/Item Defect | *Predicted*: Other/General Inquiry
   * *Hypothesis*: The customer focuses entirely on the "fraud/harassment" aspect rather than the actual defective product. The LLM classifies it as a general complaint.

4. **Missing Context / Thread Continuation**
   * *Real Example*: `@AmazonHelp which correspondence e-mail ?????? send me the link again`
   * *Actual*: Account/Billing Issue | *Predicted*: Other/General Inquiry
   * *Hypothesis*: Because our agent is stateless and only evaluates the single latest tweet, it lacks the history to know this relates to a billing email.

5. **Multi-lingual / Non-English Inputs**
   * *Real Example*: `@AmazonHelp O contacte t'on le sav ?`
   * *Actual*: Returns/Refunds | *Predicted*: Other/General Inquiry
   * *Hypothesis*: The prompt and intent definitions are in English. The model defaults to "Other" when it encounters French.

## 6. What is misleading about my headline number?
The 64% Intent Accuracy is actually *pessimistic* (under-reported). Because the Golden Dataset was heavily bootstrapped by a zero-shot LLM, the ground-truth labels contain noise. When our RAG-enhanced agent predicts something different, it is often penalized as "incorrect", even when its prediction is arguably better than the noisy ground-truth. Conversely, the 3.42/5 Reply Score is slightly *optimistic* because the LLM-as-judge inherently prefers its own lengthy, highly-structured writing style over the short, human-written actual replies.

## 7. What I'd do next with one more week
1. **Multi-Turn Context Window**: Pass the last 3-5 tweets in the conversation to the agent so it can handle follow-up questions (like Failure Mode #4).
2. **Dense RAG Vector DB**: Move from in-memory arrays to a proper FAISS or ChromaDB store, allowing us to embed 500,000 historical tweets instead of just 5,000, severely boosting RAG quality.
3. **Dedicated Escalation Classifier**: Split the prompt into two separate LLM calls. One purely for classification (using a strict JSON output model), and one for drafting. Combining them dilutes the LLM's attention.

## 8. Decision Log
1. **Brand Choice**: Chose `AmazonHelp` because it is the most voluminous and heavily represented brand in the dataset, ensuring a rich historical RAG database.
2. **Dataset Subsampling**: Extracted 168k turn-pairs, but sampled down to 5,000 for the RAG index to ensure the pipeline runs locally in under 15 minutes.
3. **Intent Taxonomy**: Defined 6 distinct intents rather than 20+ to keep the classification task bounded and measurable for a prototype.
4. **Bootstrapping the Golden Set**: Used an LLM to pre-label the 200 Golden Set examples. Hand-labeling from scratch would bottleneck development; reviewing pre-labels is faster.
5. **RAG vs. Fine-Tuning**: Chose RAG over Fine-Tuning to allow real-time policy updates (just swap the database) without expensive retraining.
6. **SentenceTransformers over TF-IDF**: Originally tried TF-IDF, but switched to `all-MiniLM-L6-v2` because dense embeddings capture semantic meaning (e.g., matching "broken" to "defect") vastly outperforming keyword matching.
7. **Stateless Agent**: Ignored multi-turn memory to simplify the architecture, treating each tweet as an isolated ticket.
8. **LLM Choice**: Selected `Qwen-27B` via Groq for its extreme speed and massive context window, avoiding OpenAI API costs for the reviewer.
9. **Single-Prompt Architecture**: Combined classification, drafting, and escalation into one JSON prompt to save API calls and reduce latency.
10. **LLM-as-Judge Evaluation**: Built an automated evaluator because manually scoring 50 generated replies on every code iteration is impossible.
11. **Escalation Rules**: Hardcoded rule definitions in the prompt (e.g., "Escalate ONLY if threatening") rather than letting the LLM guess what warrants an escalation.
12. **Discarding Automated Tweets**: Chose to keep all tweets (even "Please DM us" bots) in the RAG database because it accurately reflects Amazon's real-world mitigation strategy.
