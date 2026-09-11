import pandas as pd
import json
import os

def main():
    print("Loading raw dataset...")
    df = pd.read_csv('twcs.csv')
    
    brand_name = 'AmazonHelp'
    print(f"Filtering for {brand_name}...")
    
    brand_tweets = df[df['author_id'] == brand_name].copy()
    
    # Drop rows where Amazon isn't replying to anything
    brand_tweets = brand_tweets.dropna(subset=['in_response_to_tweet_id'])
    
    # Safely convert floats to string IDs without .0
    def float_to_str_id(x):
        try:
            return str(int(float(x)))
        except:
            return str(x)
            
    brand_tweets['in_response_to_str'] = brand_tweets['in_response_to_tweet_id'].apply(float_to_str_id)
    responded_to_ids = brand_tweets['in_response_to_str'].tolist()
    
    df['tweet_id_str'] = df['tweet_id'].astype(str)
    customer_tweets = df[df['tweet_id_str'].isin(responded_to_ids)]
    
    pairs = pd.merge(
        customer_tweets, 
        brand_tweets, 
        left_on='tweet_id_str', 
        right_on='in_response_to_str',
        suffixes=('_customer', '_brand')
    )
    
    print(f"Found {len(pairs)} Customer -> Brand turn pairs.")
    
    clean_pairs = []
    for _, row in pairs.iterrows():
        clean_pairs.append({
            'customer_tweet_id': row['tweet_id_customer'],
            'customer_text': row['text_customer'],
            'brand_tweet_id': row['tweet_id_brand'],
            'brand_text': row['text_brand']
        })
        
    out_df = pd.DataFrame(clean_pairs)
    out_df.to_csv(f'{brand_name}_pairs.csv', index=False)
    print(f"Saved to {brand_name}_pairs.csv")
    
    sample_size = 200
    if len(out_df) >= sample_size:
        sample_df = out_df.sample(n=sample_size, random_state=42)
        sample_df.to_csv(f'{brand_name}_golden_sample.csv', index=False)
        print(f"Saved {sample_size} sample to {brand_name}_golden_sample.csv for hand-labeling.")

if __name__ == '__main__':
    main()
