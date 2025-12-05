import json
import pandas as pd
import re

# 1. Load the JSON data
file_path = 'results/open_evals/gemini_eval.json'

try:
    with open(file_path, 'r') as f:
        data = json.load(f)
    print("File loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    data = []

# 2. Define the regex pattern
# Matches: "Rating", optional colon, optional spaces, "[[", number, "]]"
# Examples: "Rating [[8]]", "Rating: [[ 8.5 ]]"
rating_pattern = re.compile(r'Rating[:\s]*\[\[\s*(\d+(?:\.\d+)?)\s*\]\]', re.IGNORECASE)

structured_data = []

if data:
    print("Processing evaluations...")
    
    for entry in data:
        # Capture the question ID for reference
        q_id = entry.get('question_id')
        
        # Access the nested 'evaluations' dictionary
        # We use .get({}) to return an empty dict if 'evaluations' is missing
        evaluations = entry.get('evaluations', {})
        
        for model_name, text_output in evaluations.items():
            # Ensure text_output is a string
            text_content = str(text_output)
            
            # Extract the rating using regex
            match = rating_pattern.search(text_content)
            
            if match:
                score = float(match.group(1))
                structured_data.append({
                    'question_id': q_id,
                    'model': model_name,
                    'rating': score
                })

    # 3. Create the DataFrame
    df = pd.DataFrame(structured_data)

    if not df.empty:
        print("\nDataFrame Head (Parsed Data):")
        print(df.head())

        # 4. Calculate Statistics (Average and Std Dev)
        stats = df.groupby('model')['rating'].agg(['mean', 'std'])
        
        # Rename columns for clarity
        stats = stats.rename(columns={'mean': 'Average Rating', 'std': 'Standard Deviation'})
        
        # Sort by Average Rating
        stats = stats.sort_values(by='Average Rating', ascending=False)

        print("\nModel Performance Statistics:")
        print(stats)
    else:
        print("\nNo ratings found. Check that the JSON has an 'evaluations' key and strings contain 'Rating [[x]]'.")

else:
    print("No data to process.")