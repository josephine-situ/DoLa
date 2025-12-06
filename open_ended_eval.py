import json
import pandas as pd
import re
import os

# --- Configuration ---
# The evaluation file
file_path = 'results/open_evals/gemini_2_flash_adaptive.json'
# The predictions file (which model is best for each question)
predictions_file = "data/vicuna_eval/dec_vic.csv"

# Mapping from prediction names to actual model answer filenames
MODEL_NAME_MAP = {
    "vanilla": "vanilla_p_0_9",
    "dola_all_t1.0": "dola_dynamic",
    "dola_s24_t1.0": "dola_24",
}

def load_predictions(filepath):
    """Reads predictions from CSV file (one per line, in order of question_id)"""
    predictions = {}
    if not os.path.exists(filepath):
        print(f"Warning: Predictions file not found: {filepath}")
        return predictions
    
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            # Skip header if present
            if line_num == 0 and line == "0":
                continue
            # question_id = line_num (0-indexed), so we need to adjust
            q_id = line_num
            predictions[q_id] = line
    
    return predictions

# 1. Load the JSON data
try:    
    with open(file_path, 'r') as f:
        data = json.load(f)
    print("File loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    data = []

# 2. Load predictions (best model for each question)
print("Loading predictions...")
predictions = load_predictions(predictions_file)

# 3. Define the regex pattern
# Matches: "Rating", optional colon, optional spaces, "[[", number, "]]"
# Examples: "Rating [[8]]", "Rating: [[ 8.5 ]]"
rating_pattern = re.compile(r'Rating[:\s]*\[\[\s*(\d+(?:\.\d+)?)\s*\]\]', re.IGNORECASE)

structured_data = []

if data:
    print("Processing evaluations...")
    
    for entry in data:
        # 1. Capture the question ID
        q_id = entry.get('question_id')
        
        # Track the best model's rating for this question
        best_model_rating = None
        best_model_name = None
        
        # 2. Iterate through ALL items in the dictionary
        for key, value in entry.items():
            
            # 3. SKIP the specific metadata keys (question_id)
            # We only want to process the model response keys
            if key == 'question_id':
                continue
            
            # Now we assume 'key' is the model name (e.g., 'dola_dynamic')
            model_name = key
            text_content = str(value)
            
            # 4. Extract the rating
            match = rating_pattern.search(text_content)
            
            if match:
                score = float(match.group(1))
                structured_data.append({
                    'question_id': q_id,
                    'model': model_name,
                    'rating': score
                })
                
                # Check if this is the predicted best model
                if q_id in predictions:
                    pred_model_name = predictions[q_id]
                    mapped_model_name = MODEL_NAME_MAP.get(pred_model_name, pred_model_name)
                    if model_name == mapped_model_name:
                        best_model_rating = score
                        best_model_name = model_name
        
        # Append the best predicted model's evaluation to the original entry
        if best_model_rating is not None and q_id in predictions and best_model_name is not None:
            # Add 'model' entry to the original data with the best model's full evaluation text
            for key, value in entry.items():
                if key == best_model_name:
                    entry['model'] = value
                    break
            structured_data.append({
                'question_id': q_id,
                'model': 'model',
                'rating': best_model_rating
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
        
        # 5. Save the modified JSON with predicted model evaluations
        output_json_path = 'results/open_evals/gemini_2_flash_eval_with_model.json'
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\nModified JSON saved to: {output_json_path}")
        
        # 6. Save the statistics
        stats_output_path = 'results/open_evals/model_stats.csv'
        os.makedirs(os.path.dirname(stats_output_path), exist_ok=True)
        stats.to_csv(stats_output_path)
        print(f"Statistics saved to: {stats_output_path}")
    else:
        print("\nNo ratings found. Check that the JSON has an 'evaluations' key and strings contain 'Rating [[x]]'.")

else:
    print("No data to process.")