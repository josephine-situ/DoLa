import json
import os
import glob

# --- Configuration ---
# Folder containing all your answer files
answers_dir = "data/vicuna_eval/model_answer"
# The question file
question_file = "data/vicuna_eval/question.jsonl"
# The output file
output_file = "combined_all_models.json"

def load_jsonl_to_dict(filepath):
    """Reads a JSONL file and returns a dict: {question_id: answer_text}"""
    data_map = {}
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return data_map

    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip(): continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            q_id = item.get("question_id")
            
            # Extract the text answer regardless of format
            text = ""
            if "choices" in item:
                # FastChat format
                text = item["choices"][0]["turns"][0]
            elif "text" in item:
                # Standard DoLa format
                text = item["text"]
            elif "answer" in item:
                # Alternate DoLa format
                text = item["answer"]
            
            # Ensure ID is treated consistently (as integer) for sorting later
            if q_id is not None:
                data_map[int(q_id)] = text
                
    return data_map

def main():
    # 1. Find all model files
    # This looks for any .jsonl file in the answers directory
    model_files = glob.glob(os.path.join(answers_dir, "*.jsonl"))
    
    if not model_files:
        print(f"No answer files found in {answers_dir}!")
        return

    print(f"Found {len(model_files)} model files:")
    for f in model_files:
        print(f" - {os.path.basename(f)}")

    # 2. Load all answers into memory
    # structure: all_model_answers = { "model_name": {qid: text}, ... }
    all_model_answers = {}
    
    for filepath in model_files:
        # Use filename (without extension) as the model name key
        model_name = os.path.splitext(os.path.basename(filepath))[0]
        print(f"Loading {model_name}...")
        all_model_answers[model_name] = load_jsonl_to_dict(filepath)

    # 3. Load questions
    print("Loading questions...")
    questions = {}
    if os.path.exists(question_file):
        with open(question_file, 'r') as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                q_text = item.get("text", "")
                if not q_text and "turns" in item:
                    q_text = item["turns"][0]
                questions[int(item["question_id"])] = q_text

    # 4. Combine everything
    combined_data = []
    
    # Get all unique question IDs from the question file (or fallback to answer files)
    if questions:
        all_q_ids = sorted(questions.keys())
    else:
        # Fallback: collect all IDs seen in any model file
        all_ids_set = set()
        for m_data in all_model_answers.values():
            all_ids_set.update(m_data.keys())
        all_q_ids = sorted(list(all_ids_set))
    
    for q_id in all_q_ids:
        # Start entry with ID and Question
        entry = {
            "question_id": q_id,
            "question_text": questions.get(q_id, "Unknown Question")
        }
        
        # Add answer from EVERY model found
        for model_name, answer_data in all_model_answers.items():
            entry[model_name] = answer_data.get(q_id, "MISSING ANSWER")
            
        combined_data.append(entry)

    # 5. Output
    with open(output_file, 'w') as f:
        json.dump(combined_data, f, indent=2)
    
    print(f"\nSuccessfully combined {len(combined_data)} questions for {len(model_files)} models.")
    print(f"Saved to: {output_file}")
    
    # Preview
    print("\n--- PREVIEW (First Entry) ---")
    if combined_data:
        print(json.dumps(combined_data[0], indent=2))

if __name__ == "__main__":
    main()