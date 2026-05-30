import json
import openai
import os
import time
import argparse

# Load dataset
def load_dataset(filepath):
    dataset = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            dataset.append(json.loads(line.strip()))
    return dataset

# Reconstruct text and mask
def reconstruct_and_mask(tokens, labels):
    reconstructed = []
    expected_masked = []
    
    for token, label in zip(tokens, labels):
        if label != 'O':
            reconstructed.append(token)
            expected_masked.append(f"[{label}]")
        else:
            reconstructed.append(token)
            expected_masked.append(token)
            
    return " ".join(reconstructed), " ".join(expected_masked)

TAG_ALIASES = {
    "NAME": "PERSON",
    "EMAIL_REDACTED": "EMAIL",
    "ADDRESS": "LOCATION"

    # Add other aliases here as needed
}

# Metrics-based evaluator
def evaluate_result(actual, expected, aliases):
    actual_tokens = actual.split()
    expected_tokens = expected.split()
    
    # Store token-level results for granular stats
    token_stats = []
    
    correct_masks = 0
    wrong_tag = 0
    unmasked = 0
    incorrect_masks = 0 # e.g. masked when it shouldn't have been
    skipped_tokens = 0
    
    # We iterate based on the longer sequence to calculate skipped/added tokens
    max_len = max(len(actual_tokens), len(expected_tokens))
    
    # Pad shorter list for iteration
    a_tokens = actual_tokens + [None] * (max_len - len(actual_tokens))
    e_tokens = expected_tokens + [None] * (max_len - len(expected_tokens))
    
    # Helper to check if two tags match (considering aliases)
    def tags_match(tag1, tag2):
        if tag1 == tag2:
            return True
        # Check alias in both directions
        if aliases.get(tag1) == tag2 or aliases.get(tag2) == tag1:
            return True
        return False

    for a, e in zip(a_tokens, e_tokens):
        if e is None: # Added token
            continue
        if a is None: # Skipped token
            skipped_tokens += 1
            token_stats.append({"tag": None, "status": "skipped"})
            continue
            
        is_expected_mask = e.startswith('[') and e.endswith(']')
        is_actual_mask = a.startswith('[') and a.endswith(']')
        
        if is_expected_mask:
            tag = e[1:-1]
            if is_actual_mask:
                a_tag = a[1:-1]
                if tags_match(tag, a_tag):
                    correct_masks += 1
                    token_stats.append({"tag": tag, "status": "correct"})
                else:
                    wrong_tag += 1
                    token_stats.append({"tag": tag, "status": "wrong_tag"})
            else:
                unmasked += 1
                token_stats.append({"tag": tag, "status": "unmasked"})
        else:
            # Expected raw token
            if is_actual_mask:
                incorrect_masks += 1
                token_stats.append({"tag": "NONE", "status": "incorrect_mask"})
            else:
                token_stats.append({"tag": None, "status": "ok"})
                
    pii_hidden = correct_masks + wrong_tag
    text_damage = incorrect_masks + skipped_tokens
    return correct_masks, wrong_tag, unmasked, incorrect_masks, pii_hidden, skipped_tokens, text_damage, token_stats

def run_tests(dataset_file, output_filename="test_results"):
    dataset = load_dataset(dataset_file)
    client = openai.OpenAI(base_url="http://localhost:8088", api_key="test")
    
    results = []
    # Tag statistics
    tag_stats = {}
    
    total_correct = 0
    total_wrong_tag = 0
    total_unmasked = 0
    total_incorrect = 0
    total_pii_hidden = 0
    total_skipped = 0
    total_text_damage = 0
    total_time = 0
    
    for i, item in enumerate(dataset, 1):
        text, expected = reconstruct_and_mask(item['tokens'], item['labels'])
        
        try:
            start_time = time.time()
            response = client.chat.completions.create(
                model="dummy-model",
                messages=[{"role": "user", "content": text}]
            )
            duration = time.time() - start_time
            total_time += duration
            
            actual = response.choices[0].message.content
            
            c, w, u, i_mask, pii_h, s, dmg, t_stats = evaluate_result(actual, expected, TAG_ALIASES)
            total_correct += c
            total_wrong_tag += w
            total_unmasked += u
            total_incorrect += i_mask
            total_pii_hidden += pii_h
            total_skipped += s
            total_text_damage += dmg
            
            # Update tag stats
            for ts in t_stats:
                tag = ts['tag']
                if tag:
                    if tag not in tag_stats:
                        tag_stats[tag] = {"correct": 0, "wrong_tag": 0, "unmasked": 0}
                    status = ts['status']
                    if status in tag_stats[tag]:
                        tag_stats[tag][status] += 1
            
            results.append({
                "input": text,
                "expected": expected,
                "actual": actual,
                "correct_masks": c,
                "wrong_tag": w,
                "unmasked": u,
                "incorrect_masks": i_mask,
                "pii_hidden": pii_h,
                "skipped_tokens": s,
                "text_damage": dmg,
                "duration_seconds": round(duration, 4)
            })
            print(f"Test {i} - Correct: {c}, Wrong Tag: {w}, Unmasked: {u}, Incorrect Mask: {i_mask}, PII Hidden: {pii_h}, Time: {duration:.4f}s")
        except Exception as e:
            print(f"Test {i} error: {e}")
            
    print(f"\nFinal Summary:")
    print(f"Total Correct Masks: {total_correct}")
    print(f"Total Wrong Tag: {total_wrong_tag}")
    print(f"Total Unmasked: {total_unmasked}")
    print(f"Total Incorrect Masks: {total_incorrect}")
    print(f"Total Skipped: {total_skipped}")
    print(f"Total Text Damage: {total_text_damage}")
    print(f"Total PII Hidden: {total_pii_hidden}")
    print(f"Total Processing Time: {total_time:.4f}s")
    print(f"Average Time Per Request: {total_time/len(dataset):.4f}s")
    
    print("\nEntity Statistics:")
    for tag, stats in tag_stats.items():
        print(f"  {tag}: Correct: {stats['correct']}, Wrong Tag: {stats['wrong_tag']}, Unmasked: {stats['unmasked']}")
            
    with open(f'{output_filename}.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_filename}.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test filtering proxy.")
    parser.add_argument("dataset", help="Path to the dataset file", nargs="?", default="demo_dataset.jsonl")
    parser.add_argument("-o", "--output", help="Output file name without extension", default="test_results")
    args = parser.parse_args()
    
    run_tests(args.dataset, args.output)
