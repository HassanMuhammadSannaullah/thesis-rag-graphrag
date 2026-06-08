import json

# Load the question data
with open('data/hybridqa/original/dev.jsonl', encoding='utf-8') as f:
    lines = [json.loads(line) for line in f]

# Find the specific question
q = [x for x in lines if x['question_id'] == '00153f694413a536'][0]

print(f"Question: {q['question']}")
print(f"Answer: {q['answer']}")
print(f"Total linked passages: {q['num_linked_passages']}")

passages = q['linked_passages']

# Check for Walter Payton passages
walter_passages = [p for p in passages if 'Walter' in str(p)]
print(f"\nPassages mentioning 'Walter': {len(walter_passages)}")

if walter_passages:
    print("\n--- First Walter Payton passage ---")
    print(f"Link: {walter_passages[0]['link']}")
    text = walter_passages[0]['text']
    print(f"Contains 'Jerry': {'Jerry' in text}")
    print(f"\nFirst 500 chars:\n{text[:500]}")

# Also check the parsed corpus
print("\n\n=== CHECKING PARSED CORPUS ===")
with open('data/hybridqa/parsed/baseline_corpus_v2.jsonl', encoding='utf-8') as f:
    for line in f:
        if '00153f694413a536' in line:
            data = json.loads(line)
            print(f"Parsed corpus linked passages: {len(data.get('linked_passages', []))}")
            break
