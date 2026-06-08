import json

# Load the question data
with open('data/hybridqa/original/dev.jsonl', encoding='utf-8') as f:
    lines = [json.loads(line) for line in f]

# Find the specific question
q = [x for x in lines if x['question_id'] == '00153f694413a536'][0]

passages = q['linked_passages']

# Check for Walter Payton passages
walter_passages = [p for p in passages if 'Walter' in str(p)]
print(f"Found {len(walter_passages)} passages mentioning Walter\n")

for i, p in enumerate(walter_passages):
    text = p['text']
    has_jerry = 'Jerry' in text
    print(f"{i+1}. {p['link']}")
    print(f"   Contains 'Jerry': {has_jerry}")
    if has_jerry:
        # Find the context around "Jerry"
        idx = text.find('Jerry')
        start = max(0, idx - 100)
        end = min(len(text), idx + 100)
        print(f"   Context: ...{text[start:end]}...")
    print()
