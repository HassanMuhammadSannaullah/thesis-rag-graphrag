"""
Find where Walter Payton passage appears in the full list.
"""
import json

# Load the record
with open('data/hybridqa/original/dev.jsonl', encoding='utf-8') as f:
    for line in f:
        if '00153f694413a536' in line:
            record = json.loads(line)
            break

print("Searching all 50 passages for Walter Payton...")
print(f"Total passages: {len(record['linked_passages'])}\n")

for i, p in enumerate(record['linked_passages'], 1):
    link = p['link']
    text = p.get('text', '')
    
    # Check if this is the Walter Payton article
    if '/wiki/Walter_Payton' in link:
        print(f"✓ FOUND Walter Payton passage at position {i}/50")
        print(f"  Link: {link}")
        print(f"  Text length: {len(text)} chars")
        
        if 'Jerry' in text:
            print(f"  ✓ Contains 'Jerry'!")
            idx = text.find('Jerry')
            start = max(0, idx - 50)
            end = min(len(text), idx + 50)
            print(f"\n  Context: ...{text[start:end]}...")
        else:
            print(f"  ✗ Does NOT contain 'Jerry'")
        
        print(f"\n  ⚠️  THIS PASSAGE IS POSITION {i}, BUT ONLY FIRST 30 ARE INCLUDED!")
        break
