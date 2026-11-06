import json

path = r'D:\Downloads_2\biohub-final-notebook.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

replacements_made = 0
for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        for i, line in enumerate(cell.get('source', [])):
            if 'BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION' in line:
                new_line = line.replace('"0.90"', '"0.50"').replace("'0.90'", "'0.50'")
                if line != new_line:
                    cell['source'][i] = new_line
                    replacements_made += 1

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write('\n')

print(f'Successfully updated! Replacements made: {replacements_made}')
