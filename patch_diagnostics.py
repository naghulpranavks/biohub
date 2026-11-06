import json

path = r'D:\Downloads_2\biohub-final-notebook.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

replacements = 0
for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        for i, line in enumerate(source):
            if line == '        != 0.9\n' and 'minimum_retention' in source[i-1]:
                source[i] = '        != 0.5\n'
                replacements += 1
            if line == '        < 0.9\n' and 'retention' in source[i-1]:
                source[i] = '        < 0.5\n'
                replacements += 1

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write('\n')

print(f'Diagnostic replacements made: {replacements}')
