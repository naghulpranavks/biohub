import json

path = r'D:\Downloads_2\biohub-final-notebook.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

replacements = []

for cell_idx, cell in enumerate(nb.get('cells', [])):
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        for i, line in enumerate(source):
            # 1. DeepCenter safe division veto
            if 'os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_VETO"] = "1"' in line:
                source[i] = line.replace('"1"', '"0"')
                replacements.append((cell_idx, i, line.strip(), source[i].strip()))
            
            # 2. Minimum candidate retention
            if 'BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION' in line:
                if '"0.50"' in line:
                    source[i] = line.replace('"0.50"', '"0.35"')
                    replacements.append((cell_idx, i, line.strip(), source[i].strip()))
            
            # 3. Diagnostic checks
            if i > 0 and 'minimum_retention' in source[i-1] and '!= 0.5' in line:
                source[i] = line.replace('!= 0.5', '!= 0.35')
                replacements.append((cell_idx, i, line.strip(), source[i].strip()))
            if i > 0 and 'retention' in source[i-1] and '< 0.5' in line:
                source[i] = line.replace('< 0.5', '< 0.35')
                replacements.append((cell_idx, i, line.strip(), source[i].strip()))

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write('\n')

print(f'Total replacements made: {len(replacements)}')
for r in replacements:
    print(f'  Cell {r[0]}, line {r[1]}: {r[2]} -> {r[3]}')
