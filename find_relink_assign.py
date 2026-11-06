import json

path = r'D:\Downloads_2\biohub-final-notebook.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

source = nb['cells'][0]['source']
for i, line in enumerate(source):
    if 'motion_relink_replaced_raw_edges' in line and '=' in line and '0' not in line:
        print(f'Assignment at line {i}:')
        for j in range(max(0, i-5), min(len(source), i+10)):
            print(f'{j}: {repr(source[j])}')
