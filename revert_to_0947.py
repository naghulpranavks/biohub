import json

path = r'D:\Downloads_2\biohub-final-notebook.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

source = nb['cells'][0]['source']
reverts = []

for i, line in enumerate(source):
    # 1. Restore DeepCenter safe division veto to 1
    if 'os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_VETO"] = "0"' in line:
        source[i] = line.replace('"0"', '"1"')
        reverts.append((i, "Re-enabled DeepCenter safe div veto (0 -> 1)"))

    # 2. Restore retention threshold to 0.50
    if 'BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION' in line:
        if '"0.35"' in line:
            source[i] = line.replace('"0.35"', '"0.50"')
            reverts.append((i, "Restored retention threshold (0.35 -> 0.50)"))

    # 3. Restore diagnostic checks to 0.5
    if i > 0 and 'minimum_retention' in source[i-1] and '!= 0.35' in line:
        source[i] = line.replace('!= 0.35', '!= 0.5')
        reverts.append((i, "Restored diagnostic assert (0.35 -> 0.5)"))
    if i > 0 and 'retention' in source[i-1] and '< 0.35' in line:
        source[i] = line.replace('< 0.35', '< 0.5')
        reverts.append((i, "Restored diagnostic check (0.35 -> 0.5)"))

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write('\n')

print(f"Total reverts applied: {len(reverts)}")
for r in reverts:
    print(f"  Line {r[0]}: {r[1]}")
