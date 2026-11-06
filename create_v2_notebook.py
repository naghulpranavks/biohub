import json
from pathlib import Path

src_path = Path(r"D:\Downloads_2\biohub-final-notebook.ipynb")
dst_path = Path(r"D:\Downloads_2\biohub-final-notebook-v2.ipynb")

with open(src_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

source = nb["cells"][0]["source"]
modifications = []

# 1. Bidirectional weight: 0.15 -> 0.40
for i, line in enumerate(source):
    if 'os.environ["BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT"] = "0.15"' in line:
        source[i] = line.replace('"0.15"', '"0.40"')
        modifications.append(f"Line {i}: Set BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT to 0.40")
    if '"BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT": 0.15,' in line:
        source[i] = line.replace('0.15,', '0.40,')
        modifications.append(f"Line {i}: Updated _EXPECTED_NUMERIC for bidirectional weight to 0.40")

# 2. DeepCenter gap veto: 1 -> 0
for i, line in enumerate(source):
    if 'os.environ["BIOHUB_DEEPCENTER_GAP_VETO"] = "1"' in line:
        source[i] = line.replace('"1"', '"0"')
        modifications.append(f"Line {i}: Disabled BIOHUB_DEEPCENTER_GAP_VETO (set to 0)")

# 3. Velocity weight: default 0.5 -> 1.0, and add to header
for i, line in enumerate(source):
    if 'MOTION_RELINK_VELOCITY_WEIGHT = float(os.environ.get("BIOHUB_MOTION_RELINK_VELOCITY_WEIGHT", "0.5"))' in line:
        source[i] = line.replace('"0.5"', '"1.0"')
        modifications.append(f"Line {i}: Set MOTION_RELINK_VELOCITY_WEIGHT default to 1.0")

# Insert header env var if not present
for i, line in enumerate(source):
    if 'os.environ["BIOHUB_MOTION_RELINK_LEARNED_BONUS"]' in line:
        source.insert(i + 1, 'os.environ["BIOHUB_MOTION_RELINK_VELOCITY_WEIGHT"] = "1.0"\n')
        modifications.append(f"Inserted BIOHUB_MOTION_RELINK_VELOCITY_WEIGHT = '1.0' at line {i+1}")
        break

# 4. Non-Destructive Additive Relinker
# Replace the destructive 'edges = motion_edges' assignment
for i, line in enumerate(source):
    if 'stats["motion_relink_replaced_raw_edges"] = len(edges)' in line and i + 1 < len(source) and 'edges = motion_edges' in source[i+1]:
        relink_replacement = (
            '            # Non-destructive additive relinker: preserve 100% of ILP edges and divisions!\n'
            '            ilp_edges_set = {(int(e["source_id"]), int(e["target_id"])) for e in edges}\n'
            '            ilp_sources = {int(e["source_id"]) for e in edges}\n'
            '            ilp_targets = {int(e["target_id"]) for e in edges}\n'
            '            merged_edges = list(edges)\n'
            '            added_relink = 0\n'
            '            for me in motion_edges:\n'
            '                s_id = int(me["source_id"])\n'
            '                t_id = int(me["target_id"])\n'
            '                if (s_id, t_id) in ilp_edges_set:\n'
            '                    continue\n'
            '                if s_id not in ilp_sources and t_id not in ilp_targets:\n'
            '                    merged_edges.append(me)\n'
            '                    ilp_sources.add(s_id)\n'
            '                    ilp_targets.add(t_id)\n'
            '                    added_relink += 1\n'
            '            stats["motion_relink_replaced_raw_edges"] = added_relink\n'
            '            edges = merged_edges\n'
        )
        source[i] = relink_replacement
        source[i+1] = "" # clear the old edges = motion_edges line
        modifications.append(f"Line {i}-{i+1}: Replaced destructive relinker with Non-Destructive Additive Relinker")
        break

# Write out to new v2 notebook
with open(dst_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

print(f"SUCCESS: Created {dst_path}")
print(f"Total modifications made: {len(modifications)}")
for m in modifications:
    print(" ", m)
