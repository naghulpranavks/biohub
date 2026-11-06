import json
from pathlib import Path

f2 = Path(r"D:\Downloads_2\biohub-geometric-fusion.ipynb")
with open(f2, "r", encoding="utf-8") as f:
    nb2 = json.load(f)

def print_cell(idx):
    c = nb2['cells'][idx]
    print("="*80)
    print(f"CELL {idx}")
    print("="*80)
    print("".join(c.get("source", [])))

print("--- CELL 0 ---")
print_cell(0)
print("--- CELL 1 ---")
print_cell(1)
print("--- CELL 10 ---")
print_cell(10)
