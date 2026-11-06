import re
from pathlib import Path
from collections import Counter

text = Path(r"c:\Users\Naghul Pranav\Documents\Biohub_Cell_Tracking\geom_fusion_source.py").read_text(encoding="utf-8")
scores = re.findall(r'0\.9\d{2,4}', text)
print(Counter(scores).most_common(20))

# Print context around LB or public mentions
for line in text.splitlines():
    if any(k in line.lower() for k in ["public", "leaderboard", "lb", "0.94", "0.95"]):
        print(line[:120])
