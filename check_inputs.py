import re

with open(r'c:\Users\Naghul Pranav\Documents\Biohub_Cell_Tracking\nb_source.py', 'r', encoding='utf-8') as f:
    text = f.read()

models = re.findall(r'/kaggle/input/[^\s"\'\\]+', text)
for m in sorted(set(models)):
    print('Referenced input:', m)
