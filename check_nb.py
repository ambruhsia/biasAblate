import json, sys
sys.stdout.reconfigure(encoding="utf-8")
nb = json.load(open(r"C:/Users/rajpu/OneDrive/Desktop/mitigating bias from llms/bias-mitigation-in-llms/layer-trials.ipynb", encoding="utf-8"))
print("Total cells:", len(nb["cells"]))
for i, c in enumerate(nb["cells"]):
    src = c.get("source", "")
    if isinstance(src, list):
        src = "".join(src)
    first = src.split("\n")[0][:70] if src.strip() else "(empty)"
    print(f"{i:>2}  {c['id']:20}  {first}")
