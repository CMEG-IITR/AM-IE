"""
Parses am_taxonomy_v4.html into a structured vocabulary:
{
  "BJT": {
     "L1": "Binder Jetting",
     "L2": {"BJT-M": "Metal Binder Jetting", ...},
     "params": {
        "Feedstock Properties — Powder": ["D10 (µm)", "D50 (µm)", ...],
        "Process Parameters — BJT Metal": [...],
        ...
     }
  },
  ...
}
Also produces a flat parameter -> unit lookup used later for regex building.
"""
import json, re
from bs4 import BeautifulSoup

SRC = "papers/am_taxonomy_v4.html"
soup = BeautifulSoup(open(SRC), "html.parser")

CAT_ORDER = ["BJT", "DED", "MEX", "MJT", "PBF", "SHL", "VPP", "OTHER"]

vocab = {c: {"L1": None, "L2": {}, "params": {}} for c in CAT_ORDER}

# --- L1 / L2 node cards ---
for card in soup.select(".node-card"):
    code_el = card.select_one(".node-code")
    name_el = card.select_one(".node-name")
    lvl_el = card.select_one(".lvl")
    if not code_el or not name_el:
        continue
    code = code_el.get_text(strip=True)
    name = name_el.get_text(strip=True)
    lvl = lvl_el.get_text(strip=True) if lvl_el else ""
    top = code.split("-")[0]
    if top not in vocab:
        continue
    if lvl == "L1":
        vocab[top]["L1"] = name
    elif lvl == "L2":
        vocab[top]["L2"][code] = name

# --- L4-L7 (and shared L9) parameter blocks: level-card + shared-block ---
def category_of_card(card):
    # walk up to find nearest cat-XXX class among ancestors
    for anc in card.parents:
        classes = anc.get("class", []) if hasattr(anc, "get") else []
        for c in classes:
            if c.startswith("cat-") and c[4:] in CAT_ORDER:
                return c[4:]
    return None

for card in soup.select(".level-card"):
    title_el = card.select_one(".lc-title")
    if not title_el:
        continue
    title = title_el.get_text(strip=True)
    tags = [t.get_text(strip=True) for t in card.select(".lc-tag")]
    cat = category_of_card(card)
    if cat is None:
        continue
    vocab[cat]["params"].setdefault(title, [])
    for t in tags:
        if t not in vocab[cat]["params"][title]:
            vocab[cat]["params"][title].append(t)

# --- shared blocks (L9 post-processing etc, apply to ALL categories) ---
shared = {}
for block in soup.select(".shared-block"):
    title_el = block.select_one(".shared-block-title")
    if not title_el:
        continue
    title = title_el.get_text(strip=True)
    tags = [t.get_text(strip=True) for t in block.select(".s-tag")]
    shared[title] = tags

for cat in vocab:
    vocab[cat]["params"]["_shared"] = shared

# --- flat parameter -> unit lookup ---
UNIT_RE = re.compile(r"\(([^)]+)\)\s*$")
flat = {}
for cat, data in vocab.items():
    for section, params in data["params"].items():
        for p in params:
            m = UNIT_RE.search(p)
            unit = m.group(1) if m else None
            name = UNIT_RE.sub("", p).strip()
            flat.setdefault(name, {"unit": unit, "raw": p, "sections": set()})
            flat[name]["sections"].add(section)

for k in flat:
    flat[k]["sections"] = sorted(flat[k]["sections"])

json.dump(vocab, open("vocab_tree.json", "w"), indent=2, ensure_ascii=False)
json.dump(flat, open("vocab_flat.json", "w"), indent=2, ensure_ascii=False)

print(f"Categories parsed: {len(vocab)}")
print(f"Flat parameter count: {len(flat)}")
print("Sample entries:")
for k in list(flat)[:8]:
    print(" -", k, "->", flat[k])
