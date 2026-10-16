"""
chunk_retrieval.py -- step 1 of the lean agentic pipeline.

What this file does (no LLM calls, no extra dependencies):

  1. CHUNK   the paper's (relevant) sections into numbered chunks
             (one sentence each; each table row becomes its own chunk).
  2. SEARCH  chunks per taxonomy parameter with BM25 (keyword ranking) plus a
             "unit match" bonus: a chunk containing "195 W" is a strong candidate
             for a parameter whose taxonomy unit is W, even if it never says
             "laser power".
  3. MEASURE recall@k against the rows your existing fixed pipeline already
             verified (agentic_extraction_results.json), so you can see whether
             retrieval finds the right chunk BEFORE building the extraction
             step on top of it. Also reports how many tokens retrieval saves.

Usage:
  python chunk_retrieval.py --demo
  python chunk_retrieval.py paper.xml \
      --triage triage_result.json \
      --results agentic_extraction_results.json \
      --vocab vocab_tree.json

Later steps import build_chunks / ChunkRetriever / render_chunks from here.
"""
import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass


# =====================================================================
# 1. Chunking
# =====================================================================
@dataclass
class Chunk:
    id: str        # "c0007" -- what the model will cite instead of copying a quote
    idx: int       # position in the chunk list (document order)
    section: str
    text: str            # what the model sees (table rows carry caption + column headers)
    kind: str = "text"   # "text" | "table"
    quote: str = ""      # verbatim source string for this chunk (== text, except table
                         # rows where it is just the row cells, without the synthetic header)


def approx_tokens(text):
    """Rough token estimate (~4 chars/token). Good enough for before/after comparisons."""
    return max(1, len(text) // 4)


_ODD_SPACES = re.compile("[\u00a0\u2000-\u200a\u202f\u205f\u3000]")
_INVISIBLE = re.compile("[\u00ad\u200b-\u200d\ufeff]")


def normalize_spaces(text):
    """Publisher XML is full of non-breaking / thin spaces ('500\u00a0\u03bcm', 'et\u00a0al.').
    They break sentence splitting and exact value matching, so they become plain
    spaces (and soft hyphens / zero-width characters are dropped). The SAME
    normalisation must be applied to the text used for verification."""
    return _INVISIBLE.sub("", _ODD_SPACES.sub(" ", text))


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[\(])")
_ABBREV_END = re.compile(
    r"(?:\bet al|\be\.g|\bi\.e|\bFigs?|\bEqs?|\bRefs?|\bvs|\bapprox|\bca|\bcf|\bNo|\bDr|\bSec)\.$",
    re.IGNORECASE,
)


def split_sentences(text):
    """Regex sentence splitter that doesn't break on 'et al.', 'Fig.', 'e.g.' etc."""
    merged = []
    for part in _SENT_SPLIT.split(text):
        if merged and _ABBREV_END.search(merged[-1]):
            merged[-1] += " " + part
        else:
            merged.append(part)
    return merged


def _hard_split(s, max_chars):
    """Split an over-long piece at ';' / ',' / space so no chunk explodes in size."""
    out = []
    while len(s) > max_chars:
        cut = s.rfind("; ", 0, max_chars)
        if cut < max_chars * 0.4:
            cut = s.rfind(", ", 0, max_chars)
        if cut < max_chars * 0.4:
            cut = s.rfind(" ", 0, max_chars)
        if cut <= 0:
            cut = max_chars - 1
        out.append(s[:cut + 1].strip())
        s = s[cut + 1:].strip()
    if s:
        out.append(s)
    return out


def _iter_blocks(text):
    """Splits a section's text into ('text', paragraph) and ('table', [lines]) blocks.

    triage_pipeline joins every text node with '\\n' (inline tags like <sub>/<i>
    create extra breaks mid-sentence), so newlines are NOT sentence boundaries:
    plain lines are re-joined with spaces. Only the [TABLE...]/Columns:/Row:
    blocks emitted by TablePreservingExtractor are kept as line groups."""
    para, table = [], None
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("[TABLE"):
            if para:
                yield ("text", " ".join(para))
                para = []
            if table:
                yield ("table", table)
            table = [s]
        elif table is not None and (s.startswith("Columns:") or s.startswith("Row:")):
            table.append(s)
        else:
            if table is not None:
                yield ("table", table)
                table = None
            if s:
                para.append(s)
    if table:
        yield ("table", table)
    if para:
        yield ("text", " ".join(para))


def _table_row_chunks(lines):
    """Returns (prefix, [row strings]). One chunk per table row is built from
    these: the model sees prefix + row (caption + column headers keep a cell's
    meaning when read on its own), while the verbatim quote is just the row."""
    caption, cols, rows = "", "", []
    for l in lines:
        if l.startswith("[TABLE"):
            caption = l[len("[TABLE"):].lstrip(":").strip().rstrip("]").strip()
        elif l.startswith("Columns:"):
            cols = l[len("Columns:"):].strip()
        elif l.startswith("Row:"):
            rows.append(l[len("Row:"):].strip())
    head = "TABLE" + (f" {caption[:120]}" if caption else "")
    prefix = f"[{head}] Columns: {cols[:200]} | Row: "
    return prefix, [r for r in rows if r]


def build_chunks(sectioned, section_names=None, max_chars=600, min_chars=30):
    """sectioned: {section_name: text} from triage_pipeline.strip_markup_sectioned.
    section_names: optional iterable of exact section names to keep (see
    pick_section_names). The pre-heading 'body' bucket is always skipped."""
    keep = set(section_names) if section_names is not None else None
    chunks = []

    def add(section, text, kind, quote=None):
        chunks.append(Chunk(id=f"c{len(chunks):04d}", idx=len(chunks),
                            section=section, text=text, kind=kind,
                            quote=text if quote is None else quote))

    for name, text in sectioned.items():
        if name == "body" or (keep is not None and name not in keep):
            continue
        for kind, payload in _iter_blocks(normalize_spaces(text)):
            if kind == "table":
                prefix, rows = _table_row_chunks(payload)
                for row in rows:
                    for piece in _hard_split(row, int(max_chars * 1.5)):
                        add(name, prefix + piece, "table", quote=piece)
                continue
            sents = []
            for s in split_sentences(payload):
                s = s.strip()
                if not s:
                    continue
                if sents and len(sents[-1]) < min_chars:   # glue tiny fragments forward
                    sents[-1] += " " + s
                else:
                    sents.append(s)
            for s in sents:
                for piece in _hard_split(s, max_chars):
                    add(name, piece, "text")
    return chunks


def _norm_section(name):
    """Same idea as triage_pipeline._normalize_section_name (kept local so this
    file has no imports from the rest of the project)."""
    cleaned = name.strip()
    m = re.match(r"^\[SECTION:\s*(.*?)\]$", cleaned)
    if m:
        cleaned = m.group(1)
    return re.sub(r"^\s*\d+(\.\d+)*\.?\s*", "", cleaned).strip().lower()


def pick_section_names(triage, sectioned):
    """Maps triage['relevant_sections'] (LLM guesses) onto real section names,
    with the same fuzzy matching and fallbacks as get_relevant_text()."""
    real = [k for k in sectioned if k != "body"]
    guesses = (triage or {}).get("relevant_sections") or []
    # Fallback (no guesses / no matches): everything except noise sections such as
    # References or Acknowledgments. (get_relevant_text keeps them; here they would
    # only add ~thousands of tokens of retrieval noise.)
    try:
        from triage_pipeline import _is_noise_section
    except ImportError:
        _is_noise_section = lambda n: any(k in n.lower() for k in (
            "reference", "acknowledg", "declaration", "supplementary", "credit", "keywords"))
    fallback = [k for k in real if not _is_noise_section(k)] or real
    if not guesses:
        return fallback
    picked = []
    for g in guesses:
        gn = _norm_section(g)
        if not gn:
            continue
        for name in real:
            if name in picked:
                continue
            nn = _norm_section(name)
            if gn == nn or gn in nn or nn in gn:
                picked.append(name)
                break
    return picked or fallback


def render_chunks(chunks):
    """Text block to put in a prompt. The model cites the [cXXXX] ids."""
    return "\n".join(f"[{c.id}] ({c.section}) {c.text}" for c in chunks)


# =====================================================================
# 2. Search: BM25 + unit match
# =====================================================================
_STOP = {"the", "of", "a", "an", "and", "or", "in", "on", "for", "to", "at",
         "by", "with", "is", "are", "was", "were", "be", "as", "per"}


def tokenize(text):
    """Lowercase word tokens; drops stopwords; crude plural stripping so
    'speeds' matches 'speed'. Keeps things like 'd50' and 'tm' intact."""
    out = []
    for t in re.findall(r"[^\W_]+", text.lower()):
        if t in _STOP:
            continue
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.append(t)
    return out


class BM25:
    """Plain BM25 over a small corpus (one paper's chunks)."""
    def __init__(self, docs_tokens, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs_tokens]
        self.dl = [len(d) for d in docs_tokens]
        self.N = len(docs_tokens)
        self.avgdl = (sum(self.dl) / self.N) if self.N else 1.0
        df = Counter()
        for c in self.tf:
            df.update(c.keys())
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def scores(self, weighted_terms):
        """weighted_terms: {term: weight}. Returns one score per document."""
        out = [0.0] * self.N
        for i, tf in enumerate(self.tf):
            norm = self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl)
            s = 0.0
            for term, w in weighted_terms.items():
                f = tf.get(term, 0)
                if f:
                    s += w * self.idf.get(term, 0.0) * f * (self.k1 + 1) / (f + norm)
            out[i] = s
        return out


# --- taxonomy parameter -> (bare name, unit) -> unit regex ---
def split_param(param):
    """'Laser Power (W)' -> ('Laser Power', 'W'); no trailing (...) -> (name, '')."""
    m = re.match(r"^(.*?)\s*\(([^)]*)\)\s*$", param.strip())
    return (m.group(1).strip(), m.group(2).strip()) if m else (param.strip(), "")


_UNIT_ATOMS = {
    "w", "mw", "kw", "mm", "cm", "m", "nm", "pm", "\u00b5m", "\u03bcm", "um",
    "s", "ms", "\u00b5s", "min", "h", "hr", "hz", "khz", "mhz", "ppm", "rpm",
    "pa", "kpa", "mpa", "gpa", "bar", "psi", "\u00b0c", "\u00b0", "k", "j", "kj",
    "kg", "g", "mg", "\u00b5g", "l", "ml", "\u00b5l", "pl", "nl", "v", "kv", "mv",
    "a", "ma", "\u00b5a", "n", "kn", "\u03c9", "gm", "mol", "cells", "%", "vol%", "wt%",
    "mj", "mn", "mbar", "px", "cycles", "v%", "%td",
}


def _units_are_real(unit):
    """True only if EVERY atom of the unit is a known physical unit. This is what
    keeps enum-style annotations out: '(Y/N)', '(Ar/N2)', '(HT/HIP/None)',
    '(Laser/Blade)', '(GMAW/GTAW/PAW/CMT)' all contain a non-unit atom and are
    rejected, while 'mm/s', 'mW/cm\u00b2', 'W/m\u00b7K', 'g/10min', 'cells/mL' pass."""
    atoms = [a for a in re.split(r"[/\u00b7*\u221a]", unit) if a.strip()]
    if not atoms:
        return False
    for a in atoms:
        a = re.sub(r"^\d+", "", a.strip())                        # '10min' -> 'min'
        if len(a) > 2:
            a = re.sub(r"\d+$", "", a)                            # 'mm3'/'mm^3' -> 'mm' ('N2' stays 'N2')
        a = re.sub(r"[\^\u00b3\u00b2\-\u2212]+$", "", a)         # 'mm^', 'mm\u00b3' -> 'mm'
        if a.lower() not in _UNIT_ATOMS:
            return False
    return True


def unit_regex(unit):
    """Regex matching '<number> <unit>' for a taxonomy unit string, tolerant of
    micro-sign variants (u / \u00b5 / \u03bc), a missing space, and 'mm^3' vs 'mm3' vs 'mm\u00b3'.
    Returns None for anything that is not a real unit (see _units_are_real), so
    categorical parameters like 'Support Required (Y/N)' get no unit logic."""
    u = (unit or "").strip()
    if not u or len(u) > 16 or " " in u or not _units_are_real(u):
        return None
    parts = []
    for ch in u:
        if ch in "\u00b5\u03bc":
            parts.append("[\u00b5\u03bcu]")
        elif ch == "^":
            continue
        elif ch == "\u00b3":
            parts.append("3")
        elif ch == "\u00b2":
            parts.append("2")
        elif ch == "/":
            parts.append(r"\s?/\s?")
        else:
            parts.append(re.escape(ch))
    if not parts:
        return None
    return re.compile(r"\d\s?" + "".join(parts) + r"(?![A-Za-z])")


UNIT_BONUS = 0.6   # added to a chunk's 0..1 BM25 score when its unit pattern matches


class ChunkRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self._bm25 = BM25([tokenize(c.text) for c in chunks])
        self._has_digit = [bool(re.search(r"\d", c.text)) for c in chunks]

    def search_param(self, param, synonyms=(), k=5, use_unit=True):
        """Top-k chunks for ONE taxonomy parameter string like 'Laser Power (W)'.
        Returns [(Chunk, score)]. Score = BM25 (normalised to 0..1) + UNIT_BONUS
        if the chunk contains a number with this parameter's unit. If the
        parameter has a unit, chunks with no digit at all are down-weighted
        (they can't contain the value)."""
        name, unit = split_param(param)
        terms = {t: 1.0 for t in tokenize(name)}
        for syn in synonyms:
            for t in tokenize(syn):
                terms.setdefault(t, 0.7)
        bm = self._bm25.scores(terms) if terms else [0.0] * len(self.chunks)
        top = max(bm) if bm else 0.0
        mx = top if top > 0 else 1.0
        urx = unit_regex(unit) if use_unit else None
        scored = []
        for i, c in enumerate(self.chunks):
            s = bm[i] / mx
            if urx is not None:
                if urx.search(c.text):
                    s += UNIT_BONUS
                elif not self._has_digit[i]:
                    s *= 0.3
            if s > 0:
                scored.append((s, i))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [(self.chunks[i], s) for s, i in scored[:k]]

    def retrieve_for_skill(self, skill, per_param_k=3, max_chunks=40, expand=0, use_unit=True):
        """The chunk set to send in ONE skill's extraction call.

        Every parameter in the skill votes for its top `per_param_k` chunks;
        chunks are ranked by summed score and the best `max_chunks` are kept
        (this caps prompt size no matter how many parameters the skill has).
        expand=1 also pulls in the neighbouring chunk (same section) of each hit.
        skill: anything with .param_sections {section: [param strings]} and
        .synonyms {bare_param_name: [alt phrasings]} (i.e. CategorySkill).
        Returns (selected_chunks_in_document_order, {param: [chunk ids]})."""
        syn_by_bare = {k.lower(): v for k, v in (getattr(skill, "synonyms", None) or {}).items()}
        weight = defaultdict(float)
        per_param = {}
        for params in skill.param_sections.values():
            for p in params:
                bare, _ = split_param(p)
                hits = self.search_param(p, syn_by_bare.get(bare.lower(), ()),
                                         k=per_param_k, use_unit=use_unit)
                per_param[p] = [c.id for c, _ in hits]
                for c, s in hits:
                    weight[c.idx] += s
        keep = set(sorted(weight, key=lambda i: (-weight[i], i))[:max_chunks])
        if expand:
            for i in list(keep):
                for d in range(1, expand + 1):
                    for j in (i - d, i + d):
                        if 0 <= j < len(self.chunks) and self.chunks[j].section == self.chunks[i].section:
                            keep.add(j)
        return [self.chunks[i] for i in sorted(keep)], per_param


# =====================================================================
# 3. Evaluation: does retrieval find the chunk your fixed pipeline used?
# =====================================================================
def _norm(s):
    return re.sub(r"\s+", " ", str(s or "").lower()).strip()


def _wordset(s):
    return set(re.findall(r"[^\W_]+", s.lower()))


def find_gold_chunks(row, chunks, min_overlap=0.6):
    """Pseudo-gold: chunks that contain the row's value AND are mostly made of
    words from the row's verified quote. Returns None if the row has no quote
    (e.g. review table cells) -- those can't be aligned, so they're reported
    separately instead of silently counted as misses."""
    value, quote = _norm(row.get("value")), _norm(row.get("quote"))
    if not value or not quote:
        return None
    qwords = _wordset(quote)
    gold = set()
    for c in chunks:
        if value not in _norm(c.text):
            continue
        cw = _wordset(c.text)
        if cw and len(cw & qwords) / len(cw) >= min_overlap:
            gold.add(c.id)
    return gold


def evaluate(chunks, results, skills_by_code, ks=(1, 3, 5, 10),
             per_param_k=3, max_chunks=40, show_misses=10):
    retriever = ChunkRetriever(chunks)
    all_tokens = sum(approx_tokens(c.text) for c in chunks)
    print(f"\nChunks: {len(chunks)} (~{all_tokens} tokens) "
          f"from {len({c.section for c in chunks})} sections")

    # collect verified rows + gold chunks once
    rows, unaligned, outside = [], 0, 0
    for key, r in results.items():
        code = r.get("skill") or key.split("::")[0]
        skill = skills_by_code.get(code)
        for row in r.get("parameters", {}).get("verified", []):
            gold = find_gold_chunks(row, chunks)
            if gold is None:
                unaligned += 1
            elif not gold:
                outside += 1
            else:
                rows.append((code, skill, row, gold))
    total = len(rows) + unaligned + outside
    print(f"Verified rows in results: {total} | scored: {len(rows)} | "
          f"no quote (skipped): {unaligned} | quote not in selected chunks (skipped): {outside}")
    if not rows:
        print("Nothing to score.")
        return

    max_k = max(ks)
    for label, use_unit in (("BM25 only", False), ("BM25 + unit match", True)):
        hit_at = {k: 0 for k in ks}
        misses = []
        for code, skill, row, gold in rows:
            param = row.get("parameter", "")
            bare, _ = split_param(param)
            syns = ()
            if skill is not None:
                syns = {k.lower(): v for k, v in (skill.synonyms or {}).items()}.get(bare.lower(), ())
            ids = [c.id for c, _ in retriever.search_param(param, syns, k=max_k, use_unit=use_unit)]
            first = next((i + 1 for i, cid in enumerate(ids) if cid in gold), None)
            for k in ks:
                if first is not None and first <= k:
                    hit_at[k] += 1
            if first is None:
                misses.append((code, param, row.get("value"), (row.get("quote") or "")[:110]))
        line = "  ".join(f"@{k}: {100 * hit_at[k] / len(rows):5.1f}%" for k in ks)
        print(f"\n[{label}] per-parameter recall  {line}")
        if use_unit and misses and show_misses:
            print(f"  Not found in top {max_k} ({len(misses)} rows), first {min(show_misses, len(misses))}:")
            for code, param, value, quote in misses[:show_misses]:
                print(f"    {code} | {param} = {value} | \"{quote}\"")

    # skill-level: what would actually be sent to the model
    print(f"\nSkill-level retrieval (per_param_k={per_param_k}, max_chunks={max_chunks}):")
    cache = {}
    hits = 0
    for code, skill, row, gold in rows:
        if skill is None:
            continue
        if code not in cache:
            cache[code] = retriever.retrieve_for_skill(skill, per_param_k, max_chunks)[0]
        if gold & {c.id for c in cache[code]}:
            hits += 1
    scored = sum(1 for _, s, _, _ in rows if s is not None)
    for code, sel in cache.items():
        toks = sum(approx_tokens(c.text) for c in sel)
        print(f"  {code}: {len(sel)} chunks, ~{toks} tokens "
              f"({100 * toks / all_tokens:.0f}% of all selected-section chunks)")
    if scored:
        print(f"  gold chunk present in the skill's chunk set: {hits}/{scored} rows "
              f"({100 * hits / scored:.1f}%)")


# =====================================================================
# 4. CLI
# =====================================================================
def _load_skills(vocab_path):
    from agentic_pipeline import build_skill_library, build_subtype_skill_library
    skills = dict(build_skill_library(vocab_path))
    skills.update(build_subtype_skill_library(vocab_path))
    return skills


def _demo():
    @dataclass
    class DemoSkill:
        code: str
        param_sections: dict
        synonyms: dict

    sectioned = {
        "body": "Journal of Demo DOI 10.0000/xyz",
        "Introduction": "Additive manufacturing has grown fast. Smith et al. reported "
                        "good results with SLM in 2019.",
        "Experimental": "Ti-6Al-4V powder with a D50 of 32 \u00b5m was used.\n"
                        "The LPBF process was run with laser power of 195 W and a\n"
                        "scan speed of 1100 mm/s. Layer thickness was set to 30 \u00b5m; "
                        "hatch spacing was 90 \u00b5m.\n"
                        "[TABLE: Process parameters]\nColumns: Sample | Power | Speed\n"
                        "Row: A | 200 W | 900 mm/s\nRow: B | 220 W | 1000 mm/s",
        "Results": "Relative density reached 99.4 % after HIP.",
    }
    skill = DemoSkill(
        code="PBF",
        param_sections={"Process": ["Laser Power (W)", "Scan Speed (mm/s)",
                                    "Layer Thickness (\u00b5m)", "Hatch Spacing (\u00b5m)",
                                    "D50 (\u00b5m)", "Relative Density (%)"]},
        synonyms={"Scan Speed": ["scanning speed", "laser scan velocity"],
                  "Laser Power": ["irradiation power", "beam power"]},
    )
    chunks = build_chunks(sectioned)
    print("--- chunks ---")
    print(render_chunks(chunks))
    retriever = ChunkRetriever(chunks)
    print("\n--- top 2 for each PBF parameter ---")
    for p in skill.param_sections["Process"]:
        bare, _ = split_param(p)
        syn = {k.lower(): v for k, v in skill.synonyms.items()}.get(bare.lower(), ())
        for c, s in retriever.search_param(p, syn, k=2):
            print(f"{p:<26} {c.id} score={s:.2f}  {c.text[:70]}")
    results = {"PBF": {"skill": "PBF", "parameters": {"verified": [
        {"parameter": "Laser Power (W)", "value": "195",
         "quote": "The LPBF process was run with laser power of 195 W and a scan speed of 1100 mm/s."},
        {"parameter": "Scan Speed (mm/s)", "value": "1100",
         "quote": "The LPBF process was run with laser power of 195 W and a scan speed of 1100 mm/s."},
        {"parameter": "D50 (\u00b5m)", "value": "32",
         "quote": "Ti-6Al-4V powder with a D50 of 32 \u00b5m was used."},
    ]}}}
    evaluate(chunks, results, {"PBF": skill}, ks=(1, 3, 5))


def main():
    ap = argparse.ArgumentParser(description="Chunk a paper, run BM25+unit retrieval, measure recall@k.")
    ap.add_argument("paper", nargs="?", help="paper markup file (xml/html)")
    ap.add_argument("--demo", action="store_true", help="run on a built-in sample, no files needed")
    ap.add_argument("--triage", default="triage_result.json", help="triage output (default triage_result.json)")
    ap.add_argument("--results", default="agentic_extraction_results.json",
                    help="output of the existing fixed pipeline, used as pseudo-gold")
    ap.add_argument("--vocab", default="vocab_tree.json")
    ap.add_argument("--ks", default="1,3,5,10")
    ap.add_argument("--per-param-k", type=int, default=3)
    ap.add_argument("--max-chunks", type=int, default=40)
    ap.add_argument("--all-sections", action="store_true", help="ignore triage's relevant_sections")
    ap.add_argument("--show-misses", type=int, default=10)
    args = ap.parse_args()

    if args.demo or not args.paper:
        return _demo()

    from triage_pipeline import strip_markup_sectioned
    with open(args.paper, encoding="utf-8") as f:
        sectioned = strip_markup_sectioned(f.read())

    triage = None
    if not args.all_sections and os.path.exists(args.triage):
        triage = json.load(open(args.triage))
    names = pick_section_names(triage, sectioned)
    print(f"Sections used ({len(names)}): {names}")
    chunks = build_chunks(sectioned, names)

    if not os.path.exists(args.results):
        print(f"\n{args.results} not found -- run agentic_pipeline.py on this paper first "
              f"to get pseudo-gold rows. Showing chunk stats only.")
        print(f"Chunks: {len(chunks)} (~{sum(approx_tokens(c.text) for c in chunks)} tokens)")
        return
    results = json.load(open(args.results))
    evaluate(chunks, results, _load_skills(args.vocab),
             ks=tuple(int(x) for x in args.ks.split(",")),
             per_param_k=args.per_param_k, max_chunks=args.max_chunks,
             show_misses=args.show_misses)


if __name__ == "__main__":
    main()
