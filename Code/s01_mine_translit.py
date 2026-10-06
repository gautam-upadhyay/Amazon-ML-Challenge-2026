"""Mine Indic-script -> Latin token dictionaries from TRAINING ground-truth pairs only.

Names: when an Indic-script S2/S3 name and its matched Latin S1 name have the same
number of tokens they are aligned position by position; each Indic token maps to its
majority Latin counterpart. Addresses: Indic tokens (mostly state names) map to the
Latin S1-address tokens with high conditional probability and lift.
"""
import json
from collections import Counter, defaultdict

import polars as pl
from rapidfuzz import fuzz

from config import WORK_DIR, raw_path
from textnorm import IN_STATES, INDIC_RE, latin, raw_tokens, romanize


def read(split, src):
    return pl.read_csv(raw_path(split, src), separator="\t", quote_char=None, infer_schema=False)


def main():
    s1 = read("train", "1")
    r = pl.concat([read("train", "2"), read("train", "3")])
    gt = pl.read_csv(raw_path("train", "1").with_name("train_ground_truth.tsv"), separator="\t",
                     quote_char=None, infer_schema=False)
    pairs = (gt.drop_nulls("matched_entity_ids")
               .with_columns(pl.col("matched_entity_ids").str.split(","))
               .explode("matched_entity_ids")
               .rename({"source1_entity_id": "s1", "matched_entity_ids": "entity_id"}))
    r = r.filter(pl.col("business_name").str.contains(r"[ऀ-෿]")
                 | pl.col("business_address").fill_null("").str.contains(r"[ऀ-෿]"))
    j = (pairs.join(r, on="entity_id")
              .join(s1.select(pl.col("entity_id").alias("s1"), pl.col("business_name").alias("n1"),
                              pl.col("business_address").alias("a1")), on="s1"))
    print("pairs with Indic text:", j.height)

    name_counts = defaultdict(Counter)
    a_tok, a_pair, a_w = Counter(), defaultdict(Counter), Counter()
    n_pairs_addr = 0
    for n1, n2, a1, a2 in zip(j["n1"], j["business_name"], j["a1"], j["business_address"]):
        if n2 and INDIC_RE.search(n2) and not INDIC_RE.search(n1):
            t1 = [latin(t) for t in raw_tokens(n1)]
            t2 = raw_tokens(n2)
            if len(t1) == len(t2):
                for x, y in zip(t2, t1):
                    if INDIC_RE.search(x):
                        name_counts[x][y] += 1
        if a2 and a1 and INDIC_RE.search(a2):
            w1 = set(latin(t) for t in raw_tokens(a1.replace(",", " ")) if not t.isdigit())
            n_pairs_addr += 1
            a_w.update(w1)
            for t in set(t for t in raw_tokens(a2) if INDIC_RE.search(t)):
                a_tok[t] += 1
                a_pair[t].update(w1)

    def finalize(counts):
        out = {}
        for tok, c in counts.items():
            best, n = c.most_common(1)[0]
            if n >= 2 and n / sum(c.values()) >= 0.5:
                out[tok] = best
        return out

    name_dict = finalize(name_counts)
    unequal = [(n1, n2) for n1, n2 in zip(j["n1"], j["business_name"])
               if n2 and INDIC_RE.search(n2) and not INDIC_RE.search(n1)]
    for _ in range(3):
        extra = defaultdict(Counter)
        for n1, n2 in unequal:
            t1 = [latin(t) for t in raw_tokens(n1)]
            t2 = raw_tokens(n2)
            left1 = list(t1)
            left2 = []
            for x in t2:
                y = name_dict.get(x) if INDIC_RE.search(x) else latin(x)
                if y is not None and y in left1:
                    left1.remove(y)
                elif INDIC_RE.search(x) and x not in name_dict:
                    left2.append(x)
            if not left2 or not left1:
                continue
            if len(left1) == len(left2):
                for x, y in zip(left2, left1):
                    extra[x][y] += 1
            else:
                for x in left2:
                    rx = romanize(x)
                    best = max(left1, key=lambda y: fuzz.ratio(rx, y))
                    if fuzz.ratio(rx, best) >= 55:
                        extra[x][best] += 1
        new = {k: v for k, v in finalize(extra).items() if k not in name_dict}
        print("second-pass new name tokens:", len(new))
        if not new:
            break
        name_dict.update(new)
    addr_dict = {}
    for tok, n in a_tok.items():
        if n < 3:
            continue
        cands = []
        for w, k in a_pair[tok].items():
            p = k / n
            lift = p / (a_w[w] / n_pairs_addr)
            if p >= 0.7 and lift >= 3:
                cands.append((p * lift, w))
        cands.sort(reverse=True)
        words = {w for _, w in cands}
        states = [s for s in IN_STATES if set(s.split()) <= words]
        if states:
            addr_dict[tok] = max(states, key=len)
        elif cands:
            addr_dict[tok] = cands[0][1]
    print("name dict:", len(name_dict), "addr dict:", len(addr_dict))
    for k in list(name_dict)[:15]:
        print("  ", k, "->", name_dict[k])
    for k in list(addr_dict)[:15]:
        print("  ", k, "->", addr_dict[k])
    with open(WORK_DIR / "translit.json", "w") as f:
        json.dump({"name": name_dict, "addr": addr_dict}, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
