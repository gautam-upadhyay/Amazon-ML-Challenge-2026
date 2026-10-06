"""Cross-encoder reranker for the uncertain pairs of the entity-resolution pipeline.

A multilingual transformer reads both records together ("name | address" vs
"name | address") and outputs a match probability. It is fine-tuned on labelled
training pairs, then scores held-out pairs (to tune/verify the blend), stress pairs
(France proxy: model trained on US only, scored on India) and test pairs.

usage:
  python rerank2.py run            # XLM-R: main model -> held-out + France test; stress model -> stress
  python rerank2.py run --with_iu  # ...then also India/US test pairs (only if time allows)
Outputs go to ./out/ as each step finishes:
  ce_heldout.parquet, ce_testfr.parquet, ce_stress.parquet (+ ce_testiu.parquet)
"""
import argparse
import math
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

HERE = os.path.dirname(os.path.abspath(__file__))
# DATA: input parquets written by s07_export_rerank.py (default: $ER_WORK/rerank_data)
# OUT:  output parquets read by s08b_blend2.py (default: $ER_WORK/rerank_out/xlmr)
# run_all.sh sets both via env vars; no manual copying needed.
DATA = os.environ.get("RERANK_DATA", os.path.join(HERE, "data"))
OUT  = os.environ.get("RERANK_OUT",  os.path.join(HERE, "out"))
DEV = "cuda"


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def encode(tok, df, max_len):
    enc = tok(df["text_a"].tolist(), df["text_b"].tolist(), truncation=True, max_length=max_len)
    return enc["input_ids"], enc.get("attention_mask")


def batches(ids, order, bs, pad_id):
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        seqs = [torch.tensor(ids[j], dtype=torch.long) for j in idx]
        x = pad_sequence(seqs, batch_first=True, padding_value=pad_id)
        yield idx, x, (x != pad_id).long()


def train(args, df, tag):
    log(f"[{tag}] training on {len(df):,} pairs with {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1).to(DEV)
    ids, _ = encode(tok, df, args.max_len)
    y = torch.tensor(df["label"].values, dtype=torch.float32)
    n = len(ids)
    hold = set(random.Random(0).sample(range(n), max(1, min(20000, n // 50))))
    tr_idx = [i for i in range(n) if i not in hold]
    va_idx = sorted(hold)
    steps = math.ceil(len(tr_idx) / args.batch) * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=DEV == "cuda")
    lossf = torch.nn.BCEWithLogitsLoss()
    step, t0 = 0, time.time()
    for ep in range(args.epochs):
        model.train()
        order = tr_idx[:]
        random.Random(ep).shuffle(order)
        # sort within chunks of 100 batches by length -> less padding, still shuffled
        chunk = args.batch * 100
        order = [j for k in range(0, len(order), chunk) for j in sorted(order[k:k + chunk], key=lambda j: len(ids[j]))]
        blocks = [order[k:k + args.batch] for k in range(0, len(order), args.batch)]
        random.Random(ep + 1).shuffle(blocks)
        for idx in blocks:
            seqs = [torch.tensor(ids[j], dtype=torch.long) for j in idx]
            x = pad_sequence(seqs, batch_first=True, padding_value=tok.pad_token_id).to(DEV)
            m = (x != tok.pad_token_id).long()
            with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
                logits = model(input_ids=x, attention_mask=m).logits.squeeze(-1)
            loss = lossf(logits.float(), y[idx].to(DEV))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sch.step()
            step += 1
            if step % 500 == 0:
                rate = step * args.batch / (time.time() - t0)
                eta = (steps - step) * args.batch / rate / 60
                log(f"[{tag}] step {step}/{steps} loss {loss.item():.4f} | {rate:.0f} pairs/s | ETA {eta:.0f} min")
    p = predict(model, tok.pad_token_id, [ids[j] for j in va_idx], args)
    yv = y[va_idx].numpy()
    ll = -np.mean(yv * np.log(np.clip(p, 1e-6, 1)) + (1 - yv) * np.log(np.clip(1 - p, 1e-6, 1)))
    acc = np.mean((p > 0.5) == (yv > 0.5))
    log(f"[{tag}] validation on {len(va_idx):,} held-back training pairs: logloss {ll:.4f}, accuracy {acc:.4f}")
    return model, tok


@torch.no_grad()
def predict(model, pad_id, ids, args):
    model.eval()
    order = sorted(range(len(ids)), key=lambda j: len(ids[j]))
    out = np.zeros(len(ids), dtype=np.float32)
    for idx, x, m in batches(ids, order, args.batch * 8, pad_id):
        with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
            logits = model(input_ids=x.to(DEV), attention_mask=m.to(DEV)).logits.squeeze(-1)
        out[idx] = torch.sigmoid(logits.float()).cpu().numpy()
    return out


def score(model, tok, name, args):
    df = pd.read_parquet(os.path.join(DATA, f"{name}_pairs.parquet"))
    if args.smoke:
        df = df.head(300)
    t = time.time()
    ids, _ = encode(tok, df, args.max_len)
    p = predict(model, tok.pad_token_id, ids, args)
    pd.DataFrame({"s1_id": df["s1_id"], "r_id": df["r_id"], "ce": p}).to_parquet(os.path.join(OUT, f"ce_{name}.parquet"), index=False)
    log(f"scored {name}: {len(df):,} pairs in {(time.time() - t) / 60:.1f} min -> out/ce_{name}.parquet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run"])
    ap.add_argument("--model", default="xlm-roberta-base")
    ap.add_argument("--max_len", type=int, default=96)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--train_limit", type=int, default=0, help="use at most this many training pairs (0 = all)")
    ap.add_argument("--with_iu", action="store_true", help="also score India/US test pairs at the end")
    ap.add_argument("--smoke", action="store_true", help="tiny CPU run to test the code path")
    args = ap.parse_args()
    global DEV
    if args.smoke:
        DEV = "cpu"
        args.train_limit, args.batch, args.model = 400, 16, "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    elif not torch.cuda.is_available():
        sys.exit("CUDA GPU not available - install a CUDA build of torch (see README).")
    else:
        log(f"GPU: {torch.cuda.get_device_name(0)}")
    os.makedirs(OUT, exist_ok=True)
    random.seed(0)
    torch.manual_seed(0)
    tr = pd.read_parquet(os.path.join(DATA, "train_pairs.parquet"))
    if args.train_limit:
        tr = tr.sample(n=min(args.train_limit, len(tr)), random_state=0)
    model, tok = train(args, tr.reset_index(drop=True), "main")
    score(model, tok, "heldout", args)
    score(model, tok, "testfr", args)
    iu_model = model
    us = tr[tr["country"] == "US"].reset_index(drop=True)
    smodel, stok = train(args, us, "stress")
    score(smodel, stok, "stress", args)
    del smodel
    torch.cuda.empty_cache() if DEV == "cuda" else None
    if args.with_iu:
        score(iu_model, tok, "testiu", args)
    log(f"done. Outputs written to {OUT}/")


if __name__ == "__main__":
    main()
