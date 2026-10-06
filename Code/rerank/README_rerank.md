# GPU reranker: run this on the laptop with the NVIDIA GPU

Fine-tunes a small multilingual transformer (Apache-2.0, 118M parameters) to judge
the pairs the main pipeline is unsure about, then scores them. You only run it and
send back the `out/` folder; everything else happens on the other machine.

## 1. Install (once, ~5 min)

Open a terminal (Command Prompt / PowerShell on Windows) **inside this folder**.

Check the GPU is visible to PyTorch:

```
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

- Prints `True` → go to step 2.
- Prints `False` or torch is missing → install the CUDA build of PyTorch:
  ```
  pip install torch --index-url https://download.pytorch.org/whl/cu121
  ```

Then install the rest:

```
pip install -r requirements.txt
```

## 2. Run (~1 hour, unattended)

```
python rerank.py all
```

- Downloads the model weights once (~470 MB) from Hugging Face.
- Prints progress every 500 steps with pairs/s and an ETA.
- Runs **main** first (train on all pairs, score held-out + test, ~40 min), then
  **stress** (train on US pairs only, score India pairs, ~20 min).

If the laptop is short on time, run `python rerank.py main` only: that alone is
enough to improve the submission; `stress` only decides whether to also apply it to France.

**If it runs out of GPU memory**, add `--batch 32`.
**If it is much slower than ~600 pairs/s**, add `--train_limit 400000`.

## 3. Send back

Zip the `out/` folder (about 40 MB) and send it back. It must contain:

- `ce_heldout.parquet`
- `ce_test.parquet`
- `ce_stress.parquet` (only if you ran `all` or `stress`)

The last lines of the terminal output (the "validation" lines) are also useful:
copy-paste them along with the zip.
