# Full Retrain Execution Plan — clean data, verified contracts

**Date:** 2026-09-21
**Trigger:** all datasets (CIC-2018 CSV+PCAP, CIC-2017 CSV+PCAP, CTU-13) to be re-provided complete.
**Goal:** every model retrained on data that is verifiably intact, under one frozen split, with
results that survive the questions that invalidated three earlier runs.

Companion reports: `09_data_integrity_audit.md` (what is damaged), `10_contract_and_model_consistency.md`
(what disagrees). This file is the ordered plan.

---

## 0. What changed today, and why a full retrain is required

Five defects were found and fixed. Each one independently invalidates every checkpoint trained
before it, which is why this is a full retrain and not an incremental one.

| # | Defect | Evidence | Consequence if unfixed |
|---|---|---|---|
| 1 | **Epoch seconds 1000x too small.** Every pandas adapter did `astype("int64")/1e9`, but pandas 3 returns `datetime64[us]`, not `[ns]`. | `fri_16` parsed as 1970-01-18; span 43 s instead of 11.96 h | A 12-hour capture collapsed into ~22 two-second windows instead of ~21,600. Every host trajectory meaningless. |
| 2 | **CIC-2017 timestamps silently zeroed.** `utc=True` yields a tz-aware dtype; casting it raised, and the adapter's bare `except` set all timestamps to 0. | 0 valid timestamps from a 100k-row read | Entire corpus silently unusable while appearing to load fine. |
| 3 | **Focal loss was binary logic on 6 classes.** `at[targets==1]=alpha` weighted Benign 0.75 and C2 0.25. | per-class acc `{0:1.0, 1..5:0.0}`, macro F1 0.158 | TGNE's category head collapsed to predicting Benign for everything. |
| 4 | **Blank padding rows became records.** CIC-2017 Thursday-WebAttacks is 63% empty rows. | 288,602 records with `src_ip='nan'`, ts `-9.22e9` | Phantom hosts on a window grid stretching to 1677 AD. |
| 5 | **Snapshot builder hoarded Python lists.** Columnar store was compact once finalized but not while building. | run died with spill file at 0 bytes | Full-density training impossible; two hard laptop freezes. |

Fixes landed: `data_unification/time_utils.py` (one shared `to_epoch_seconds`), row-guards in the CIC
adapters, damped/clipped multi-class focal alpha in `bita/train.py`, numpy-backed
`TrajectoryStoreBuilder`, per-file/per-day record streaming.

---

## 1. Acceptance gate for newly provided data

Run **before** any training. Nothing proceeds until this passes.

```bash
python3 scripts/freeze_splits.py            # re-inventories every capture, rewrites the lock
python3 -m pytest tests/ -q                 # 118 tests
```

Per-corpus checks (detail and commands in `09_data_integrity_audit.md`):

1. **No file at exactly 1,048,576 rows** (2^20 = spreadsheet truncation).
2. **No `.~lock.*` files** anywhere in the data directories — presence means a spreadsheet has the
   file open and may rewrite/truncate it.
3. **Every CIC-2018 CSV has `Src IP`/`Dst IP`.** Currently only `tue_20` does; the other nine force
   `cic2018_adapter.py` to fabricate host IPs by row index, which is the original reason host
   trajectories were meaningless.
4. **Zero blank/padding rows** — check tail rows parse and carry a label.
5. **Timestamps land in the right year** and span the expected capture duration.
6. **PCAP:** all ten day directories present; record chains walkable; note pcapng files (`.pcap`
   extension but pcapng magic) which the bridge skips with a warning.

Acceptance is a checklist, not a vibe: record the measured numbers per file so a later change can
be diffed against them.

---

## 2. Execution order (strict — each stage gates the next)

### Stage A — TGNE (foundation; everything consumes its embeddings)

```bash
systemd-run --user --unit=tgne-retrain \
  -p MemoryMax=17G -p MemoryHigh=16G -p MemorySwapMax=0 \
  --working-directory="$PWD" --setenv=PYTHONPATH="$PWD" \
  -p StandardOutput=append:logs/tgne.out -p StandardError=append:logs/tgne.out \
  "$(which python3)" -u bita/train.py \
    --cic2017_dir "<CIC2017 TrafficLabelling dir>" \
    --cic2018_dir "<CIC2018 CSV dir>" \
    --ctu13_dir  "<CTU13 dir>" \
    --rows_per_file 300000 --gpu 0 --n_epoch 30 --patience 5 --data_name unified_clean
```

Accept only if: link-prediction AUC healthy on **both** transductive and inductive test, **and**
`Per-class Accuracy` is non-zero for more than just class 0. The previous run showed
`{0:1.0, 1..5:0.0}` — that is a collapsed head and must not be accepted again.

Then: `python3 scripts/verify_offline_live_parity.py --state <telemetry.jsonl>` must report
`0.000e+00` deviation, confirming the offline and live paths embed on the same graph (finding E7).

### Stage B — Branch A

Reads the new TGNE via `TGNE_CHECKPOINT_PATH`. Uses `--stride 1`, no row cap, `--spill-dir`.
Now emits a three-way file-disjoint split and a **held-out test scored once**, plus the
credibility gate on both val and test.

### Stage C — Branch B, then DeepOP (chained, single process)

`scripts/retrain_future_models_live.py --pcap-root ... --cic2018-csv-dir ... --spill-dir ...`
Branch B returns its best model and DeepOP conditions on **real `wdt.rollout()` output**, not the
oracle future (audit E1). Confirm the log line reads `DeepOP conditioning: Branch-B rollouts (E1 fixed)`.

---

## 3. Credibility gate — the thing that makes results reportable

`scripts/credibility_check.py` ports the v4 gate to the three-branch pipeline. It runs automatically
in both trainers. A run is **not reportable** if any fires:

| Check | Threshold | Why it invalidates |
|---|---|---|
| label churn | < 0.01 | nothing changes over the horizon; "forecast" = nowcast renamed |
| base rate | >0.95 or <0.05 | any ranking, including random noise, scores ~1.0 PR-AUC |
| host groups | < 5 | no bootstrap CI exists |
| persistence | > 0.99 | a no-change rule already wins |
| val samples | < 500 | checkpoint selection is noise |
| technique classes | < 2 | accuracy is meaningless |
| model vs persistence | lift <= 0 | the model beat nothing |

Three earlier runs reported **PR-AUC 0.9998** and were discarded solely because this class of check
caught them. Treat a gate failure as a data problem to fix, never as a number to report anyway.

---

## 4. Frozen split (already locked)

`data_unification/splits.lock.json`, rationale in `data_unification/split_policy.py`.

- **Unit = capture** (CSV day / CTU-13 scenario / PCAP day). Finer splitting leaks: stride-1 windows
  inside one capture are not independent (audit D5).
- **Stratified by measured attack fraction**, dealt round-robin 4:1:1. Attack rates span 0.00%
  (CIC-2017 Monday) to 65.6% (`wed_21`); an unstratified split can put an all-benign capture in test,
  which is the extreme-base-rate trap.
- **Test never trained on, never used for selection, scored once.**
- CTU-13 splits by scenario, holding out whole botnet families.

Current assignment: CIC-2018 8/1/1, CIC-2017 6/1/1, CTU-13 9/2/2, PCAP 8/1/1, with test attack
fractions 27.3% / 36.5% / 0.7–8.1% / 28.1%.

**Outstanding wiring TODO:** the trainers still compute their own splits. They must be changed to
read the lock (see `10_contract_and_model_consistency.md` for exact file:line sites). Until then the
lock documents intent but does not enforce it.

---

## 5. Known-remaining limitations to state alongside any result

These are not blockers but must be disclosed, not buried:

1. **CIC-2018 CSV host identity is fabricated for 9 of 10 days** unless the re-provided files include
   `Src IP`/`Dst IP`. Branch B/DeepOP avoid this by using PCAP; Branch A does not.
2. **263 PCAP files carry mid-file corruption**; the adapter keeps each file's valid prefix, so those
   hosts have truncated trajectories.
3. **No calibration split** for the three-branch models (v4 has one). Conformal intervals and
   post-hoc temperature scaling are therefore not available for Branch A/B/DeepOP.
4. **Branch A's val split was 73:1 smaller than train** in one measured run; the gate now flags
   ratios above 25:1.
5. **Better data does not guarantee a better model.** The dense CTU-13-only run reached 99.7%
   technique accuracy; a cleaner, broader corpus may score lower and still be more trustworthy.

---

## 6. Definition of done

- [ ] Acceptance gate passes on all newly provided data, numbers recorded
- [ ] `splits.lock.json` regenerated and trainers read it
- [ ] TGNE retrained; category head non-degenerate; parity check 0.000e+00
- [ ] Branch A retrained; held-out test scored once; gate clean
- [ ] Branch B + DeepOP retrained; DeepOP conditioned on real rollouts; gate clean
- [ ] 118 tests green at every stage
- [ ] Results reported **with** the limitations in §5 attached
