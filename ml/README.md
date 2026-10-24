# ml/

- `build_notebook.py`: the source of truth. Edit this, then run `python build_notebook.py` to regenerate `surecure_train.ipynb`.
- `surecure_train.ipynb`: run on Kaggle or Colab (GPU T4). Set `QUICK = True` for a 5-minute smoke test.
- `make_mock.py`: builds a tiny fake Capsure folder for testing the data cells without the real dataset.

Locked settings (do not change between runs): SEED, IMG, CACHE, FILL, ALPHA, DEV_BURSTS_PER_CLASS, preprocessing, ONNX input/output names.

## Eval-only mode (score saved weights, no training)

Use this to re-score the frozen model (version 7) with new evaluation cells, such as section 7b.

- Kaggle: Add Input → Notebook Output → `surecure-model`, version 7. Then, before the config cell, run
  `import os; os.environ["SC_EVAL_WEIGHTS"] = "auto"` (finds the first `best.pt` under `/kaggle/input`).
- Colab: `os.environ["SC_EVAL_WEIGHTS"] = "/content/drive/MyDrive/SureCure/runs/v7/surecure_model/best.pt"` (your path).

Training (EPOCHS = 0) and fine-tuning are skipped. Calibration, tests, the guard and the ONNX export still run.
Splits are the same as in training because they depend only on SEED.

## Section 7b: accuracy by form

`data/capsure_burst_forms.csv` labels every Capsure real burst (Val and Test merged, 375 bursts) as `box`,
`blister` or `loose` from one middle frame per burst; `sure = 0` marks 8 bursts (6 hard to tell, 2 single-frame Val bursts labelled from the shooting order without viewing). The notebook
carries a built-in copy, so nothing has to be attached; set `SC_FORMS_CSV` to use an edited file.
Outputs: `accuracy_by_form.json`, `test_bursts_by_form.csv`, `figures/accuracy_by_form.png`.

## Section 8c: guard fitted on real photos

The synthetic-fitted Mahalanobis guard scored AUROC 0.34 on an external blister video, because its score tracks
image domain (synthetic vs phone photo), not product identity. Section 8c fits the same guard on real dev photos
(2 bursts per product fit, 2 calibrate; Ledoit-Wolf shrinkage), and compares it with the synthetic version, energy
and max-softmax on a class-holdout test on real test photos. The highest mean class-holdout AUROC is shipped
(rule fixed before the run; `SC_GUARD_MODE` forces one). `meta.json` → `guard` tells the app which score and
threshold to use. Output: `guard_compare.json`, `figures/guard_compare.png`.

**Final (2026-10-24):** the rule picked the energy score (class-holdout AUROC 0.811). `SC_GUARD_MODE` now defaults
to `energy`, and the ONNX model has a third output, `energy`, computed in the graph, so the app only compares it with
`meta.json` → `guard.threshold`. Always ship `surecure.onnx` and `meta.json` from the same run: the threshold is
calibrated for that run.
