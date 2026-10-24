# Budget-Aware AI Decision Intelligence for SOC Queue Triage

Reproducibility code for the manuscript **“Budget-Aware AI Decision Intelligence from Cybersecurity Incident Data: Cost–Safety–Priority Trade-offs in SOC Queue Triage.”**

The experiment converts calibrated three-class incident probabilities and queue-relative expert priority into auditable `close`, `defer`, and `review now` actions under exact analyst-review budgets. It evaluates modeled cost, true-positive automatic closure, expert top-budget coverage, and NDCG rather than claiming a universally best policy.

## Repository contents

- `run_guide_experiment.py` — complete v0.6.0 training, calibration, policy, bootstrap, sensitivity, and figure pipeline.
- `test_guide_active_pipeline.py` — focused deterministic tests for the active pipeline.
- `guide_experiment_requirements.txt` — bounded Python dependencies.
- `RESEARCH_CHARTER.md` — hypotheses, novelty boundary, and frozen protocol.
- `CITATION.cff` — citation metadata.

Raw GUIDE files, incident Parquet files, and generated results are intentionally excluded because of their size and source-data terms.

## Data prerequisites

Prepare these four local inputs before running the experiment:

1. `guide_train_incidents.parquet`
2. `guide_test_incidents.parquet`
3. `GUIDE_Test_Queue_Rankings.csv`
4. `guide_incident_build_report.json`

The first two files are the incident-level tables produced from the official Microsoft GUIDE train and test CSV files by the validated preprocessing stage. The rankings file is joined only for expert-priority evaluation. Do not commit raw or derived GUIDE data to a public repository until its license and redistribution conditions have been confirmed.

## Installation

Python 3.12 was used for the reported experiment.

```powershell
py -m pip install -r .\guide_experiment_requirements.txt
```

## Full experiment

```powershell
py .\run_guide_experiment.py `
  --train-parquet "C:\path\guide_train_incidents.parquet" `
  --test-parquet "C:\path\guide_test_incidents.parquet" `
  --rankings "C:\path\GUIDE_Test_Queue_Rankings.csv" `
  --build-report "C:\path\guide_incident_build_report.json" `
  --output-dir "C:\path\guide_results_v06" `
  --threads 4
```

The run writes an auditable result directory and a `guide_results_bundle.zip`. The frozen manuscript run completed in 1,768 seconds on a four-thread Windows workstation.

## Verification

```powershell
py -m pytest -q .\test_guide_active_pipeline.py
```

Every run records configuration, package versions, source hashes, data fingerprints, bootstrap seeds, result tables, and generated research figures. The primary run used 2,000 queue-clustered bootstrap repetitions and a declared 864-configuration sensitivity grid.

## Frozen evidence summary

- Training incidents: 730,778
- Test incidents: 303,220
- Expert-ranked incidents: 9,980 across 499 queues
- HGB macro-F1: 0.699
- Expert top-budget coverage gain: 4.81–14.31 percentage points
- NDCG gain: 7.33–10.91 percentage points
- Temperature-scaling reduction in true-positive automatic closure: 0.50–0.77 percentage points
- Joint-policy increase in worst-case normalized modeled cost: 0.00338–0.00464

These results establish a cost–safety–priority frontier. They do not establish universal dominance, real-dollar savings, or live-SOC causal effects.

## License and citation

No source-data license is granted by this repository. Users must obtain GUIDE from its official source and comply with the provider's terms. Cite the accompanying manuscript and the GUIDE dataset paper; `CITATION.cff` contains project citation metadata.
