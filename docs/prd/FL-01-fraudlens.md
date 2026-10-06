# PRD FL-01: FraudLens

**Owner:** Seedy M. Jahateh
**Status:** Draft for build
**Origin:** rebuilds the CSC 450 final project "AI-Based Fraud Detection in Financial Transactions" (Fall 2024). Its code was never pushed and cannot be recovered; its design documents (supervised models plus an anomaly detector, explainability, a small API) carry forward.
**One line:** detect fraudulent card transactions on a heavily imbalanced public dataset, evaluate honestly with the metrics that matter for fraud, pick a decision threshold by cost, and serve the model behind a tested scoring API.

---

## 1. Problem

Fraud is rare. In the target dataset about 0.17% of transactions are fraudulent, so a model that predicts "legitimate" every time scores 99.8% accuracy and catches nothing. Most portfolio fraud projects report accuracy anyway, or tune on the test set, or leak future information into training. They look good and prove little.

FraudLens is built to avoid those mistakes and to show it: a time-respecting split, imbalance-aware metrics, a threshold chosen from explicit business costs, and a service that applies the same preprocessing at scoring time as at training time.

## 2. Goals

| ID | Goal | Measure |
|---|---|---|
| G1 | Honest evaluation | PR-AUC, recall at a fixed precision, and precision at a fixed recall, on a held-out final time slice the model never saw during development |
| G2 | Model comparison | Logistic regression baseline, random forest, gradient boosting and an isolation-forest anomaly detector, all on the same splits, with bootstrap confidence intervals |
| G3 | Cost-based decisions | The threshold is chosen to minimise expected cost from a documented cost model, and the result is reported in money terms |
| G4 | Explainability | Global and per-transaction feature contributions for the chosen model |
| G5 | A served model | A FastAPI `/score` endpoint with input validation, model versioning, tests and a latency budget |
| G6 | Reproducible | One command trains, evaluates and writes the report from a fixed seed |

## 3. Non-goals

- Real customer data or any personally identifiable information. The public dataset is already anonymised.
- Real-time streaming infrastructure (Kafka and the like). Batch training plus a synchronous scoring API is the scope.
- Claims about production performance. Results describe this dataset only.
- Deep learning as a requirement. An autoencoder is an optional stretch goal, not a deliverable.

## 4. Data

**Primary dataset:** "Credit Card Fraud Detection" (Machine Learning Group, Université Libre de Bruxelles), published on Kaggle.

- 284,807 transactions over two days in September 2013, with 492 fraud cases (about 0.172%).
- Features `V1`–`V28` are PCA components (the originals are confidential), plus `Time` (seconds since the first transaction), `Amount` and the label `Class`.
- **Access:** download it manually from Kaggle (it needs a free account) and place it at `data/raw/creditcard.csv`. The file is git-ignored and never committed. Record its SHA-256 in `data/README.md`.
- **License:** read the license on the dataset page, record it in `data/README.md`, and follow its terms.

**Limitation to state in the README:** because the features are PCA components, explanations describe components (for example "V14"), not human-readable features. That limits what explainability can say; acknowledge it rather than hide it.

## 5. Evaluation protocol (the heart of the project)

- **EV-1 Time-based split.** Sort by `Time`. Use the first 60% for training, the next 20% for validation (model selection and threshold choice) and the last 20% as the **final test set**. The test set is used exactly once, at the end. Random splits are not used for headline results because they leak future patterns into training.
- **EV-2 Primary metrics.** PR-AUC (average precision); recall at 90% precision; precision at 80% recall; and the confusion matrix at the chosen threshold. Accuracy is reported only to show why it is misleading.
- **EV-3 Secondary metrics.** ROC-AUC (with a note that it flatters imbalanced problems), Brier score and a calibration curve.
- **EV-4 Uncertainty.** 1,000-sample bootstrap 95% confidence intervals on the test metrics. Model differences are only claimed when the intervals support them.
- **EV-5 Imbalance handling.** Compare class weights, random undersampling and SMOTE. Resampling is applied inside the training folds only, never to validation or test data.
- **EV-6 Leakage checks.** Scalers and any resampling are fitted inside a scikit-learn `Pipeline` on training data only. A test asserts that no test-set row influences any fitted parameter.
- **EV-7 Cost model.** A configurable cost table in `config.yaml`, with defaults: a missed fraud costs the transaction amount; a false alarm costs a fixed review cost (for example $5). The threshold is chosen on the validation set to minimise total expected cost, then reported on the test set together with the money saved compared with "flag nothing" and "flag everything".

## 6. Functional requirements

### 6.1 Data and features (`fraudlens/data.py`, `fraudlens/features.py`)

- **FR-1** Load and validate the CSV against a schema: column names, dtypes, no missing values, and `Class` in {0, 1}. Fail loudly on mismatch.
- **FR-2** Time-based split per EV-1, written as a reproducible index file.
- **FR-3** Features: scaled `Amount` (log1p), hour-of-day derived from `Time`, and the PCA components as given. Feature code is shared by training and serving, never duplicated.

### 6.2 Models (`fraudlens/models.py`)

- **FR-4** Logistic regression (with class weights) as the baseline.
- **FR-5** Random forest.
- **FR-6** Gradient boosting: scikit-learn `HistGradientBoostingClassifier` (XGBoost or LightGBM optional; if used, pin versions).
- **FR-7** Isolation forest, trained on legitimate training transactions only, as the unsupervised comparison from the original design. Report it with the same metrics, so the comparison is honest.
- **FR-8** Hyperparameter search on the validation slice only (randomised search with a fixed seed and a small, documented budget).
- **FR-9** Probability calibration (isotonic or sigmoid) for the chosen model, fitted on validation data.

### 6.3 Evaluation and report (`fraudlens/evaluate.py`, `reports/`)

- **FR-10** `python -m fraudlens.train` trains every model and writes `artifacts/<model>/<version>/` (the model, the feature list, metrics, a data hash, the code version and the seed).
- **FR-11** `python -m fraudlens.evaluate` writes `reports/metrics.csv` (every model and metric, with CIs), `reports/pr_curves.png`, `reports/calibration.png`, `reports/cost_curve.png` and `reports/report.md`.
- **FR-12** The README results table is generated from `reports/metrics.csv`, never typed by hand.

### 6.4 Explainability (`fraudlens/explain.py`)

- **FR-13** Global importance via permutation importance on the validation set (SHAP optional).
- **FR-14** Per-transaction explanation: the top 5 contributing features for any scored transaction, returned by the API.

### 6.5 Scoring service (`service/`)

- **FR-15** FastAPI app with `POST /score`: input is a transaction with all model features; output is `fraud_probability`, `decision` (`flag` or `allow`, at the cost-chosen threshold), `threshold`, `model_version` and the top contributing features.
- **FR-16** Pydantic validation: a wrong type, a missing feature or a negative `Amount` returns 422 with a clear message.
- **FR-17** `GET /health` and `GET /model` (version, training date, test PR-AUC, threshold).
- **FR-18** The service loads one versioned artifact at startup and never retrains.
- **FR-19** Structured JSON logging of request id, latency, decision and model version, with no raw feature values in the logs.
- **FR-20** Dockerfile and `compose.yaml`; the container runs as a non-root user.

## 7. Non-functional requirements

- **NFR-1** Python 3.12, scikit-learn, pandas, numpy, FastAPI, pydantic, uvicorn, matplotlib, pytest. Pin everything in `pyproject.toml` with a lock file.
- **NFR-2** `/score` p95 latency under 50 ms for a single transaction on a laptop, measured with a small load test (for example 1,000 requests) and reported.
- **NFR-3** The full train-and-evaluate run finishes in under 15 minutes on a 16 GB laptop.
- **NFR-4** Deterministic: the same seed and data give the same metrics.
- **NFR-5** `ruff`, `mypy` on `fraudlens/` and `service/`, and test coverage of at least 85% on `fraudlens/` and `service/`.
- **NFR-6** CI on every push runs lint, types and tests against a small synthetic fixture (the real dataset is never in CI). A `workflow_dispatch` job runs the full evaluation when you provide the data.

## 8. Architecture

```
fraudlens/
  data.py        load, validate, time split
  features.py    shared feature pipeline (train and serve)
  models.py      baseline, RF, gradient boosting, isolation forest
  train.py       CLI: train all models, write versioned artifacts
  evaluate.py    CLI: metrics with bootstrap CIs, curves, cost analysis, report
  explain.py     permutation importance, per-transaction contributions
  costs.py       cost model and threshold search
service/
  app.py         FastAPI: /score, /health, /model
  schemas.py     pydantic request and response models
  Dockerfile
tests/
  fixtures/      small synthetic dataset with the same schema
  test_data.py  test_features.py  test_leakage.py  test_costs.py
  test_models.py  test_service.py
reports/          generated metrics, curves and report.md
data/README.md    how to obtain the dataset, its license and SHA-256
config.yaml       seeds, split ratios, cost table, search budgets
docs/prd/FL-01-fraudlens.md
.github/workflows/ci.yml
```

## 9. Testing strategy

| Area | Tests |
|---|---|
| Data | schema validation rejects bad files; the time split keeps every test row later than every training row |
| Leakage | fitted scaler statistics match training data only; resampling never touches validation or test rows |
| Features | identical output from the training path and the serving path for the same input |
| Costs | the threshold search finds the known optimum on a constructed example; money-saved arithmetic is correct |
| Models | each model trains on the fixture and returns probabilities in [0, 1]; seeds make results repeatable |
| Service | valid request gives 200 and a decision; missing or wrong-typed fields give 422; `/model` reports the loaded version; logs contain no feature values |
| Load | a scripted 1,000-request run records p95 latency (reported, not asserted in CI) |

## 10. Milestones

| # | Milestone | Done when |
|---|---|---|
| M1 | Data and split | FR-1–FR-3; leakage and split tests green |
| M2 | Baseline and honest metrics | FR-4, EV-1–EV-4; baseline PR-AUC with CIs in `metrics.csv` |
| M3 | Model comparison | FR-5–FR-9, EV-5; all models compared on the same splits |
| M4 | Cost-based threshold | EV-7, FR-11; cost curve and money saved reported |
| M5 | Explainability | FR-13–FR-14 |
| M6 | Service | FR-15–FR-20, NFR-2; Docker image runs; service tests green |
| M7 | Write-up | README with generated results, limitations and how to run |

## 11. README outline

1. What FraudLens does, in one paragraph, and why accuracy is the wrong metric here.
2. The dataset, its license, and its limits (PCA features, two days, 2013, Europe).
3. **How it is evaluated:** the time split, the metrics, the confidence intervals and the cost model.
4. **Results:** the generated table, the PR curves, the cost curve. State what is and is not significantly different.
5. **Explainability:** top features, and what PCA features can and cannot tell you.
6. **The API:** example request and response, the latency measurement, and how to run it with Docker.
7. **Limitations and next steps:** concept drift, label delay in real fraud systems, and what a production version would add (monitoring, retraining, human review loops).

## 12. Risks

| Risk | Mitigation |
|---|---|
| Overstating results | Use only test-set numbers with CIs; state the dataset's limits; never say "production-ready" |
| Leakage | Pipelines, time split and dedicated leakage tests (EV-6) |
| Dataset access or license | Manual download; license recorded; data never committed |
| Slow training on a 16 GB laptop | Histogram gradient boosting; capped search budgets; subsample for exploration only |
| SMOTE inflating validation scores | Resampling inside training folds only (EV-5) |

## 13. Definition of done

- [ ] CI green on the synthetic fixture: lint, types, tests and coverage
- [ ] `reports/metrics.csv`, the curves and `report.md` generated from one command and committed
- [ ] The chosen model and threshold are justified by validation results and reported once on the test set
- [ ] The Docker image serves `/score` with validation, a measured p95 latency and per-transaction explanations
- [ ] The README states the limitations honestly

## 14. Resume bullet templates

Use only after M4/M6, with the bracketed values taken from `reports/metrics.csv` and the load test.

- Built a fraud-detection pipeline on 284,807 card transactions (0.17% fraud) with a time-based split and leakage tests, reaching PR-AUC [X] (95% CI [a–b]) on a held-out final time slice versus [Y] for a logistic-regression baseline
- Chose the alert threshold from an explicit cost model, cutting expected fraud cost by [Z]% against flagging nothing, and served the model through a FastAPI scoring API with input validation and per-transaction explanations at [N] ms p95
