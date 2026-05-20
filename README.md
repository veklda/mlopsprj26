# Category Classification

**Note: Set project and API key in category_classification_fti.toml** (`project`, `api_key`)

1. **Docker** — quickest, works identically on Windows, macOS, and Linux.
2. **Manual setup** — Use this if you prefer a local Python environment or if the Docker route is not available.

**Note**: Data set size has been set to 15'000 rows in the TOML file to speed up the
process. Locally tested with 240'000 rows.

---

## Option A · Docker (Recommended)

The supplied `Dockerfile` packages the notebooks, shared utilities, and all
dependencies into a single image with a ready-to-run Jupyter server.

### Build the image

```bash
docker build -t category-classification-jupyter .
```

### Run the container

**Interactive (foreground)**
```bash
docker run -p 8888:8888 category-classification-jupyter
```

**Detached (background) with a named volume for data persistence**
```bash
# Linux / macOS
docker run -d -p 8888:8888 -v $(pwd)/data:/app/data --name cc-jupyter category-classification-jupyter

# Windows (PowerShell)
docker run -d -p 8888:8888 -v ${PWD}/data:/app/data --name cc-jupyter category-classification-jupyter

# Windows (cmd)
docker run -d -p 8888:8888 -v %cd%/data:/app/data --name cc-jupyter category-classification-jupyter
```

### Access Jupyter

Once the container is running, open your browser and navigate to:

**http://localhost:8888**

No token or password is required (the image is configured for local development).

---

## Option B · Manual Setup

### 1 · System Prerequisites

Install **Python 3.11** and **uv** for your platform before proceeding.

**Linux (Debian / Ubuntu)**
```bash
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev libgomp1
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env   # add uv to PATH (or restart shell)
```

**macOS**
```bash
brew install python@3.11 uv libomp
```

**Windows**
```
winget install Python.Python.3.11 astral-sh.uv
```
> If `import lightgbm` raises a DLL error on Windows, install the
> [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe).

---

### 2 · Create the Virtual Environment

Run from the `notebooks/` directory:

```bash
uv venv .venv --python 3.11 --seed
```

This creates a `.venv/` directory containing an isolated Python 3.11 interpreter.

---

### 3 · Activate the Virtual Environment

**Linux / macOS**
```bash
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (cmd)**
```cmd
.venv\Scripts\activate.bat
```

Your shell prompt will show `(.venv)` when the environment is active.

---

### 4 · Install Dependencies

```bash
uv pip install -r requirements.txt
```

All versions are pinned in `requirements.txt`. The install typically takes 2–5 minutes
on first run (sentence-transformers downloads model weights separately at runtime).

---

### 5 · Hopsworks Credentials

The pipeline reads credentials from environment variables first, then falls back to the
values in `category_classification_fti.toml`. Set environment variables to avoid
storing secrets in the TOML:

**Linux / macOS**
```bash
export HOPSWORKS_HOST="eu-west.cloud.hopsworks.ai"
export HOPSWORKS_PROJECT="<your-project-name>"
export HOPSWORKS_API_KEY="<your-api-key>"
```

**Windows (PowerShell)**
```powershell
$env:HOPSWORKS_HOST    = "eu-west.cloud.hopsworks.ai"
$env:HOPSWORKS_PROJECT = "<your-project-name>"
$env:HOPSWORKS_API_KEY = "<your-api-key>"
```

API keys can be generated in the Hopsworks UI under
**Account → API Keys**.

---

### 6 · Configuration

All paths, hyper-parameters, and service endpoints are centralised in
`category_classification_fti.toml`. The defaults work out of the box; the only values
that typically need changing are the three `[hopsworks]` credentials above (or override
via env vars as shown in step 5).

---

### 7 · Register the Kernel with Jupyter

If you launch Jupyter from outside the venv, register the kernel so the notebooks can
select it:

```bash
uv pip install ipykernel
python -m ipykernel install --user --name=category-classification --display-name "Category Classification (.venv)"
```

Then start Jupyter:

```bash
jupyter notebook
```

Select **Category Classification (.venv)** as the kernel in each notebook.

---

## FTI Pipeline

The workflow is split into three decoupled stages.  
Configuration is centralised in `category_classification_fti.toml`; shared
helper code lives in `category_classification.py`.

### Feature Extraction (`category_classification_f.ipynb`)

- **Data acquisition** – Downloads daily Parquet dumps from the EC DSA
  Transparency Database (CloudFront) or loads existing local chunks.
- **Stratified sampling** – Draws a configurable number of rows (default
  15 000, tested up to 240 000) while preserving category distribution.
- **Cleaning & parsing** – Handles missing values, stringifies lists, and
  extracts primary items / counts from multi-value columns (`content_type`,
  `territorial_scope`, `decision_visibility`).
- **Feature engineering** – Produces **28 structured features**:
  - *11 categorical* (`source_type`, `automated_detection`, `platform_name`,
    …) – low-cardinality columns are ordinal-encoded, high-cardinality
    columns are target-encoded.
  - *17 numerical* (`territorial_scope_count`, date parts, decision-action
    flags, and 3-hour rolling platform aggregates).
- **Text extraction** – Pulls the three free-text columns
  (`incompatible_content_ground`, `incompatible_content_explanation`,
  `decision_facts`) that feed TF-IDF in later tiers.
- **Chronological split** – Splits 80/20 by `application_date` (older rows
  train, newer rows test) to avoid future leakage.
- **Feature Store upload** – Writes structured and text columns into
  separate Hopsworks Feature Groups and creates a Feature View that joins
  them on `row_id`.

### Training (`category_classification_t.ipynb`)

- **Feature retrieval** – Reads the engineered feature matrix from the
  Hopsworks Feature View.
- **Preprocessing** – Fits a `ColumnTransformer` on a subsample
  (`preprocessor_fit_sample = 1 000`) applying ordinal / target encoding
  and `StandardScaler`.
- **TF-IDF vectorisation (Tier 2/3)** – Transforms the three text columns
  into sparse vectors (5 000 + 3 000 + 2 000 dimensions) with unigrams,
  bigrams, and sublinear TF.
- **Baseline models** – Trains a dummy (most-frequent) and a balanced
  `LogisticRegression` (saga solver) for reference.
- **LightGBM search** – Explores 50 random hyper-parameter candidates via
  `HalvingRandomSearchCV` (factor 3, 5 stratified folds) over tree count,
  depth, learning rate, regularisation, etc.
- **Architectures compared** – Evaluates flat vs. two-stage classifiers:
  - *Flat* – single multiclass LightGBM.
  - *Two-stage* – Stage-1 binary (OTHER vs. NOT-OTHER) and Stage-2
    multiclass on the non-OTHER subset.
- **Model selection** – Ranks pipelines by **macro F1**; the best Tier-2
  Two-Stage LightGBM (macro F1 ≈ 0.869) is promoted.
- **Registration** – Serialises the winning artifact, preprocessors, label
  maps, and `category_metadata.json` to the Hopsworks Model Registry.

### Inference (`category_classification_i.ipynb`)

- **Artifact loading** – Retrieves the registered model, preprocessors,
  and metadata from Hopsworks (or local `category_best_model.pkl`).
- **Local feature engineering** – Re-applies the identical transformation
  logic via `CategoryPredictor` on raw incoming rows.
- **Rolling-window warm start** – Optionally seeds 3-hour platform
  aggregates with `history_df` to prevent cold-start degradation.
- **Batch scoring** – Outputs predicted category, confidence, and
  per-class probabilities to `category_predictions.csv`.
- **Live stream** – Pulls the latest Parquet chunks, scores rows in real
  time (~3 rows/sec), and renders a dashboard tracking macro F1,
  precision, recall, and aggregate TP/FP/FN.

---

## Source Data

The pipeline consumes daily dumps from the **European Commission DSA Transparency Database**, hosted on a CloudFront distribution. The raw data is published as chunked Parquet files (~50 k rows per chunk) and contains statement-of-reason records submitted by online platforms.

Key raw fields used by the pipeline:

| Raw field | Meaning |
|-----------|---------|
| `category` | Target label — the violation category assigned by the platform |
| `platform_name` | Platform that issued the statement |
| `content_type` | List of modalities (TEXT, IMAGE, VIDEO, AUDIO, etc.) |
| `territorial_scope` | List of country codes where the decision applies |
| `decision_visibility` | Visibility restriction applied |
| `decision_monetary` | Monetary penalty (if any) |
| `decision_provision` | Service provision restriction (if any) |
| `decision_account` | Account action (if any) |
| `application_date` | When the statement was submitted |
| `content_date` | When the flagged content was created |
| `created_at` | Precise timestamp (used for rolling window features) |
| `automated_detection` | Whether detection was automated (YES/NO) |
| `automated_decision` | Whether the final decision was automated (YES/NO) |
| `source_type` | Who filed the statement (e.g. Trusted Flagger, user) |
| `account_type` | Type of account that posted the content |
| `content_language` | ISO language code of the content |
| `incompatible_content_illegal` | Flag alleging illegality |
| `decision_ground` | Statutory basis for the decision |
| `incompatible_content_ground` | Primary policy text (free text) |
| `incompatible_content_explanation` | Free-text rationale |
| `decision_facts` | Detailed decision facts (long prose, often sparse) |

By default the pipeline stratified-samples **15 000 rows** from a few days of dumps to keep runtime short. It has been locally tested with **240 000 rows**.

The train/test split is **chronological** (80 % older rows for training, 20 % newer rows for testing) to avoid leaking future information into the training set.

---

## Structured Features

The Feature Pipeline engineers **28 structured features** from the raw Parquet columns. These are the columns stored in the Hopsworks structured Feature Group and consumed by Tier-1 models. Tier-2 models additionally concatenate TF-IDF vectors derived from the three free-text columns.

### Categorical features (11)

| Feature | Description |
|---------|-------------|
| `source_type` | Who filed the statement (e.g. Trusted Flagger, user) |
| `automated_detection` | Boolean flag: was detection automated? |
| `automated_decision` | Boolean flag: was the final decision automated? |
| `platform_name` | Platform identifier (high cardinality) |
| `content_language` | ISO language code (high cardinality) |
| `account_type` | Type of account that posted the content (high cardinality) |
| `incompatible_content_illegal` | Flag: does the statement allege illegality? |
| `content_type_primary` | Primary content modality (first item of the raw list) |
| `territorial_scope_primary` | Primary country code (first item of the raw list) |
| `decision_ground` | Statutory basis for the decision |
| `decision_visibility_primary` | Primary visibility restriction applied (first item of list) |

**Encoding strategy**
- Low-cardinality columns (`source_type`, `automated_detection`, `automated_decision`, `incompatible_content_illegal`, `content_type_primary`, `decision_ground`, `decision_visibility_primary`) are ordinal-encoded.
- High-cardinality columns (`platform_name`, `content_language`, `account_type`, `territorial_scope_primary`) are target-encoded.

### Numerical features (17)

| Feature | Description |
|---------|-------------|
| `territorial_scope_count` | Number of countries in the territorial_scope list |
| `app_year` | Year of `application_date` |
| `app_month` | Month of `application_date` |
| `content_year` | Year of `content_date` |
| `decision_facts_len` | Character length of the `decision_facts` text |
| `content_application_lag_days` | Days between content creation and application |
| `has_monetary_decision` | Binary: fine or monetary penalty imposed |
| `has_provision_decision` | Binary: service provision restricted |
| `has_account_decision` | Binary: account suspended or terminated |
| `total_decision_actions` | Count of distinct decision actions taken |
| `is_multi_action_decision` | Flag: more than one action applied |
| `content_type_count` | Number of content types listed |
| `visibility_decision_count` | Number of visibility restrictions listed |
| `platform_volume_3h` | Statements by the same platform in the preceding 3 hours |
| `platform_auto_fraction_3h` | Fraction of automated decisions in the preceding 3 hours |
| `platform_content_type_div_3h` | Number of distinct content types in the preceding 3 hours |
| `platform_automation_rate` | Historical automation fraction per platform (computed from training rows only) |

All numerical columns are passed through `StandardScaler` inside the preprocessing pipeline.

---

## Textual Features (TF-IDF)

Tier-2 and Tier-3 pipelines augment the 28 structured features with sparse TF-IDF vectors derived from three free-text columns in the raw DSA data. Each text column gets its own `TfidfVectorizer` configured independently.

| Text column | Raw field | Description | Dimensions | Key params |
|-------------|-----------|-------------|------------|------------|
| **Primary policy text** | `incompatible_content_ground` | Statutory ground cited for the takedown (e.g. "hate speech", "child safety") | 5 000 | n-grams 1–2, sublinear TF |
| **Free-text rationale** | `incompatible_content_explanation` | Human-readable explanation written by the reporting entity | 3 000 | n-grams 1–2, sublinear TF |
| **Decision facts** | `decision_facts` | Detailed decision facts (longer prose, often empty) | 2 000 | n-grams 1–2, sublinear TF |

**Common TF-IDF settings**
- `ngram_range = (1, 2)` — unigrams and bigrams capture phrases like "hate speech" or "intellectual property".
- `sublinear_tf = true` — applies `1 + log(tf)` to dampen the influence of very frequent terms.
- `min_df = 2` — ignores terms that appear in fewer than 2 rows, reducing noise from typos and rare tokens.

**Fill rates** (on the default sample)
- `incompatible_content_ground`: ~99.8 %
- `incompatible_content_explanation`: ~99.8 %
- `decision_facts`: ~100.0 %

The pipeline raises an error if any text column falls below a **5 % fill rate**, because a near-empty TF-IDF matrix would produce almost no signal.

The resulting sparse feature matrix is concatenated with the structured features, giving Tier-2 models approximately **10 000 sparse text dimensions** on top of the 28 dense structured features.

---

## Model — LightGBM

The primary classifier is **LightGBM** (`LGBMClassifier`). It was chosen because it natively handles the mixed categorical/numerical feature matrix after preprocessing, trains quickly on sampled data, and scales to millions of rows.

**Fixed settings**
- `objective = multiclass` — softmax loss for multi-class classification
- `class_weight = balanced` — inverse-frequency weighting per class to counter the severe imbalance
- `random_state = 42` — reproducible row and feature sub-sampling
- `n_jobs = 1` during search (HalvingRandomSearchCV parallelises across folds)
- `device = cpu` by default (override via TOML to `gpu` or `cuda` if available)

**Hyper-parameter search space** (explored by `HalvingRandomSearchCV`)

| Parameter | Distribution | Range |
|-----------|--------------|-------|
| `n_estimators` | uniform int | 50 – 300 |
| `max_depth` | uniform int | 3 – 9 |
| `learning_rate` | log-uniform | 1e-3 – 0.4 |
| `num_leaves` | uniform int | 15 – 126 |
| `subsample` | uniform float | 0.6 – 1.0 |
| `colsample_bytree` | uniform float | 0.6 – 1.0 |
| `reg_alpha` | log-uniform | 1e-2 – 5.0 |
| `reg_lambda` | log-uniform | 0.1 – 5.0 |
| `min_child_samples` | uniform int | 5 – 50 |

Search uses **50 random candidates** with a reduction **factor of 3** and **5 stratified folds**, so only the best one-third of candidates survive each successive halving round. This is typically 3–5× faster than exhaustive grid search with comparable final performance.

---

## Pipelines using LightGBM

Four variants are trained and compared. All use the same LightGBM estimator family but differ in whether text features are included and whether a two-stage hierarchy is applied.

| Pipeline | Features | Architecture | Test macro F1 |
|----------|----------|--------------|---------------|
| **LightGBM (T1, flat)** | Structured only (28 features) | Single multiclass classifier | 0.505 |
| **LightGBM (T1, two-stage)** | Structured only (28 features) | Stage 1: binary OTHER vs NOT-OTHER; Stage 2: multiclass on non-OTHER | 0.526 |
| **LightGBM (T2, flat)** | Structured + TF-IDF on 3 text columns (~10 000 sparse dims) | Single multiclass classifier | 0.830 |
| **LightGBM (T2, two-stage)** | Structured + TF-IDF on 3 text columns (~10 000 sparse dims) | Stage 1: binary OTHER vs NOT-OTHER; Stage 2: multiclass on non-OTHER | **0.869** |

**TF-IDF configuration for Tier-2 pipelines**
- `incompatible_content_ground` — 5 000 dimensions, unigrams + bigrams
- `incompatible_content_explanation` — 3 000 dimensions, unigrams + bigrams
- `decision_facts` — 2 000 dimensions, unigrams + bigrams

The **Tier-2 Two-Stage LightGBM** is the best-performing model and is the one registered in the Hopsworks Model Registry for inference.

### Two-stage architecture

The two-stage approach splits the problem into:
1. **Stage 1** — a binary classifier that decides whether a row belongs to the catch-all `OTHER_VIOLATION_TC` class or not.
2. **Stage 2** — a multiclass classifier trained only on rows that are **not** OTHER, freeing it from competing against the dominant majority class for gradient weight.

Probabilities from both stages are stitched so the final output preserves the full softmax probability distribution across all 13 classes.

---

## Limitations

1. **Class imbalance** — `OTHER_VIOLATION_TC` represents ~80 % of the default sample. Rare categories (e.g. `PROTECTION_OF_MINORS`, `ANIMAL_WELFARE`) have very few examples, making them hard to learn reliably.

2. **Sample size** — The default 15 000-row sample is tiny compared with the full multi-billion-row DSA dataset. Metrics on the sample may not generalise to the full corpus.

3. **Temporal drift** — The chronological train/test split assumes that past distributions predict future ones. Real-world category prevalence and platform behaviour shift over time, so performance may degrade on future dumps without retraining.

4. **Text sparsity** — The `decision_facts` column is often sparse or empty, and its signal is correspondingly weak.

5. **Two-stage cascade risk** — If Stage 1 misclassifies an OTHER row as NOT-OTHER, Stage 2 is forced to assign a fine-grained label that does not exist. Conversely, a rare fine-grained category misclassified as OTHER is never recovered.

6. **Fixed TF-IDF vocabulary** — The vectoriser vocabulary is frozen at training time. New legal phrases or policy language that appear after training are treated as out-of-vocabulary.

7. **No online learning** — The model is static between full retraining runs. It does not adapt incrementally as new labelled data arrives.
