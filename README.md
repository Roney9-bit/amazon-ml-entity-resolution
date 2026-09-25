# Amazon ML Challenge — Business Entity Resolution

Team repository for the business entity resolution challenge.

## Fixed baseline approach

We will keep one consistent baseline while the team gets the pipeline running:

```text
Raw TSV
  -> conservative normalization
  -> multi-rule candidate generation / blocking
  -> pairwise similarity features
  -> Logistic Regression pair classifier
  -> validation threshold selection using macro F0.5
  -> test prediction
  -> candidate_pairs.tsv + matching_results.tsv
```

Do not change the architecture casually between steps. Improve one component at a time and record the measured effect in `docs/experiments.md`.

## Repository layout

```text
src/business_entity_resolution/
  preprocessing/   # name/address normalization
  blocking/        # exact, rare-token, and TF-IDF candidate generation
  features/        # RapidFuzz, Jaccard, TF-IDF, digit, country features
  models/          # pairwise Logistic Regression
  evaluation/      # ground truth, F0.5, threshold tuning
  pipeline/        # train and predict entry points
  utils/           # TSV I/O
scripts/            # EDA, training, prediction, validation, packaging
notebooks/          # Colab notebook
output/             # generated submission files (ignored by Git)
data/              # challenge files locally / in Drive (ignored by Git)
artifacts/          # trained model + metadata (ignored by Git)
```

## Colab + Google Drive

The recommended workflow is:

```text
GitHub = code / notebooks / tests / documentation
Google Drive = dataset / trained model / generated outputs
Colab = execution environment
```

Open `notebooks/amazon_entity_resolution_colab.ipynb` in Colab. The notebook:

1. mounts Google Drive;
2. clones the GitHub repository into `/content`;
3. installs the project from the cloned repository;
4. points the pipeline at the dataset in Drive;
5. runs EDA, training, prediction, and validation;
6. saves model artifacts and outputs back to Drive.

Expected Drive layout:

```text
MyDrive/amazon-ml-entity-resolution/
  data/
    train/
      train_source1.tsv
      train_source2.tsv
      train_source3.tsv
      train_ground_truth.tsv
    test/
      test_source1.tsv
      test_source2.tsv
      test_source3.tsv
  artifacts/
  output/
```

The actual TSV files should not be pushed to GitHub.

## Local commands

### EDA

```bash
python scripts/eda.py --data-dir data/train
```

### Train

```bash
python scripts/train.py \
  --data-dir data/train \
  --artifacts-dir artifacts \
  --validation-size 0.2 \
  --negatives-per-positive 5 \
  --random-state 42
```

### Predict

```bash
python scripts/predict.py \
  --data-dir data/test \
  --artifacts-dir artifacts \
  --output-dir output
```

### Validate

```bash
python scripts/validate.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir data/test
```

Expected result:

```text
PASS
```

### Package final submission

```bash
python scripts/package_submission.py --team-name YOUR_TEAM_NAME
```

## Important challenge constraints

Use only the supplied challenge data. Do not use external entity databases, business lookup APIs, geocoding APIs, or internet-based data augmentation.

The final submission requires `output/matching_results.tsv`, `output/candidate_pairs.tsv`, runnable code under `code/business_entity_resolution/src/`, a pinned requirements file, and the completed methodology document.

## Noisy names + addresses: blocking rule

Candidate generation uses the **UNION** of independent name and address channels; it does not require both fields to match.

Name channels include exact normalized name, rare name tokens, and character TF-IDF nearest neighbours.
Address channels include exact normalized address, address tokens, first numeric address token, and character TF-IDF nearest neighbours.
A combined character TF-IDF channel over both fields is also included.

This means a pair such as `Mirasol` vs `Team Air Pvt Ltd` with the same address can still become a candidate through the address branch.
Conversely, a missing address does not discard a good name candidate, and a missing name does not discard a good address candidate.

The feature layer is also missing-aware: empty-vs-empty fields are not treated as perfect similarity. Missingness is represented explicitly.
