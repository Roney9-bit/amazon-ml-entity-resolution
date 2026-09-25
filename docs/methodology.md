# Methodology — Business Entity Resolution

## 1. Problem statement

We link each Source 1 business to zero, one, or multiple matching records in Source 2 and/or Source 3.

## 2. Data preparation

Describe missing-value handling and conservative normalization of business names and addresses.

## 3. Candidate generation / blocking

Describe exact-match blocks, token blocks, and TF-IDF nearest-neighbor blocks. Report candidate recall and reduction ratio.

## 4. Feature engineering

Describe name similarity, address similarity, token overlap, digit overlap, country agreement, and length-based features.

## 5. Matching model

Describe the pairwise classifier, class balancing, training data construction, and probability threshold.

## 6. Validation

Describe the entity-level validation split and macro F0.5 evaluation, including singleton handling.

## 7. Results

Add the measured validation F0.5, candidate recall, and any experiments comparing alternative approaches.

## 8. Reproducibility

Document the exact commands, Python version, dependency versions, and expected output files.

## 9. Constraints / fair play

The pipeline uses only the supplied challenge data and does not perform external entity lookup, geocoding, or data augmentation.
