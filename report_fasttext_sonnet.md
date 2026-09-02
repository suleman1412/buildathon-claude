# Validation Report: Bag of Tricks for Efficient Text Classification

| Field | Value |
|---|---|
| Repo | https://github.com/facebookresearch/fastText |
| Dataset | YFCC100M |
| Claimed metric | precision@1 = 46.1% |
| Reproduced metric | 1.0% |
| **Result** | **FAIL** |

**Notes:** Reproduced value differs from claimed by 97.8% (tolerance: 5%).

_fastText with hidden size h=200 and bigram features, trained for 5 epochs on the YFCC100M tag prediction dataset (91,188,648 train examples), evaluated on the held-out test set of 543,424 examples using precision@1 for predicting tags from title and caption text._
