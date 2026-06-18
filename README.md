# Data Mining (CS405/CS505) — Assignment 2: Evaluation and Selection of Decision Models

Coursework for **Data Mining (CS405/CS505)**, Bishop's University.

Cross-validation for estimating generalisation error and for selecting hyperparameters, using a
multi-layer perceptron (MLP) on the Iris dataset. The full solution, answering all eleven
questions, is in [`Assignment2_Model_Selection_CV.ipynb`](Assignment2_Model_Selection_CV.ipynb).

## Topics covered

- Pairwise feature scatter plots of train/test points.
- K-fold cross-validation as a function of `k` (and its growing variance for large `k`).
- Overlaying the test-set performance on the CV estimate.
- Leave-one-out cross-validation and how it compares to k-fold.
- `GridSearchCV` over `hidden_layer_sizes` and `alpha`, with a 3-D score surface.
- The number of models trained, the `refit` parameter, and `cv_results_`.
- Refining the grid, `RandomizedSearchCV` (log-uniform for `alpha`), and `HalvingGridSearchCV`.

## Running it

```bash
pip install numpy scipy matplotlib scikit-learn pandas
jupyter notebook Assignment2_Model_Selection_CV.ipynb
```

## Files

| File | Description |
|------|-------------|
| `Assignment2_Model_Selection_CV.ipynb` | Full solution with answers to all 11 questions |
| `Assignment 2.pdf` | Assignment description |
