# Decision Model Selection and Cross-Validation

Exploring cross-validation strategies and hyperparameter search for model selection, using an MLP classifier on the Iris dataset.

Cross-validation for estimating generalisation error and for selecting hyperparameters, using a
multi-layer perceptron (MLP) on the Iris dataset. The full implementation and analysis, covering all eleven
questions, is in [`model_selection_cross_validation.ipynb`](model_selection_cross_validation.ipynb).

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
jupyter notebook model_selection_cross_validation.ipynb
```

## Files

| File | Description |
|------|-------------|
| `model_selection_cross_validation.ipynb` | Full implementation and analysis covering all 11 questions |
| `PROJECT_BRIEF.pdf` | Project brief (goals, objectives, outcomes) |
