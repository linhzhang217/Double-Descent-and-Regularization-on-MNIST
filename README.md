# Double Descent and Regularization on MNIST

An empirical study of model capacity, regularization, and generalization using MNIST. This project combines neural-network experiments in PyTorch with Monte Carlo analysis of matrix conditions associated with ridge regression.

The notebooks explore two questions:

- How do training and test error change as neural-network width increases, with and without regularization?
- How do finite-sample estimates of ridge-regression matrix conditions behave when the sample size increases from 200 to 201?

## Repository Contents

| Notebook | Focus |
| --- | --- |
| [Double_Descent_Code.ipynb](Double_Descent_Code.ipynb) | Neural-network width sweeps comparing an unregularized baseline, L2 regularization, and dropout. |
| [math 156 L2 reg empirical proofs code.ipynb](math%20156%20L2%20reg%20empirical%20proofs%20code.ipynb) | Monte Carlo estimates and eigenvalue checks for ridge-regression matrix conditions. |

## 1. Neural-Network Experiments

### Experimental Setup

| Component | Configuration |
| --- | --- |
| Dataset | MNIST; 4,000 randomly selected training images and the standard test split |
| Label noise | Each selected training label is independently replaced with an incorrect label with probability 10% |
| Architecture | Flatten → Linear(784, width) → ReLU → Linear(width, 10) |
| Hidden widths | 2, 5, 10, 20, 30, 40, 50, 100, 150, 250, 300 |
| Objective | Mean squared error between model outputs and one-hot labels |
| Optimizer | SGD, learning rate 0.1, momentum 0.95 |
| Training budget | Up to 6,000 epochs per model |
| Metrics | Training and test MSE; classification error is also computed |
| Device | CUDA when available; otherwise CPU |

The notebook compares five configurations:

1. **Baseline:** no weight decay or dropout; pixel values scaled to [0, 1].
2. **L2 regularization:** SGD weight decay of 0.01 with input normalization.
3. **Dropout:** probabilities of 0.2, 0.5, and 0.8, applied after ReLU, with normalized inputs and no weight decay.

Normalization uses the sampled training images' global mean and standard deviation. For widths at most 5, the code also applies a learning-rate scheduler and allows early stopping at zero training classification error. The width-5 model reuses overlapping weights from the width-2 model.

### Selected Saved Results

At hidden width 300, the notebook records:

| Configuration | Training MSE | Test MSE |
| --- | ---: | ---: |
| Baseline | 0.0000* | 0.0433 |
| L2 + normalization | 0.0338 | 0.0242 |
| Dropout 0.2 + normalization | 0.0020 | 0.0274 |
| Dropout 0.5 + normalization | 0.0058 | 0.0293 |
| Dropout 0.8 + normalization | 0.0704 | 0.0733 |

*Rounded to four decimal places; this does not establish exactly zero loss.*

The saved baseline curve shows nonmonotonic test error followed by improvement at larger widths. The normalized L2 configuration has lower test MSE than the baseline at width 300, while dropout of 0.8 retains substantially higher training and test error.

These are exploratory observations from the saved run. Because input normalization changes between the baseline and regularized configurations, the comparison does not isolate the effect of regularization alone. The hard-coded `interp_thresh = 5` controls training behavior; it is not an empirically established interpolation threshold.

## 2. Ridge-Regression Monte Carlo Analysis

The second notebook uses flattened MNIST images with 784 features, independently sampled datasets of sizes 200 and 201, and a regularization parameter of 0.01. Features are standardized separately within each sampled dataset.

For a design matrix X, it forms:

```text
C = XᵀX + λI
Gₙ = λ² E[C⁻²]
Hₙ = E[‖C⁻¹Xᵀ‖²_F]
```

Expectations are estimated by Monte Carlo sampling. The notebook computes derivatives with respect to λ and examines the smallest eigenvalues of:

```text
Condition (24): Gₙ − Gₙ₊₁
Condition (25): (Gₙ − Gₙ₊₁) − [(Hₙ − Hₙ₊₁) / (dHₙ/dλ)] · (dGₙ/dλ)
```

The condition numbers follow the notebook's labels; the source theorem or derivation is not included in the supplied files.

The saved 100-trial run reports minimum eigenvalues of approximately **−0.1265** for both matrices. Thus, these finite-sample estimates do not pass the notebook's positive-semidefiniteness check. This is a numerical diagnostic, not a mathematical proof or disproof of the underlying conditions.

A separate trial-count sweep from 1,000 to 10,000 is implemented, but its saved execution was interrupted at the first setting. The supplied outputs therefore do not establish convergence as the trial count grows.

## Setup

Use a Python environment with NumPy, Matplotlib, PyTorch, torchvision, and JupyterLab. The notebook metadata records Python 3.12.7 and 3.11.5; exact package versions are not pinned.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy matplotlib torch torchvision jupyterlab
jupyter lab
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

Both notebooks download MNIST automatically on first use, requiring an internet connection. Data is stored in `./data` and `./mnist_data`, respectively.

## Running the Notebooks

### Neural-Network Notebook

Open `Double_Descent_Code.ipynb` and run the cells in order from a fresh kernel. The dropout section depends on the normalized data loaders created in the L2 section.

For a quick smoke run, reduce the width list and replace each `range(1, 6001)` with a smaller epoch range. The full notebook trains 55 models and can take substantial time, particularly on CPU. Reduced training budgets will not reproduce the saved results.

### Ridge-Regression Notebook

Open `math 156 L2 reg empirical proofs code.ipynb`. For an initial run, reduce `M_values` in the first code cell, for example to `np.array([10])`, before running it. Each trial involves dense 784 × 784 matrix operations.

The later 100-trial cell depends on `all_data` initialized in the first code cell. To run it independently in a fresh kernel, insert the following immediately after loading `mnist_train`:

```python
all_data = mnist_train.data.numpy().astype(np.float64)
```

Then run the remaining code in that cell. Lowering `M` is useful for checking execution but increases Monte Carlo uncertainty.

## Reproducibility and Scope

- Random seeds are not fixed in the supplied notebooks; sampled images, corrupted labels, initial weights, and outputs may vary between runs.
- Restart the kernel before rerunning the neural-network experiments: normalization and label-noise injection modify shared dataset objects.
- The saved results are not averaged over repeated seeds and do not include confidence intervals.
- The ridge notebook studies feature-matrix quantities; it does not train or evaluate a supervised ridge predictor using MNIST labels.
- The notebooks are exploratory coursework, not formal proofs or a controlled benchmark establishing double descent.

Potential extensions include matching preprocessing across all neural-network configurations, measuring the interpolation threshold from training results, repeating experiments across seeds, and quantifying Monte Carlo uncertainty.
