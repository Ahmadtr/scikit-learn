"""
===========================================================
Nearest neighbors classification with the Hassanat distance
===========================================================

This example compares the accuracy of a k-nearest neighbors classifier using
the Euclidean, Manhattan, Canberra and Hassanat distances (see
:ref:`hassanat_distance`) on the wine dataset, in three situations:

- the raw features, which are measured on very different scales;
- standardized features;
- raw features where 5% of the entries are replaced by large outlier values.

The Hassanat distance adds up a bounded contribution for each feature, so a
feature with a large scale or a corrupted value cannot dominate the distance.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Data and outliers
# -----------------
# We load the wine dataset and build a corrupted copy of it, where 5% of the
# entries are replaced by values far above the range of their feature.
import numpy as np

from sklearn.datasets import load_wine

X, y = load_wine(return_X_y=True)

rng = np.random.RandomState(0)
X_outliers = X.copy()
is_outlier = rng.uniform(size=X.shape) < 0.05
feature_range = X.max(axis=0) - X.min(axis=0)
spikes = X.max(axis=0) + 10 * feature_range * rng.uniform(size=X.shape)
X_outliers[is_outlier] = spikes[is_outlier]

# %%
# Cross-validated accuracy
# ------------------------
# We evaluate a 5-nearest neighbors classifier with each distance, using
# repeated stratified cross-validation.
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

metrics = ["euclidean", "manhattan", "canberra", "hassanat"]
scenarios = {
    "raw features": (X, False),
    "standardized features": (X, True),
    "raw features with 5% outliers": (X_outliers, False),
}
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)

scores = {}
for scenario, (data, scale) in scenarios.items():
    for metric in metrics:
        model = KNeighborsClassifier(n_neighbors=5, metric=metric)
        if scale:
            model = make_pipeline(StandardScaler(), model)
        scores[scenario, metric] = cross_val_score(model, data, y, cv=cv)

# %%
# Results
# -------
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 4))
width = 0.2
for i, metric in enumerate(metrics):
    means = [scores[scenario, metric].mean() for scenario in scenarios]
    stds = [scores[scenario, metric].std() for scenario in scenarios]
    positions = np.arange(len(scenarios)) + (i - 1.5) * width
    ax.bar(positions, means, width, yerr=stds, label=metric)
ax.set_xticks(np.arange(len(scenarios)), list(scenarios))
ax.set_ylabel("Cross-validated accuracy")
ax.set_ylim(0.4, 1.0)
ax.legend(title="distance")
plt.tight_layout()
plt.show()

# %%
# On the raw features, the Euclidean and Manhattan distances are dominated by
# the features with the largest scales, while the Canberra and Hassanat
# distances are not. Once the features are standardized, the four distances
# perform similarly: the Hassanat distance is not expected to beat the
# Euclidean distance on clean, standardized data. With outliers, the Euclidean
# and Manhattan distances lose a lot of accuracy, while the Canberra and
# Hassanat distances, whose per-feature contributions are bounded, are much
# less affected.
