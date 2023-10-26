import numpy as np
import pandas as pd
from sklearn import datasets
from kscorer.kscorer import KScorer

# %%

ks = KScorer()

# %%

X, y = datasets.make_classification(
    n_samples=25000,
    n_features=10,
    n_informative=8,
    n_classes=9,
    n_clusters_per_class=1,
    class_sep=1.5,
    random_state=1234)


# %%

ks.fit(X)
ks.show()

# %%

labels = ks.get_labels(X)

labels_mtx = pd.Series(y).groupby([labels, y]).count().unstack()

order = np.nanargmax(labels_mtx, axis=1)

labels_mtx[order]
