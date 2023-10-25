from sklearn import datasets
from sklearn.preprocessing import scale, normalize
from src.kscorer.kscorer import KScorer

# %%

ks = KScorer()

# %%

X, y = datasets.make_classification(
    n_samples=250000,
    n_features=10,
    n_informative=8,
    n_classes=25,
    n_clusters_per_class=1,
    class_sep=1.5,
    random_state=1234)

# %%

ks.fit(normalize(scale(X)), 10, 40)
ks.show()

# %%

X, y = datasets.make_blobs(
    n_samples=10000,
    n_features=10,
    centers=5,
    random_state=1234)

# %%

ks.fit(normalize(scale(X)))
ks.show()
