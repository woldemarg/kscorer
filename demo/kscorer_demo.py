import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from kscorer.kscorer import KScorer

# %%

ks = KScorer()

# %%

X, y = datasets.load_digits(return_X_y=True)
X.shape

# %%

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234)

# %%

labels, centroids, _ = ks.fit_predict(X_train, retall=True)

# %%

ks.show()

# %%

ks.optimal_

# %%

labels_mtx = (pd.Series(y_train)
              .groupby([labels, y_train])
              .count()
              .unstack()
              .fillna(0))

# match arbitrary labely to ground-truth labels
order = []

for i, r in labels_mtx.iterrows():
    left = [x for x in np.unique(y_train) if x not in order]
    order.append(r.iloc[left].idxmax())

confusion_mtx = labels_mtx[order]
confusion_mtx

# %%

labels_unseen = ks.predict(X_test, init=centroids)

# %%

y_clustd = pd.Series(labels).replace(dict(enumerate(order)))
y_unseen = pd.Series(labels_unseen).replace(dict(enumerate(order)))

# %%

balanced_accuracy_score(y_train, y_clustd)

# %%

balanced_accuracy_score(y_test, y_unseen)
