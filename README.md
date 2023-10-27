KScorer: Auto-select optimal K-means clusters with advanced scoring

### Basic Usage
#### Load Modules
```python
In [1]: import numpy as np
   ...: import pandas as pd
   ...: from sklearn import datasets
   ...: from sklearn.metrics import balanced_accuracy_score
   ...: from sklearn.model_selection import train_test_split
   ...: from kscorer.kscorer import KScorer
```
#### Init KScorer
```python
In [2]: ks = KScorer()
```
#### Get Data
```python
In [3]: X, y = datasets.load_digits(return_X_y=True)
   ...: X.shape
Out[3]: (1797, 64)
```
#### Train/Test Split
```python
In [4]: X_train, X_test, y_train, y_test = train_test_split(
   ...:     X, y, test_size=0.2, random_state=1234)
```
#### Fit KScorer (i.e. Perform Unsupervised Clustering)
```python
In [5]: labels, centroids, _ = ks.fit_predict(X_train, retall=True)
100%|██████████| 13/13 [00:09<00:00,  1.39it/s]
```
#### Optimal Clusters
```python
In [6]: ks.show()
```
![image](https://github.com/woldemarg/kscorer/blob/main/demo/digits_demo.png?raw=true)

```python
In [7]: ks.optimal_
Out[7]: 10
```
#### Confusion Matrix
```python
In [8]: labels_mtx = (pd.Series(y_train)
   ...:               .groupby([labels, y_train])
   ...:               .count()
   ...:               .unstack()
   ...:               .fillna(0))
   ...: # match arbitrary labely to ground-truth labels
   ...: order = []
   ...: 
   ...: for i, r in labels_mtx.iterrows():
   ...:     left = [x for x in np.unique(y_train) if x not in order]
   ...:     order.append(r.iloc[left].idxmax())
   ...: 
   ...: confusion_mtx = labels_mtx[order]
   ...: confusion_mtx
```
|   	| 5     	| 9    	| 4     	| 2     	| 0     	| 6     	| 1    	| 7     	| 8    	| 3    	|
|---	|-------	|------	|-------	|-------	|-------	|-------	|------	|-------	|------	|------	|
| 0 	| 124.0 	| 5.0  	| 1.0   	| 0.0   	| 0.0   	| 0.0   	| 2.0  	| 7.0   	| 4.0  	| 2.0  	|
| 1 	| 12.0  	| 95.0 	| 0.0   	| 0.0   	| 0.0   	| 0.0   	| 0.0  	| 0.0   	| 9.0  	| 90.0 	|
| 2 	| 2.0   	| 0.0  	| 122.0 	| 0.0   	| 1.0   	| 2.0   	| 0.0  	| 1.0   	| 0.0  	| 0.0  	|
| 3 	| 0.0   	| 0.0  	| 0.0   	| 108.0 	| 0.0   	| 0.0   	| 22.0 	| 0.0   	| 1.0  	| 20.0 	|
| 4 	| 1.0   	| 2.0  	| 1.0   	| 0.0   	| 147.0 	| 0.0   	| 0.0  	| 0.0   	| 0.0  	| 0.0  	|
| 5 	| 2.0   	| 1.0  	| 0.0   	| 0.0   	| 2.0   	| 145.0 	| 3.0  	| 0.0   	| 4.0  	| 0.0  	|
| 6 	| 0.0   	| 1.0  	| 2.0   	| 22.0  	| 0.0   	| 0.0   	| 67.0 	| 7.0   	| 57.0 	| 6.0  	|
| 7 	| 0.0   	| 5.0  	| 8.0   	| 0.0   	| 0.0   	| 0.0   	| 0.0  	| 130.0 	| 4.0  	| 6.0  	|
| 8 	| 0.0   	| 15.0 	| 0.0   	| 9.0   	| 0.0   	| 0.0   	| 0.0  	| 0.0   	| 57.0 	| 21.0 	|
| 9 	| 0.0   	| 22.0 	| 3.0   	| 0.0   	| 0.0   	| 1.0   	| 48.0 	| 0.0   	| 6.0  	| 2.0  	|
#### Cluster Unseen Data (you would prefer to build classifier instead)
```python
In [9]: labels_unseen = ks.predict(X_test, init=centroids)
```
#### Evaluate Accuracy
```python
In [10]: y_clustd = pd.Series(labels).replace(dict(enumerate(order)))
    ...: y_unseen = pd.Series(labels_unseen).replace(dict(enumerate(order)))
```
```python
In [11]: balanced_accuracy_score(y_train, y_clustd)  # train data
Out[11]: 0.6940733254455871
```
```python
In [12]: balanced_accuracy_score(y_test, y_unseen)  # unseen data
Out[12]: 0.646615365026082
```
___

#### ToDo:
- apply power-transform before initial scaling-
- consider [pyckmeans](https://pypi.org/project/pyckmeans)
- consider [pyxmeans](https://github.com/mynameisfiber/pyxmeans)