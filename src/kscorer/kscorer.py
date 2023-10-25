import warnings
import numpy as np
import pandas as pd
import scipy
from sklearn.cluster import KMeans
from sklearn.metrics import (
    pairwise_distances,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score)
from kneefinder import KneeFinder
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from pmixin import ParallelMixin


# %%

class KScorer(ParallelMixin):
    def __init__(self,
                 nsplits: int = 10,
                 frac: float = 0.15,
                 lmax: int = 5000,
                 lmin: int = 100,
                 random_state: int = 1234):

        self.nsplits = nsplits
        self.frac = frac
        self.lmax = lmax
        self.lmin = lmin
        self.random_state = random_state
        self.optimal_ = None
        self.peak_scores_ = None
        self.ranked_ = None
        self.scores_ = None

    @staticmethod
    def _find_knee(*args):

        # https://pypi.org/project/kneefinder/
        # https://github.com/arvkevi/kneed

        knee_x, _ = KneeFinder(*args).find_knee()

        return knee_x

    def _ranking_fun(self,
                     scores: dict) -> pd.Series:

        scored = (pd.DataFrame
                  .from_dict(
                      scores,
                      orient='index')
                  .sort_index())

        score_bic = scored.pop(scored.columns[-1])

        scored.iloc[:, -1] = scored.iloc[:, -1] * -1

        ranked = scored.rank(pct=True, axis=0)

        ranked.columns = [
            'silhouette score',
            'calinski-harabasz',
            'dunn index simpl',
            'davis-bouldin inv']

        knee_bic = self._find_knee(ranked.index, score_bic)

        ranked['BIC elbow'] = np.nan
        ranked.loc[knee_bic, 'BIC elbow'] = 1

        self.scores_ = scores
        self.ranked_ = ranked

        return ranked

    @staticmethod
    def _calculate_bic(num_samples: int,
                       num_clusters: int,
                       wss: float) -> float:

        return (num_samples * np.log(wss / num_samples) +
                np.log(num_samples) * num_clusters)

    @staticmethod
    def _calculate_dunn_index(data: np.ndarray,
                              labels: np.ndarray,
                              centroids: np.ndarray) -> float:

        # https://gist.github.com/douglasrizzo/cd7e792ff3a2dcaf27f6
        # https://github.com/jqmviegas/jqm_cvi/blob/master/jqmcvi/base.py
        # https://python.engineering/dunn-index-and-db-index-cluster-validity-indices-set/

        cluster_distances = []

        for cluster_label in np.unique(labels):

            cluster_points = data[labels == cluster_label]

            if len(cluster_points) > 1:
                intra_cluster_distances = pairwise_distances(
                    cluster_points, metric='euclidean', n_jobs=-1)

                cluster_distances.append(np.mean(intra_cluster_distances))

        inter_cluster_distances = pairwise_distances(
            centroids, metric='euclidean', n_jobs=-1)

        min_inter_cluster_distance = np.min(
            inter_cluster_distances[inter_cluster_distances > 0])

        max_intra_cluster_distance = np.max(cluster_distances)

        dunn_index = min_inter_cluster_distance / max_intra_cluster_distance

        return dunn_index

    def kmeans_clustering(self,
                          data: np.ndarray,
                          n_clusters: int,
                          retall: bool = True) -> tuple:

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init='auto')

        labels = kmeans.fit_predict(data)

        centroids = kmeans.cluster_centers_

        inertia = kmeans.inertia_

        if retall:
            return labels, centroids, inertia

        return labels

    def _get_scores(self,
                    data: np.array,
                    *args) -> tuple:

        n_clusters = args[0]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            labels, centroids, inertia = self.kmeans_clustering(data, n_clusters)

        score_silhouette = silhouette_score(data, labels)
        score_calinski = calinski_harabasz_score(data, labels)
        score_dunn = self._calculate_dunn_index(data, labels, centroids)
        score_davis = davies_bouldin_score(data, labels)
        score_bic = self._calculate_bic(
            num_samples=data.shape[0],
            num_clusters=len(np.unique(labels)),
            wss=inertia)

        return (score_silhouette,
                score_calinski,
                score_dunn,
                score_davis,
                score_bic)

    def _cv_gen(self,
                data: np.array,
                *args) -> np.array:

        rnd = args[0]

        chunk_size = max(min(int(self.frac * len(data)), self.lmax), self.lmin)

        while True:
            yield (np.random.RandomState(rnd)
                   .permutation(len(data))[:chunk_size])

    def _scoring_fun(
            self,
            *args,
            data: np.array = None,
    ) -> np.array:

        chunks = [next(self._cv_gen(data, i)) for i, _
                  in enumerate(range(self.nsplits))]

        scores_arr = [self._get_scores(data[chunk], *args) for chunk in chunks]

        return np.mean(scores_arr, axis=0)

    def _do_clust(self,
                  data: np.array,
                  start: int = 3,
                  stop: int = 15) -> pd.DataFrame:

        clusts = list(range(start, stop + 1))

        scores_lst = self.do_parallel(
            self._scoring_fun,
            clusts,
            data=data,
            concatenate_result=False)

        scores = dict(zip(clusts, scores_lst))

        ranked = self._ranking_fun(scores)

        ranks = ranked.mean(axis=1)

        return ranks

    def fit(self,
            *args,
            **kwargs) -> int:

        ranks = self._do_clust(*args, **kwargs)

        peaks = scipy.signal.argrelmax(ranks.values, mode='clip')[0]

        if peaks.size == 0:
            peaks = [self._find_knee(range(len(ranks)), ranks)]

        optimal = ranks.iloc[peaks].idxmax()
        peak_scores = ranks.iloc[peaks].to_dict()

        self.optimal_ = optimal
        self.peak_scores_ = peak_scores

    def show(self):

        sns.set_theme(style='whitegrid')

        data = self.ranked_.copy()

        scores_bic = data.pop('BIC elbow')

        plt.vlines(
            x=scores_bic.idxmax(),
            ymin=0,
            ymax=1,
            color='#984ea3',
            alpha=0.5,
            linewidth=3,
            label='BIC elbow',
            zorder=2)

        ax = sns.lineplot(
            data=data,
            markers=True,
            dashes=False,
            zorder=1)

        plt.vlines(
            x=self.peak_scores_.keys(),
            ymin=0,
            ymax=1,
            color='black',
            ls='--',
            linewidth=0.75,
            zorder=3)

        for clust, score in self.peak_scores_.items():
            ax.text(
                x=clust,
                y=0,
                s=f'rank ~{score:.2f}',
                rotation='vertical',
                ha='right',
                va='bottom',
                zorder=3)

        ax.set(xlabel='Number of Clusters', ylabel='Ranked Scores')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax.legend(frameon=False)
        sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1))

        plt.show()
