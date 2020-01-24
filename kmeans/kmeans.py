import numpy as np
import sklearn
from sklearn import metrics
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans

digits = load_digits()
data = scale(digits.data)
y = digits.target

# kan gjøres dynamisk https://www.youtube.com/watch?v=WFr2WgN9_xE 2:18:00
k = 10
print(data)
samlpes, features = data.shape

def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                              metric='euclidean')))

clf = KMeans(n_clusters=k, init='random', n_init=100)

bench_k_means(clf, "1", data)
