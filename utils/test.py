from sklearn.datasets import make_classification
import tsneview as tsne

features, labels = make_classification(n_samples=1000, n_features=100, n_classes=4, n_clusters_per_class=1,
                                       random_state=42)
print(labels)
tsne.tsne_visil(features, labels, 4)
