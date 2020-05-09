# klasteryzacja
from matplotlib import pyplot
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score


class ClusteringAlgorithms:
    def getIris(self):
        self.iris = load_iris()
        # print(self.iris)
    def clustering(self, cls):
        y_pred = cls.fit_predict(self.iris['data'])
        clusters = [list(y_pred).index(0), list(y_pred).index(1), list(y_pred).index(2)]
        if(clusters[0] < clusters[1] and clusters[0] < clusters[2]):
            clusters[0] = 0
            if(clusters[1] < clusters[2]):
                clusters[1] = 1
                clusters[2] = 2
            else:
                clusters[1] = 2
                clusters[2] = 1
        elif(clusters[1] < clusters[0] and clusters[1] < clusters[2]):
            clusters[1] = 0
            if (clusters[0] < clusters[2]):
                clusters[0] = 1
                clusters[2] = 2
            else:
                clusters[0] = 2
                clusters[2] = 1
        elif (clusters[2] < clusters[0] and clusters[2] < clusters[1]):
            clusters[2] = 0
            if (clusters[0] < clusters[1]):
                clusters[0] = 1
                clusters[1] = 2
            else:
                clusters[0] = 2
                clusters[1] = 1

        clusters_ordered = sorted(clusters)
        # print(clusters)
        # print(clusters_ordered)
        y_cls = []
        index = 0
        while(index < len(y_pred)):
            cls_index = clusters_ordered.index(y_pred[index])
            y_cls.append(clusters[cls_index])
            index += 1
        print(y_pred)
        print(y_cls)
        print(accuracy_score(self.iris['target'], y_cls))
    def getDBSCANN(self, clf):
        y_pred = clf.fit_predict(self.iris['data'])
        print(y_pred)
    def plotRestults(self, cls_list, column1, column2):
        y_preds = []
        pyplot.figure()
        pyplot.subplot(411)
        pyplot.scatter(self.iris['data'][:,column1], self.iris['data'][:,column2])
        pyplot.title("Init samples")
        pyplot.xlabel("x1")
        pyplot.ylabel("x2")
        subplot_number = 412
        for cls_name in cls_list.keys():
            y_pred = cls_list[cls_name].fit_predict(self.iris['data'])
            pyplot.subplot(subplot_number)
            pyplot.scatter(self.iris['data'][:, column1], self.iris['data'][:, column2], c=y_pred)
            pyplot.title("Clustering: " + cls_name)
            pyplot.xlabel("x1")
            pyplot.ylabel("x2")
            subplot_number += 1
        pyplot.show()

c = ClusteringAlgorithms()
c.getIris()
c.clustering(KMeans(n_clusters=3, init='random'))
c.clustering(KMeans(n_clusters=3, init='k-means++'))
c.clustering(AgglomerativeClustering(n_clusters=3, linkage='single'))
c.clustering(AgglomerativeClustering(n_clusters=3, linkage='single', affinity='manhattan'))
c.getDBSCANN(DBSCAN())
c.plotRestults(
    {"KMeans++" : KMeans(n_clusters=3, init='k-means++'),
    "AgglomerativeClustering" : AgglomerativeClustering(n_clusters=3, linkage='single'),
    "DBSCAN" : DBSCAN()},
    0,1
)