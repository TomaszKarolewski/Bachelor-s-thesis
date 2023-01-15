# Bachelor-s-thesis
Analysis of the COVID-19 development dynamics in European countries

The following thesis presents and discusses statistical methods for processing, transforming and analyzing data used
in machine learning problems.

The subject of the work is COVID-19 data from European countries describing the levels of infections, collected by the Johns Hopkins University. We present each step of the classical exploratory analysis procedures including smoothing methods based on rolling average approach and dimension reduction using principal component analysis. Then, the processed data was subject to cluster analysis using two state of the art methods: k-means and hierarchical clustering. Additionally, apart from the classic statistical approaches, an original method for information extraction, describing waves of new infections, was introduced and applied. The method aims to describe each wave as a triangle given by its height, height spot and base. Moreover, an approach to prediction of the end of a wave was made using the Random Forest algorithm.

One of the main findings of the analyzes is the importance of the geographical proximity between countries belonging to the same cluster that characterizes a certain course of the epidemic. Finally, the results are discussed and further modifications to the model and its improvements are considered.

### Below are attached images summarising my results
Choosing the right number of clusters with Silhouette score

![Silhouette](/graphics/grid_kmeans_pca+triangles.png?raw=true "Silhouette score")

Visualization of countries belonging to given clusters

![choropleth](/graphics/choropleth_kmeans_pca_trian.png?raw=true "Clusters on map")

Visualization of new cases number in a given cluster

![lines](/graphics/line_kmeans_pca_trian.png?raw=true "Cluster number of cases")

An expression of waves with triangles, along with a prediction of the end of the last wave
![triangles](/graphics/aus.png?raw=true "Wave length prediction")
