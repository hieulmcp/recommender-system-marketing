## for data
import numpy as np
import pandas as pd
## for plotting
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
## for statistical tests
import scipy
import statsmodels.formula.api as smf
import statsmodels.api as sm
import ppscore
## for machine learning
from sklearn import preprocessing, impute, utils, linear_model, feature_selection, model_selection, metrics, decomposition, cluster, ensemble
import imblearn
## for deep learning
from tensorflow.keras import models, layers
import minisom
## for explainer
from lime import lime_tabular
#import shap
## for geospatial
import folium
import geopy
#from step3ml_utils_model_design_testing_classification import *
#from step3ml_utils_model_design_testing_regression import *
#from step4ml_utils_model_design_testing_explainability import *
#from step5ml_utils_model_design_testing_visualize_models import *
#from step6ml_utils_model_design_testing_geospatial_analysis import *
import tensorflow as tf
from keras import backend as K


###############################################################################
#                       CLUSTERING (UNSUPERVISED)                             #
###############################################################################
'''
Find the best K-Means with the within-cluster sum of squares (Elbow method).
:paramater
    :param X: array
    :param max_k: num or None- max iteration for wcss
    :param plot: bool - if True plots
:return
    k
'''
def find_best_k(X, max_k=10, plot=True):
    ## iterations
    distortions = [] 
    for i in range(1, max_k+1):
        if len(X) >= i:
            model = cluster.KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
            model.fit(X)
            distortions.append(model.inertia_)

    ## best k: the lowest second derivative
    k = [i*100 for i in np.diff(distortions,2)].index(min([i*100 for i in np.diff(distortions,2)]))

    ## plot
    if plot is True:
        fig, ax = plt.subplots()
        ax.plot(range(1, len(distortions)+1), distortions)
        ax.axvline(k, ls='--', color="red", label="k = "+str(k))
        ax.set(title='The Elbow Method', xlabel='Number of clusters', ylabel="Distortion")
        ax.legend()
        ax.grid(True)
        plt.show()
    return k



'''
Plot clustering in 2D.
:paramater
    :param dtf - dataframe with x1, x2, clusters, centroids
    :param x1: str - column name
    :param x2: str - column name
    :param th_centroids: array - (kmeans) model.cluster_centers_, if None deosn't plot them
'''
def utils_plot_cluster(dtf, x1, x2, th_centroids=None, figsize=(10,5)):
    ## plot points and real centroids
    fig, ax = plt.subplots(figsize=figsize)
    k = dtf["cluster"].nunique()
    sns.scatterplot(x=x1, y=x2, data=dtf, palette=sns.color_palette("bright",k),
                        hue='cluster', size="centroids", size_order=[1,0],
                        legend="brief", ax=ax).set_title('Clustering (k='+str(k)+')')

    ## plot theoretical centroids
    if th_centroids is not None:
        ax.scatter(th_centroids[:,dtf.columns.tolist().index(x1)], 
                   th_centroids[:,dtf.columns.tolist().index(x2)], 
                   s=50, c='black', marker="x")

    ## plot links from points to real centroids
    # if plot_links is True:
    #     centroids_idx = dtf[dtf["centroids"]==1].index
    #     colors = itertools.cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    #     for k, col in zip(range(k), colors):
    #         class_members = dtf["cluster"].values == k
    #         cluster_center = dtf[[x1,x2]].values[centroids_idx[k]]
    #         plt.plot(dtf[[x1,x2]].values[class_members, 0], dtf[[x1,x2]].values[class_members, 1], col + '.')
    #         plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)
    #         for x in dtf[[x1,x2]].values[class_members]:
    #             plt.plot([cluster_center[0], x[0]], 
    #                      [cluster_center[1], x[1]], 
    #                      col)

    ax.grid(True)
    plt.show()



'''
Fit clustering model with K-Means or Affinity Propagation.
:paramater
    :param X: dtf
    :param model: sklearn object
    :param k: num - number of clusters, if None Affinity Propagation is used, else K-Means
    :param lst_2Dplot: list - 2 features to use for a 2D plot, if None it plots only if X is 2D
:return
    model and dtf with clusters
'''
def fit_ml_cluster(X, model=None, k=None, lst_2Dplot=None, figsize=(10,5)):
    ## model
    if (model is None) and (k is None):
        model = cluster.AffinityPropagation()
        print("--- k not defined: using Affinity Propagation ---")
    elif (model is None) and (k is not None):
        model = cluster.KMeans(n_clusters=k, init='k-means++')
        print("---", "k="+str(k)+": using k-means ---")

    ## clustering
    dtf_X = X.copy()
    dtf_X["cluster"] = model.fit_predict(X)
    k = dtf_X["cluster"].nunique()
    print("--- found", k, "clusters ---")
    print(dtf_X.groupby("cluster")["cluster"].count().sort_values(ascending=False))

    ## find real centroids
    closest, distances = scipy.cluster.vq.vq(model.cluster_centers_, dtf_X.drop("cluster", axis=1).values)
    dtf_X["centroids"] = 0
    for i in closest:
        dtf_X["centroids"].iloc[i] = 1
    
    ## plot
    if (lst_2Dplot is not None) or (X.shape[1] == 2):
        lst_2Dplot = X.columns.tolist() if lst_2Dplot is None else lst_2Dplot
        th_centroids = model.cluster_centers_ if "KMeans" in str(model) else None
        utils_plot_cluster(dtf_X, x1=lst_2Dplot[0], x2=lst_2Dplot[1], th_centroids=th_centroids, figsize=figsize)

    return model, dtf_X



'''
Fit a Self Organizing Map neural network.
:paramater
    :param X: dtf
    :param model: minisom instance - if None uses a map of 5*sqrt(n) x 5*sqrt(n) neurons
    :param lst_2Dplot: list - 2 features to use for a 2D plot, if None it plots only if X is 2D
:return
    model and dtf with clusters
'''
def fit_dl_cluster(X, model=None, epochs=100, lst_2Dplot=None, figsize=(10,5)):
    ## model
    model = minisom.MiniSom(x=int(np.sqrt(5*np.sqrt(X.shape[0]))), y=int(np.sqrt(5*np.sqrt(X.shape[0]))), input_len=X.shape[1]) if model is None else model
    scaler = preprocessing.StandardScaler()
    X_preprocessed = scaler.fit_transform(X.values)
    model.train_batch(X_preprocessed, num_iteration=epochs, verbose=False)
    
    ## clustering
    map_shape = (model.get_weights().shape[0], model.get_weights().shape[1])
    print("--- map shape:", map_shape, "---")
    dtf_X = X.copy()
    dtf_X["cluster"] = np.ravel_multi_index(np.array([model.winner(x) for x in X_preprocessed]).T, dims=map_shape)
    k = dtf_X["cluster"].nunique()
    print("--- found", k, "clusters ---")
    print(dtf_X.groupby("cluster")["cluster"].count().sort_values(ascending=False))
    
    ## find real centroids
    cluster_centers = np.array([vec for center in model.get_weights() for vec in center])
    closest, distances = scipy.cluster.vq.vq(cluster_centers, X_preprocessed)
    dtf_X["centroids"] = 0
    for i in closest:
        dtf_X["centroids"].iloc[i] = 1
    
    ## plot
    if (lst_2Dplot is not None) or (X.shape[1] == 2):
        lst_2Dplot = X.columns.tolist() if lst_2Dplot is None else lst_2Dplot
        utils_plot_cluster(dtf_X, x1=lst_2Dplot[0], x2=lst_2Dplot[1], th_centroids=scaler.inverse_transform(cluster_centers), figsize=figsize)

    return model, dtf_X