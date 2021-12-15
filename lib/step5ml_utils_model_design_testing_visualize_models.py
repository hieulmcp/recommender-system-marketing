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


import tensorflow as tf
from keras import backend as K


###############################################################################
#                     VISUALIZE MODELS                                        #
###############################################################################
'''
Decomposes the feture matrix of train and test.
:parameter
    :param X_train: array
    :param X_test: array
    :param n_features: num - how many dimensions you want
:return
    dict with new train and test, and the model 
'''
def utils_dimensionality_reduction(X_train, X_test, n_features=2):
    model = decomposition.PCA(n_components=n_features)
    X_train = model.fit_transform(X_train)
    X_test = model.transform(X_test)
    return X_train, X_test, model



'''
Plots a 2d classification model result.
:parameter
    :param X_train: array
    :param y_train: array
    :param X_test: array
    :param y_test: array
    :param model: model istance (before fitting)
'''
def plot2d_classif_model(X_train, y_train, X_test, y_test, model=None, annotate=False, figsize=(10,5)):
    ## n features > 2d
    if X_train.shape[1] > 2:
        print("--- reducing dimensions to 2 ---")
        X_train, X_test, pca = utils_dimensionality_reduction(X_train, X_test, n_features=2)
     
    ## fit 2d model
    print("--- fitting 2d model ---")
    model_2d = ensemble.GradientBoostingClassifier() if model is None else model
    model_2d.fit(X_train, y_train)
    
    ## plot predictions
    print("--- plotting test set ---")
    from matplotlib.colors import ListedColormap
    colors = {np.unique(y_test)[0]:"black", np.unique(y_test)[1]:"green"}
    X1, X2 = np.meshgrid(np.arange(start=X_test[:,0].min()-1, stop=X_test[:,0].max()+1, step=0.01),
                         np.arange(start=X_test[:,1].min()-1, stop=X_test[:,1].max()+1, step=0.01))
    fig, ax = plt.subplots(figsize=figsize)
    Y = model_2d.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)
    ax.contourf(X1, X2, Y, alpha=0.5, cmap=ListedColormap(list(colors.values())))
    ax.set(xlim=[X1.min(),X1.max()], ylim=[X2.min(),X2.max()], title="Classification regions")
    for i in np.unique(y_test):
        ax.scatter(X_test[y_test==i, 0], X_test[y_test==i, 1], c=colors[i], label="true "+str(i))  
    if annotate is True:
        for n,i in enumerate(y_test):
            ax.annotate(n, xy=(X_test[n,0], X_test[n,1]), textcoords='offset points', ha='left', va='bottom')
    plt.legend()
    plt.show()
    

    
'''
Plot 3d regression plane.
'''
def plot3d_regr_model(X_train, y_train, X_test, y_test, scalerY=None, model=None, rotate=(0,0), figsize=(10,5)):
    ## n features > 2d
    if X_train.shape[1] > 2:
        print("--- reducing dimensions to 3 ---")
        X_train, X_test, pca  = utils_dimensionality_reduction(X_train, X_test, n_features=2)
    
    ## fit 2d model
    print("--- fitting 2d model ---")
    model_2d = linear_model.LinearRegression() if model is None else model
    model_2d.fit(X_train, y_train)
    
    ## plot predictions
    print("--- plotting test set ---")
    from mpl_toolkits.mplot3d import Axes3D
    ax = Axes3D(plt.figure(figsize=figsize), elev=rotate[0], azim=rotate[1])
    ax.scatter(X_test[:,0], X_test[:,1], y_test, color="black")
    X1 = np.array([[X_test.min(), X_test.min()], [X_test.max(), X_test.max()]])
    X2 = np.array([[X_test.min(), X_test.max()], [X_test.min(), X_test.max()]])
    Y = model_2d.predict(np.array([[X_test.min(), X_test.min(), X_test.max(), X_test.max()], 
                                   [X_test.min(), X_test.max(), X_test.min(), X_test.max()]]).T).reshape((2,2))
    Y = scalerY.inverse_transform(Y) if scalerY is not None else Y
    ax.plot_surface(X1, X2, Y, alpha=0.5)
    ax.set(zlabel="Y", title="Regression plane", xticklabels=[], yticklabels=[])
    plt.show()



'''
Extract info for each layer in a keras model.
'''
def utils_nn_config(model):
    lst_layers = []
    for layer in model.layers:
        if "drop" in layer.name:
            dic_layer = {"name":layer.name, "in":int(layer.input.shape[-1]), "neurons":0, 
                         "out":int(layer.output.shape[-1]), "activation":None,
                         "params":0, "bias":0} 
        else:
            dic_layer = {"name":layer.name, "in":int(layer.input.shape[-1]), "neurons":layer.units, 
                         "out":int(layer.output.shape[-1]), "activation":layer.get_config()["activation"],
                         "params":layer.get_weights()[0], "bias":layer.get_weights()[1]} 
        lst_layers.append(dic_layer)
    return lst_layers



'''
Plot the structure of a keras neural network.
'''
def visualize_nn(model, titles=False, figsize=(10,8)):
    ## get layers info
    lst_layers = utils_nn_config(model)
    layer_sizes, layer_infos = [], []
    for i,layer in enumerate(lst_layers):
        if "drop" not in layer["name"]:
            layer_sizes.append(layer["in"] if i==0 else layer["out"])
            layer_infos.append( (layer["activation"], layer["params"], layer["bias"]) )
    
    ## fig setup
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()
    ax.set(title="Neural Network structure")
    ax.axis('off')
    left, right, bottom, top = 0.1, 0.9, 0.1, 0.9
    x_space = (right-left) / float(len(layer_sizes)-1)
    y_space = (top-bottom) / float(max(layer_sizes))
    p = 0.025
    
    ## nodes
    for i,n in enumerate(layer_sizes):
        top_on_layer = y_space*(n-1)/2.0 + (top+bottom)/2.0
        
        ### add titles
        if titles is True:
            if i == 0:
                plt.text(x=left, y=top, fontsize=10, s="Input layer")
                plt.text(x=left, y=top-p, fontsize=8, s="activation: "+str(layer_infos[i][0]))
                plt.text(x=left, y=top-2*p, fontsize=8,
                         s="inputs:"+str(layer_infos[i][1].shape[0]))
            elif i == len(layer_sizes)-1:
                plt.text(x=right, y=top, fontsize=10, s="Output layer")
                plt.text(x=right, y=top-p, fontsize=8, s="activation: "+str(layer_infos[i][0]))
                plt.text(x=right, y=top-2*p, fontsize=8,
                         s="output:"+str(layer_infos[i][1].shape[1])+" comb + bias")
            else:
                plt.text(x=left+i*x_space, y=top, fontsize=10, s="Hidden layer "+str(i))
                plt.text(x=left+i*x_space, y=top-p, fontsize=8, s="activation: "+str(layer_infos[i][0]))
                plt.text(x=left+i*x_space, y=top-2*p, fontsize=8,
                         s="neurons:"+str(layer_infos[i][1].shape[0])+" comb + bias")
        
        ### circles
        for m in range(n):
            color = "limegreen" if (i == 0) or (i == len(layer_sizes)-1) else "red" 
            circle = plt.Circle(xy=(left+i*x_space, top_on_layer-m*y_space-4*p), radius=y_space/4.0, color=color, ec='k', zorder=4)
            ax.add_artist(circle)
            
            ### add text
            if i == 0:
                plt.text(x=left-4*p, y=top_on_layer-m*y_space-4*p, fontsize=10, s=r'$X_{'+str(m+1)+'}$')
            elif i == len(layer_sizes)-1:
                plt.text(x=right+4*p, y=top_on_layer-m*y_space-4*p, fontsize=10, s=r'$y_{'+str(m+1)+'}$')
            else:
                plt.text(x=left+i*x_space+p, y=top_on_layer-m*y_space+(y_space/8.+0.01*y_space)-4*p, fontsize=10, s=r'$H_{'+str(m+1)+'}$')
    
    ## links
    for i, (n_a, n_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = y_space*(n_a-1)/2. + (top+bottom)/2. -4*p
        layer_top_b = y_space*(n_b-1)/2. + (top+bottom)/2. -4*p
        for m in range(n_a):
            for o in range(n_b):
                line = plt.Line2D([i*x_space+left, (i+1)*x_space+left], [layer_top_a-m*y_space, layer_top_b-o*y_space], c='k', alpha=0.5)
                ax.add_artist(line)
    plt.show()
