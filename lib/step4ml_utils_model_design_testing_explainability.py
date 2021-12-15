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
import shap
## for geospatial
import folium
import geopy

import tensorflow as tf
from keras import backend as K


###############################################################################
#                       EXPLAINABILITY                                        #
###############################################################################
'''
Use shap to build an a explainer.
:parameter
    :param model: model instance (after fitting)
    :param X_names: list
    :param X_instance: array of size n x 1 (n,)
    :param X_train: array - if None the model is simple machine learning, if not None then it's a deep learning model
    :param task: string - "classification", "regression"
    :param top: num - top features to display
:return
    dtf with explanations
'''
def explainer_shap(model, X_names, X_instance, X_train=None, task="classification", top=10):
    ## create explainer
    ### machine learning
    if X_train is None:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_instance)
    ### deep learning
    else:
        explainer = shap.DeepExplainer(model, data=X_train[:100])
        shap_values = explainer.shap_values(X_instance.reshape(1,-1))[0].reshape(-1)

    ## plot
    ### classification
    if task == "classification":
        shap.decision_plot(explainer.expected_value, shap_values, link='logit', feature_order='importance',
                           features=X_instance, feature_names=X_names, feature_display_range=slice(-1,-top-1,-1))
    ### regression
    else:
        shap.waterfall_plot(explainer.expected_value[0], shap_values, 
                            features=X_instance, feature_names=X_names, max_display=top)



'''
Use lime to build an a explainer.
:parameter
    :param X_train: array
    :param X_names: list
    :param model: model instance (after fitting)
    :param Y_train: array
    :param X_instance: array of size n x 1 (n,)
    :param task: string - "classification", "regression"
    :param top: num - top features to display
:return
    dtf with explanations
'''
def explainer_lime(X_train, X_names, model, y_train, X_instance, task="classification", top=10):
    if task == "classification":
        explainer = lime_tabular.LimeTabularExplainer(training_data=X_train, feature_names=X_names, class_names=np.unique(y_train), mode=task)
        explained = explainer.explain_instance(X_instance, model.predict_proba, num_features=top)
        dtf_explainer = pd.DataFrame(explained.as_list(), columns=['feature','effect'])
        explained.as_pyplot_figure()
        
    elif task == "regression":
        explainer = lime_tabular.LimeTabularExplainer(training_data=X_train, feature_names=X_names, class_names="Y", mode=task)
        explained = explainer.explain_instance(X_instance, model.predict, num_features=top)
        dtf_explainer = pd.DataFrame(explained.as_list(), columns=['feature','effect'])
        explained.as_pyplot_figure()
    
    return dtf_explainer
