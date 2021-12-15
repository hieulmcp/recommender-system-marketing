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

# Mục đích lựa chọn các thuộc tính của bài toán
###############################################################################
#                  FEATURES SELECTION                                         #
###############################################################################
'''
Performs features selections: by correlation (keeping the lowest p-value) and by lasso.
:prameter
    :param dtf: dataframe - feature matrix dtf
    :param y: str - name of the dependent variable
    :param top: num - number of top features
    :param task: str - "classification" or "regression"
:return
    dic with lists of features to keep.
'''     
def features_selection(dtf, y, top=10, task="classification", figsize=(20,10)):
    try:
        dtf_X = dtf.drop(y, axis=1)
        feature_names = dtf_X.columns
        
        ## p-value (one way anova F-test)
        model = feature_selection.f_classif if task=="classification" else feature_selection.f_regression
        selector = feature_selection.SelectKBest(score_func=model, k=top).fit(dtf_X.values, dtf[y].values)
        pvalue_selected_features = feature_names[selector.get_support()]
        
        ## regularization (classif-->lasso (l1), regr-->ridge (l2))
        model = linear_model.LogisticRegression(C=1, penalty="l1", solver='liblinear') if task=="classification" else linear_model.Ridge(alpha=1.0, fit_intercept=True)
        selector = feature_selection.SelectFromModel(estimator=model, max_features=top).fit(dtf_X.values, dtf[y].values)
        regularization_selected_features = feature_names[selector.get_support()]
        
        ## plot
        dtf_features = pd.DataFrame({"features":feature_names})
        dtf_features["p_value"] = dtf_features["features"].apply(lambda x: "p_value" if x in pvalue_selected_features else "")
        dtf_features["num1"] = dtf_features["features"].apply(lambda x: 1 if x in pvalue_selected_features else 0)
        dtf_features["regularization"] = dtf_features["features"].apply(lambda x: "regularization" if x in regularization_selected_features else "")
        dtf_features["num2"] = dtf_features["features"].apply(lambda x: 1 if x in regularization_selected_features else 0)
        dtf_features["method"] = dtf_features[["p_value","regularization"]].apply(lambda x: (x[0]+" "+x[1]).strip(), axis=1)
        dtf_features["selection"] = dtf_features["num1"] + dtf_features["num2"]
        dtf_features["method"] = dtf_features["method"].apply(lambda x: x.split()[0]+" + "+x.split()[1] if len(x.split())==2 else x)
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(y="features", x="selection", hue="method", data=dtf_features.sort_values("selection", ascending=False), ax=ax, dodge=False)
               
        join_selected_features = list(set(pvalue_selected_features).intersection(regularization_selected_features))
        return {"p_value":pvalue_selected_features, "regularization":regularization_selected_features, "join":join_selected_features}
    
    except Exception as e:
        print("--- got error ---")
        print(e)



'''
Computes features importance.
:parameter
    :param X: array
    :param X_names: list
    :param model: model istance (after fitting)
    :param figsize: tuple - plot setting
:return
    dtf with features importance
'''
def features_importance(X, y, X_names, model=None, task="classification", figsize=(10,10)):
    ## model
    if model is None:
        if task == "classification":
            model = ensemble.GradientBoostingClassifier()  
        elif task == "regression":
            model = ensemble.GradientBoostingRegressor()
    model.fit(X,y)
    print("--- model used ---")
    print(model)
    
    ## importance dtf
    importances = model.feature_importances_
    dtf_importances = pd.DataFrame({"IMPORTANCE":importances, "VARIABLE":X_names}).sort_values("IMPORTANCE", ascending=False)
    dtf_importances['cumsum'] = dtf_importances['IMPORTANCE'].cumsum(axis=0)
    dtf_importances = dtf_importances.set_index("VARIABLE")
    
    ## plot
    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=figsize)
    fig.suptitle("Features Importance", fontsize=20)
    ax[0].title.set_text('variables')
    dtf_importances[["IMPORTANCE"]].sort_values(by="IMPORTANCE").plot(kind="barh", legend=False, ax=ax[0]).grid(axis="x")
    ax[0].set(ylabel="")
    ax[1].title.set_text('cumulative')
    dtf_importances[["cumsum"]].plot(kind="line", linewidth=4, legend=False, ax=ax[1])
    ax[1].set(xlabel="", xticks=np.arange(len(dtf_importances)), xticklabels=dtf_importances.index)
    plt.xticks(rotation=70)
    plt.grid(axis='both')
    plt.show()
    return dtf_importances.reset_index()