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
#                   MODEL DESIGN & TESTING - CLASSIFICATION                   #
###############################################################################


'''
Fits a sklearn classification model.
:parameter
    :param model: model object - model to fit (before fitting)
    :param X_train: array
    :param y_train: array
    :param X_test: array
    :param threshold: num - predictions > threshold are 1, otherwise 0 (only for classification)
:return
    model fitted and predictions
'''
def fit_ml_classif(model, X_train, y_train, X_test, threshold=0.5):
    ## model
    model = ensemble.GradientBoostingClassifier() if model is None else model
    
    ## train/test
    model.fit(X_train, y_train)
    predicted_prob = model.predict_proba(X_test)[:,1]
    predicted = (predicted_prob > threshold)
    return model, predicted_prob, predicted



'''
Perform k-fold validation.
'''
def utils_kfold_roc(model, X, y, cv=10, figsize=(10,5)):
    cv = model_selection.StratifiedKFold(n_splits=cv, shuffle=True)
    tprs, aucs = [], []
    mean_fpr = np.linspace(0,1,100)
    fig = plt.figure(figsize=figsize)
    
    i = 1
    for train, test in cv.split(X, y):
        prediction = model.fit(X[train], y[train]).predict_proba(X[test])
        fpr, tpr, t = metrics.roc_curve(y[test], prediction[:, 1])
        tprs.append(scipy.interp(mean_fpr, fpr, tpr))
        roc_auc = metrics.auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i = i+1
        
    plt.plot([0,1], [0,1], linestyle='--', lw=2, color='black')
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='blue', label=r'Mean ROC (AUC = %0.2f )' % (mean_auc), lw=2, alpha=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('K-Fold Validation')
    plt.legend(loc="lower right")
    plt.show()



'''
Find the best classif threshold for metrics: accuracy, precision, recall, f1
'''
def utils_threshold_selection(model, X, y, figsize=(10,5)):
    ## calculate scores for different thresholds
    dic_scores = {'accuracy':[], 'precision':[], 'recall':[], 'f1':[]}
    XX_train, XX_test, yy_train, yy_test = model_selection.train_test_split(X, y, test_size=0.2)
    predicted_prob = model.fit(XX_train, yy_train).predict_proba(XX_test)[:,1] #if "tensorflow" not in type(model) else model.fit(XX_train, yy_train, batch_size=32, epochs=100, verbose=0).model.predict(XX_test)
    thresholds = []
    for threshold in np.arange(0.1, 1, step=0.1):
        predicted = (predicted_prob > threshold)
        thresholds.append(threshold)
        dic_scores["accuracy"].append(metrics.accuracy_score(yy_test, predicted))
        dic_scores["precision"].append(metrics.precision_score(yy_test, predicted))
        dic_scores["recall"].append(metrics.recall_score(yy_test, predicted))
        dic_scores["f1"].append(metrics.f1_score(yy_test, predicted))
    
    ## find best
    dtf_scores = pd.DataFrame(dic_scores).set_index(pd.Index(thresholds))
    for k in dic_scores.keys():
        print(k, "--> best threshold:", round(dtf_scores[dtf_scores[k]==dtf_scores[k].max()][k].index[0], 1))
        
    ## plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.set(title="Threshold Selection", xlabel="Threshold", ylabel="Scores")
    dtf_scores.plot(ax=ax)
    plt.show()


# Xác định ngưỡng tối ưu
from sklearn.metrics import precision_recall_curve, roc_curve
def threshold_precision_recall(y, y_prob):
    # Calculate roc curve
    precision, recall, threshold = precision_recall_curve(y, y_prob)
    scores = (2*precision*recall)/(precision+recall)
    pos = np.argmax(scores)
    return threshold[pos], scores[pos]

def threshold_ROC(y, y_prob):
    # Calculate roc curve
    fpr, tpr, threshold = roc_curve(y, y_prob)
    scores = tpr -fpr
    pos = np.argmax(scores)
    return threshold[pos], scores[pos]


'''
Tunes the hyperparameters of a sklearn classification model.
:parameter
    :param model_base: model object - model istance to tune (before fitting)
    :param param_dic: dict - dictionary of parameters to tune
    :param X_train: array - feature matrix
    :param y_train: array - y vector
    :param scoring: string - "roc_auc", "accuracy", "f1", "precision", "recall"
    :param searchtype: string - "RandomSearch" or "GridSearch"
:return
    model with hyperparams tuned
'''
def tune_classif_model(X_train, y_train, model_base=None, param_dic=None, scoring="f1", searchtype="RandomSearch", n_iter=1000, cv=10, figsize=(10,5)):
    ## params
    model_base = ensemble.GradientBoostingClassifier() if model_base is None else model_base
    param_dic = {'learning_rate':[0.15,0.1,0.05,0.01,0.005,0.001], 'n_estimators':[100,250,500,750,1000,1250,1500,1750], 'max_depth':[2,3,4,5,6,7]} if param_dic is None else param_dic                        
    dic_scores = {'accuracy':metrics.make_scorer(metrics.accuracy_score), 'precision':metrics.make_scorer(metrics.precision_score), 
                  'recall':metrics.make_scorer(metrics.recall_score), 'f1':metrics.make_scorer(metrics.f1_score)}
    
    ## Search
    print("---", searchtype, "---")
    if searchtype == "RandomSearch":
        random_search = model_selection.RandomizedSearchCV(model_base, param_distributions=param_dic, n_iter=n_iter, scoring=dic_scores, refit=scoring).fit(X_train, y_train)
        print("Best Model parameters:", random_search.best_params_)
        print("Best Model "+scoring+":", round(random_search.best_score_, 2))
        model = random_search.best_estimator_
        
    elif searchtype == "GridSearch":
        grid_search = model_selection.GridSearchCV(model_base, param_dic, scoring=dic_scores, refit=scoring).fit(X_train, y_train)
        print("Best Model parameters:", grid_search.best_params_)
        print("Best Model mean "+scoring+":", round(grid_search.best_score_, 2))
        model = grid_search.best_estimator_
    
    ## K fold validation
    print("")
    print("--- Kfold Validation ---")
    Kfold_base = model_selection.cross_validate(estimator=model_base, X=X_train, y=y_train, cv=cv, scoring=dic_scores)
    Kfold_model = model_selection.cross_validate(estimator=model, X=X_train, y=y_train, cv=cv, scoring=dic_scores)
    for score in dic_scores.keys():
        print(score, "mean - base model:", round(Kfold_base["test_"+score].mean(),2), " --> best model:", round(Kfold_model["test_"+score].mean()))
    utils_kfold_roc(model, X_train, y_train, cv=cv, figsize=figsize)
    
    ## Threshold analysis
    print("")
    print("--- Threshold Selection ---")
    utils_threshold_selection(model, X_train, y_train, figsize=figsize)
    
    return model



'''
Plot loss and metrics of keras training.
'''
def utils_plot_keras_training(training):
    metrics = [k for k in training.history.keys() if ("loss" not in k) and ("val" not in k)]
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(15,3))
    
    ## training
    ax[0].set(title="Training")
    ax11 = ax[0].twinx()
    ax[0].plot(training.history['loss'], color='black')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss', color='black')
    for metric in metrics:
        ax11.plot(training.history[metric], label=metric)
    ax11.set_ylabel("Score", color='steelblue')
    ax11.legend()
    
    ## validation
    ax[1].set(title="Validation")
    ax22 = ax[1].twinx()
    ax[1].plot(training.history['val_loss'], color='black')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss', color='black')
    for metric in metrics:
        ax22.plot(training.history['val_'+metric], label=metric)
    ax22.set_ylabel("Score", color="steelblue")
    plt.show()



'''
Fits a keras artificial/deep neural network.
:parameter
    :param X_train: array
    :param y_train: array
    :param X_test: array
    :param batch_size: num - keras batch
    :param epochs: num - keras epochs
    :param threshold: num - predictions > threshold are 1, otherwise 0
:return
    model fitted and predictions
'''
def fit_dl_classif(X_train, y_train, X_test, model=None, batch_size=32, epochs=100, threshold=0.5):
    ## model
    if model is None:
        ### define F1 metrics for Keras
        def Recall(y, y_hat):
            true_positives = K.sum(K.round(K.clip(y * y_hat, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall

        def Precision(y, y_hat):
            true_positives = K.sum(K.round(K.clip(y * y_hat, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_hat, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision

        def F1(y, y_hat):
            precision = Precision(y, y_hat)
            recall = Recall(y, y_hat)
            return 2*((precision*recall)/(precision+recall+K.epsilon()))

        ### build ann
        n_features = X_train.shape[1]
        n_neurons = int(round((n_features + 1)/2))
        model = models.Sequential([
            layers.Dense(input_dim=n_features, units=n_neurons, kernel_initializer='uniform', activation='relu'),
            layers.Dropout(rate=0.2),
            layers.Dense(units=n_neurons, kernel_initializer='uniform', activation='relu'),
            layers.Dropout(rate=0.2),
            layers.Dense(units=1, kernel_initializer='uniform', activation='sigmoid') ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy',F1])
        print(model.summary())
    
    ## train
    verbose = 0 if epochs > 1 else 1
    training = model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=verbose, validation_split=0.3)
    if epochs > 1:
        utils_plot_keras_training(training)
    
    ## test
    predicted_prob = training.model.predict(X_test)
    predicted = (predicted_prob > threshold)
    return training.model, predicted_prob, predicted



'''
Evaluates a model performance.
:parameter
    :param y_test: array
    :param predicted: array
    :param predicted_prob: array
    :param show_thresholds: bool - if True annotates thresholds on the curves
'''
def evaluate_classif_model(y_test, predicted, predicted_prob, show_thresholds=True, figsize=(25,5)):
    classes = np.unique(y_test)
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=figsize)
    
    ## Accuray e AUC
    accuracy = metrics.accuracy_score(y_test, predicted)
    auc = metrics.roc_auc_score(y_test, predicted_prob)
    print("Accuracy (overall correct predictions):",  round(accuracy,2))
    print("Auc:", round(auc,2))
    
    ## Precision e Recall
    recall = metrics.recall_score(y_test, predicted)  #= true 1s / all 1s in test (got and missed)
    precision = metrics.precision_score(y_test, predicted)  #= true 1s / all 1s predicted (true 1s + false 1s)
    f1 = metrics.f1_score(y_test, predicted)  #=2 * (precision * recall) / (precision + recall)
    print("Recall (all 1s predicted right):", round(recall,2))  #true positive rate, how many 1s I got
    print("Precision (confidence when predicting a 1):", round(precision,2))  #how many 1s were really 1s
    print("F1 score:", round(f1,2))
    print("Detail:")
    print(metrics.classification_report(y_test, predicted, target_names=[str(i) for i in classes]))
       
    ## Plot confusion matrix
    cm = metrics.confusion_matrix(y_test, predicted, labels=classes)
    sns.heatmap(cm, annot=True, annot_kws={"size":15}, fmt='d', ax=ax[0], cmap=plt.cm.Blues, cbar=False)
    ax[0].set(xlabel="Pred", ylabel="True", title="Confusion matrix")
    ax[0].set_yticklabels(labels=classes, rotation=0)
 
    ## Plot roc
    fpr, tpr, thresholds = metrics.roc_curve(y_test, predicted_prob)    
    ax[1].plot(fpr, tpr, color='darkorange', lw=3, label='area = %0.2f' % metrics.auc(fpr, tpr))
    ax[1].plot([0,1], [0,1], color='navy', lw=3, linestyle='--')
    ax[1].hlines(y=recall, xmin=-0.05, xmax=1-cm[0,0]/(cm[0,0]+cm[0,1]), color='red', linestyle='--', alpha=0.7, label="chosen threshold")
    ax[1].vlines(x=1-cm[0,0]/(cm[0,0]+cm[0,1]), ymin=0, ymax=recall, color='red', linestyle='--', alpha=0.7)
    ax[1].set(xlim=[-0.05,1], ylim=[0.0,1.05], xlabel='False Positive Rate', ylabel="True Positive Rate (Recall)", title="Receiver operating characteristic")     
    ax[1].legend(loc="lower right")
    ax[1].grid(True)
    if show_thresholds is True:
        thres_in_plot = []
        for i,t in enumerate(thresholds):
            t = np.round(t,1)
            if t not in thres_in_plot:
                ax[1].annotate(t, xy=(fpr[i],tpr[i]), xytext=(fpr[i],tpr[i]), textcoords='offset points', ha='left', va='bottom')
                thres_in_plot.append(t)
    
    ## Plot precision-recall curve
    precisions, recalls, thresholds = metrics.precision_recall_curve(y_test, predicted_prob)
    ax[2].plot(recalls, precisions, color='darkorange', lw=3, label='area = %0.2f' % metrics.auc(recalls, precisions))
    ax[2].plot([0,1], [(cm[1,0]+cm[1,0])/len(y_test), (cm[1,0]+cm[1,0])/len(y_test)], linestyle='--', color='navy', lw=3)
    ax[2].hlines(y=precision, xmin=0, xmax=recall, color='red', linestyle='--', alpha=0.7, label="chosen threshold")
    ax[2].vlines(x=recall, ymin=0, ymax=precision, color='red', linestyle='--', alpha=0.7)
    ax[2].set(xlim=[0.0,1.05], ylim=[0.0,1.05], xlabel='Recall', ylabel="Precision", title="Precision-Recall curve")
    ax[2].legend(loc="lower left")
    ax[2].grid(True)
    if show_thresholds is True:
        thres_in_plot = []
        for i,t in enumerate(thresholds):
            t = np.round(t,1)
            if t not in thres_in_plot:
                ax[2].annotate(np.round(t,1), xy=(recalls[i],precisions[i]), xytext=(recalls[i],precisions[i]), textcoords='offset points', ha='right', va='bottom')
                thres_in_plot.append(t)

    plt.show()

##############################################################################################################################################################################

##############################################################################################################################################################################
from lazypredict.Supervised import LazyClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyRegressor
### Importing dataset available in sklearn
from sklearn import datasets
from sklearn.utils import shuffle
import numpy as np

def check_model_final_lazy(df, y, test_size=0.3, random_state=42, choose='model'):
    df_X = df.drop([y], axis=1)
    df_y = df[y]
    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=test_size, random_state = random_state)
    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    if choose == 'model':
        return models
    elif choose == 'predictions':
        return predictions
     
