import re
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import os
import random

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import PredefinedSplit, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import argparse

def parse_args():
    parser = argparse.ArgumentParser("fits a RF to the data")
    parser.add_argument("--q", default="all", help="scaling factors")
    parser.add_argument("--base", default="all", help="bases to be used for benford computation")
    parser.add_argument("--type", default="full", help="type of processed signal can be:\n"
            "full: the whole signal is processed\n"
            "sound: processes only parts with high enough energy\n"
            "silence: processes only parts with low enough energy\n"
    )
    return parser.parse_args()

def plot_confusion_matrix(dataset, y_true, y_pred, labels, cmap):

  cm = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize='true')
  class_names = labels.keys()
  df_cm = pd.DataFrame(cm,index = class_names, columns = class_names)

  plt.figure(figsize = (15,10))
  sn.heatmap(df_cm, annot=True, cmap=cmap)
  plt.title('Confusion Matrix {}'.format(dataset))
  plt.xlabel('Predicted')
  plt.ylabel('True')
  plt.show()

def select_useful_features(X, q, base):

    X = X.to_numpy()

    if q != "all":
        q = int(q)
        delta = X.shape[1] // 4
        X = X.reshape((-1, 4, X.shape[1] // 4))
        X = X[:, :q]

    if base != "all":
        base = int(base)
        X = X.reshape((-1, X.shape[1] // 13, 13))
        X = X[:, (0 if base==10 else 1)::2]

    X = X.reshape([X.shape[0], -1])
    
    return X



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    args = parse_args()
    q = args.q
    base = args.base
    signal_type = args.type
    # importing the data
    train_data = pd.read_csv(f"datasets/processed/ASVspoof-LA/df_train_{signal_type}.csv")
    dev_data = pd.read_csv(f"datasets/processed/ASVspoof-LA/df_dev_{signal_type}.csv")
    eval_data = pd.read_csv(f"datasets/processed/ASVspoof-LA/df_eval_{signal_type}.csv")

    train_data = train_data[train_data["length"] > 5000]
    dev_data = dev_data[dev_data["length"] > 5000]
    eval_data = eval_data[eval_data["length"] > 5000]

    train_data = train_data.drop(columns="length")
    dev_data = dev_data.drop(columns="length")
    eval_data = eval_data.drop(columns="length")

    train_data = train_data.dropna()
    dev_data = dev_data.dropna()
    eval_data = eval_data.dropna()

    #  train on same number of natural and generated audio
    df_train_nat = train_data.loc[train_data['system ID'] == '-']
    df_train_gen_A01 = train_data.loc[train_data['system ID'] == 'A01'].sample(frac=1, random_state=np.random.randint(0, 10000))[0:430]
    df_train_gen_A02 = train_data.loc[train_data['system ID'] == 'A02'].sample(frac=1, random_state=np.random.randint(0, 10000))[0:430]
    df_train_gen_A03 = train_data.loc[train_data['system ID'] == 'A03'].sample(frac=1, random_state=np.random.randint(0, 10000))[0:430]
    df_train_gen_A04 = train_data.loc[train_data['system ID'] == 'A04'].sample(frac=1, random_state=np.random.randint(0, 10000))[0:430]
    df_train_gen_A05 = train_data.loc[train_data['system ID'] == 'A05'].sample(frac=1, random_state=np.random.randint(0, 10000))[0:430]
    df_train_gen_A06 = train_data.loc[train_data['system ID'] == 'A06'].sample(frac=1, random_state=np.random.randint(0, 10000))[0:430]
    df_train_gen = pd.concat([df_train_gen_A01, df_train_gen_A02, df_train_gen_A03, df_train_gen_A04, df_train_gen_A05, df_train_gen_A06], axis=0)

    df_train = pd.concat([df_train_nat, df_train_gen], axis=0)  # create the new training dataset

    # extract labels and system IDs
    y_train = df_train['label'].to_numpy()

    # remove names, System ID and labels columns
    X_train = df_train.iloc[:, :-2]
    X_train = X_train.iloc[:, 1:]
    X_train = select_useful_features(X_train, q, base)

    # split training set into actual training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, shuffle=True, train_size=0.8)

    # GRID SEARCH
    X = np.concatenate((X_train, X_valid), axis=0) # concatenate training and validation set
    y = np.concatenate((y_train, y_valid), axis=0)

    # to avoid k-fold cross validation in GridSearchCV and use instead the training-validation split defined before
    # Create a list where train data indices are -1 and validation data indices are 0
    split_index = [-1] * len(X_train) + [0] * len(X_valid)

    # Use the list to create PredefinedSplit
    pds = PredefinedSplit(test_fold=split_index)

    n_estimators = [10, 100, 500, 1000]  # number of trees in the forest of the model
    criterion = ['gini', 'entropy']

    params = dict(n_estimators=n_estimators, criterion=criterion)

    clf = RandomForestClassifier(n_jobs=os.cpu_count(), random_state=seed)

    y_dev = dev_data["label"].value_counts()
    gs = GridSearchCV(clf, params, cv=pds, scoring='accuracy', verbose=10)

    # fitting the model (on X and y)
    gs.fit(X, y)

    # print best parameter after tuning
    print("Best parameters:", gs.best_params_)
    print("Best validation accuracy:", gs.best_score_)

    # best configurations
    best_clf = gs.best_estimator_
    best_clf.fit(X_train, y_train)

    #trainig set
    train_algs = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06']

    final_str = ""

    for alg in train_algs:

        label = re.findall("\d+", alg)[0]
        labels = {'Natural':0, alg : label}

        df_train_nat = train_data.loc[train_data['system ID'] == '-']
        df_train_gen = train_data.loc[train_data['system ID'] == alg]
        df_train = pd.concat([df_train_nat, df_train_gen], axis=0)

        y_train = df_train['label'].to_numpy()
        X_train = df_train.iloc[:, :-2]
        X_train = X_train.iloc[:, 1:]
        X_train = select_useful_features(X_train, q, base)

        y_train_pred = best_clf.predict(X_train)
        y_train_pred = np.array(y_train_pred)  # predicted labels

        final_str += f" & {accuracy_score(y_train, y_train_pred):.3f}"
        print("Train {} accuracy:".format(alg), accuracy_score(y_train, y_train_pred), "F1 score:", f1_score(y_train, y_train_pred, average='macro'))

    print(final_str)
    final_str = ""
        # plot_confusion_matrix('Development ({}) set'.format(alg), y_train, y_train_pred, labels, 'Reds')
    # Development set
    dev_algs = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06']

    for alg in dev_algs:

        label = re.findall("\d+", alg)[0]
        labels = {'Natural':0, alg : label}

        df_dev_nat = dev_data.loc[dev_data['system ID'] == '-']
        df_dev_gen = dev_data.loc[dev_data['system ID'] == alg]
        df_dev = pd.concat([df_dev_nat, df_dev_gen], axis=0)

        y_dev = df_dev['label'].to_numpy()
        X_dev = df_dev.iloc[:, :-2]
        X_dev = X_dev.iloc[:, 1:]
        X_dev = select_useful_features(X_dev, q, base)

        y_dev_pred = best_clf.predict(X_dev)
        y_dev_pred = np.array(y_dev_pred)  # predicted labels

        final_str += f" & {accuracy_score(y_dev, y_dev_pred):.3f}"
        print("Development {} accuracy:".format(alg), accuracy_score(y_dev, y_dev_pred), "F1 score:", f1_score(y_dev, y_dev_pred, average='macro'))

        # plot_confusion_matrix('Development ({}) set'.format(alg), y_dev, y_dev_pred, labels, 'Reds')
    y_dev = dev_data["label"].to_numpy()
    x_dev = select_useful_features(dev_data.iloc[:, 1:-2], q, base)
    y_dev_pred = best_clf.predict(x_dev)
    sample_weights = 1/np.sum(y_dev) * y_dev + 1/np.sum(1-y_dev)*(1-y_dev)
    print(f"dev accuracy {accuracy_score(y_dev, y_dev_pred, sample_weight=sample_weights):.3f}")



    # Evaluation set
    print(final_str)
    final_str = ""
    eval_algs = ['A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19']

    for alg in eval_algs:

        label = re.findall("\d+", alg)[0]
        labels = {'Natural': 0, alg: label}

        df_eval_nat = eval_data.loc[eval_data['system ID'] == '-']
        df_eval_gen = eval_data.loc[eval_data['system ID'] == alg]
        df_eval = pd.concat([df_eval_nat, df_eval_gen], axis=0)

        # remove nan values
        list = (df_eval[df_eval.isna().any(axis=1)]['name'])
        df_eval = df_eval[df_eval.name.isin(list) == False]

        y_eval = df_eval['label'].to_numpy()
        X_eval = df_eval.iloc[:, :-2]
        X_eval = X_eval.iloc[:, 1:]
        X_eval = select_useful_features(X_eval, q, base)

        y_eval_pred = best_clf.predict(X_eval)
        y_eval_pred = np.array(y_eval_pred)  # predicted labels

        final_str += f" & {accuracy_score(y_eval, y_eval_pred):.3f}"
        print("Evaluation {} accuracy:".format(alg), accuracy_score(y_eval, y_eval_pred), "F1 score:",f1_score(y_eval, y_eval_pred, average='macro'))

    print(final_str)
    y_eval = eval_data["label"].to_numpy()
    x_eval = select_useful_features(eval_data.iloc[:, 1:-2], q, base)
    y_eval_pred = best_clf.predict(x_eval)
    sample_weights = 1/np.sum(y_eval) * y_eval + 1/np.sum(1-y_eval)*(1-y_eval)
    print(f"eval accuracy {accuracy_score(y_eval, y_eval_pred, sample_weight=sample_weights):.3f}")
        # plot_confusion_matrix('Evaluation ({}) set'.format(alg), y_eval, y_eval_pred, labels, 'Reds')



