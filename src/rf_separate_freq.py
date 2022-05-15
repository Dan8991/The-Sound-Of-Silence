import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # importing the data
    data = pd.read_csv("datasets/df_inter_ljspeech.csv")

    labels = data['label'].tolist()

    # remove names and labels columns
    features = data.iloc[:, :-1]
    features = features.iloc[:, 1:]

    X_train, X_test, y_train, y_test = train_test_split(features, labels, shuffle=False)

    clf = RandomForestClassifier()

    # Train the model using the training sets
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))







