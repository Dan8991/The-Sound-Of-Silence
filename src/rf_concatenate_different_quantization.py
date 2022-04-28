from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def concatenate_features(data):

    # converting column data to list
    mse = data['mse'].tolist()
    kl = data['kl'].tolist()
    reny = data['reny'].tolist()
    tsallis = data['tsallis'].tolist()

    # concatenate the divergences
    features = np.column_stack((mse, kl, reny, tsallis))  # shape (num_samples, 4)
    return features

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # importing the data
    data1 = pd.read_csv("datasets/df_jsut_multiband_melgan_q1.csv")
    data2 = pd.read_csv("datasets/df_jsut_multiband_melgan_q2.csv")
    data3 = pd.read_csv("datasets/df_jsut_multiband_melgan_q3.csv")
    data4 = pd.read_csv("datasets/df_jsut_multiband_melgan_q4.csv")

    labels = labels = data1['labels'].tolist()

    # concatenate the features
    features1 = concatenate_features(data1)
    features2 = concatenate_features(data2)
    features3 = concatenate_features(data3)
    features4 = concatenate_features(data4)

    features = np.column_stack((features1, features2, features3, features4)) # shape (10,000 , 16)

    # Shuffle = false since the data has been already shuffled in the .csv file
    X_train, X_test, y_train, y_test = train_test_split(features, labels, shuffle=False)

    clf = RandomForestClassifier()

    # Train the model using the training sets
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))

    # Predict the classes on the test data, and return the probabilities for each class
    y_proba = clf.predict_proba(X_test)

    # Plot ROC curve as in: https://sachinkalsi.github.io/blog/category/ml/2018/08/20/top-8-performance-metrics-one-should-know.html#receiver-operating-characteristic-curve-roc
    # For each of the probability scores(y_proba) if y_proba >= threshold, then predicted label would be positive(1).

    fpr, tpr, th = roc_curve(y_test, y_proba[:, 1], pos_label=1)  # label = 1 = 'Generated' is the positive class

    # AUC is the area under the ROC curve. More the area under the curve, the more good is the model
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()




