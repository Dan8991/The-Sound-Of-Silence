from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def fit_multiple_estimators(classifiers, X_list, y):

    # Convert the labels `y` using LabelEncoder, because the predict method is using index-based pointers
    # which will be converted back to original data later.
    le_ = LabelEncoder()
    le_.fit(y)

    # Fit all estimators with their respective feature arrays
    estimators_ = [clf.fit(X, y) for clf, X in zip([clf for _, clf in classifiers], X_list)]

    return estimators_, le_


def predict_from_multiple_estimator(estimators, label_encoder, X_list):

    # Predict 'soft' voting with probabilities

    pred1 = np.asarray([clf.predict_proba(X) for clf, X in zip(estimators, X_list)]) # shape = (n_classifiers, n_test_samples, prob_for_each_class) = (3, 2500, 2)

    # compute the average across the three classifiers
    pred2 = np.average(pred1, axis=0)  # shape = (n_test_samples, n_classes) = (2500, 2)

    # return the index of the max between the two probabilities
    pred = np.argmax(pred2, axis=1)  # shape = (n_test_samples, ) = (2500,)

    # Convert integer predictions to original labels:
    return pred2, label_encoder.inverse_transform(pred)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # create four different classifiers, one for each divergence
    rf_mse =RandomForestClassifier()
    rf_kl = RandomForestClassifier()
    rf_reny = RandomForestClassifier()
    rf_tsallis = RandomForestClassifier()

    # importing the data
    data = pd.read_csv("datasets/df_jsut_multiband_melgan.csv")

    # converting column data to list
    mse = data['mse'].tolist()
    kl = data['kl'].tolist()
    reny = data['reny'].tolist()
    tsallis = data['tsallis'].tolist()
    labels = data['labels'].tolist()

    # Shuffle = false since the data has been already shuffled in the .csv file (the labels are always the same)
    X_train1, X_test1, y_train, y_test = train_test_split(kl, labels, shuffle=False) # kl
    X_train2, X_test2, y_train, y_test = train_test_split(reny, labels, shuffle=False)  # reny
    X_train3, X_test3, y_train, y_test = train_test_split(tsallis, labels, shuffle=False)  # tsallis
    X_train4, X_test4, y_train, y_test = train_test_split(mse, labels, shuffle=False)  # mse


    X_train1 = np.array(X_train1).reshape(-1, len(X_train1)).T # shape (7500, 1)
    X_train2 = np.array(X_train2).reshape(-1, len(X_train2)).T
    X_train3 = np.array(X_train3).reshape(-1, len(X_train3)).T
    X_train4 = np.array(X_train4).reshape(-1, len(X_train4)).T

    X_test1 = np.array(X_test1).reshape(-1, len(X_test1)).T # shape (2500, 1)
    X_test2 = np.array(X_test2).reshape(-1, len(X_test2)).T
    X_test3 = np.array(X_test3).reshape(-1, len(X_test3)).T
    X_test4 = np.array(X_test4).reshape(-1, len(X_test4)).T

    X_train_list = [X_train1, X_train2, X_train3, X_train4]  # shape (4, 7500, 1)
    X_test_list = [X_test1, X_test2, X_test3, X_test4]  # shape (4, 2500, 1)

    classifiers = [('rf_kl', rf_kl), ('rf_reny', rf_reny), ('rf_tsallis', rf_tsallis),('rf_mse', rf_mse)]

    fitted_estimators, label_encoder = fit_multiple_estimators(classifiers, X_train_list, y_train)

    prob_pred, y_pred = predict_from_multiple_estimator(fitted_estimators, label_encoder, X_test_list)
    
    print("Accuracy:", accuracy_score(y_test, y_pred))

    fpr, tpr, th = roc_curve(y_test, prob_pred[:,0], pos_label=0) # label = 0 = 'Natural' is the positive class
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




