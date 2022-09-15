import pandas as pd
import os

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # take the train, development, evaluation lists and convert them into .csv files
    train_splitting_path = os.path.join(
        "datasets",
        "raw",
        "ASVspoof-LA",
        "ASVspoof2019_LA_cm_protocols",
        "ASVspoof2019.LA.cm.train.trn.txt"
    )

    data = pd.read_csv(train_splitting_path, sep=" ", header=None)
    data.columns = ["Speaker ID", "Audio file name", "-", "System ID", "Label"]
    data = data.drop(['-'], axis=1)
    data.to_csv(os.path.join(
        "datasets",
        "processed",
        "ASVspoof-LA",
        "asv_training_set.csv"
    ))

    dev_splitting_path = "datasets/raw/ASVspoof-LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"

    data = pd.read_csv(dev_splitting_path, sep=" ", header=None)
    data.columns = ["Speaker ID", "Audio file name", "-", "System ID", "Label"]
    data = data.drop(['-'], axis=1)
    data.to_csv(os.path.join(
        "datasets",
        "processed",
        "ASVspoof-LA",
        "asv_development_set.csv"
    ))

    eval_splitting_path = "datasets/raw/ASVspoof-LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"

    data = pd.read_csv(eval_splitting_path, sep=" ", header=None)
    data.columns = ["Speaker ID", "Audio file name", "-", "System ID", "Label"]
    data = data.drop(['-'], axis=1)
    data.to_csv(os.path.join(
        "datasets",
        "processed",
        "ASVspoof-LA",
        "asv_evaluation_set.csv"
    ))






