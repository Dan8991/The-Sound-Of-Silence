import pandas as pd
import numpy as np

def natural_vs_generated_audio(data):

    # converting column data to list
    name = data['name'].tolist()

    tot_audio = len(name)
    correct = 0
    for i in range(tot_audio):
        if data['label'][i] == 0:  # take natural audio

            # create the name of the corresponding generated audio
            n = name[i]  # BASIC5000_0940.wav
            n = n[:-4]  # BASIC5000_0940
            n = n + '_gen.wav'  # BASIC5000_0940_gen.wav

            # extract the corresponding rows of the df
            natural = data.loc[data['name'] == name[i]]  # natural
            generated = data.loc[data['name'] == n]  # generated

            # extract corresponding values
            natural = natural.iloc[:, :-1]
            natural = natural.iloc[:, 1:]
            natural = np.array(natural)

            generated = generated.iloc[:, :-1]
            generated = generated.iloc[:, 1:]
            generated = np.array(generated)

            is_natural = 0
            x, y = np.shape(natural)
            for j in range(0,y):
                if natural[0,j] < generated[0,j]:
                    is_natural = is_natural + 1

            if is_natural >= (y/2):  # if at least half of the divergences of the natural audio is lower than one of the corresponding generated audio
                correct = correct + 1

    tot_natural = tot_audio/2

    return correct / tot_natural


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # importing the data
    data = pd.read_csv("datasets/df_inter_jsut.csv")

    # computing the accuracies
    accuracy = natural_vs_generated_audio(data)

    print("-------------Comparisons between natural and generated audio of the same sentence-------------")
    print("Accuracy :", accuracy)

    # importing the data
    data = pd.read_csv("datasets/df_inter_ljspeech.csv")

    # computing the accuracies
    accuracy = natural_vs_generated_audio(data)

    print("-------------Comparisons between natural and generated audio of the same sentence-------------")
    print("Accuracy :", accuracy)

































