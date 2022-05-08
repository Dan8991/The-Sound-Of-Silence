import pandas as pd

def natural_vs_generated_audio(data):

    # converting column data to list
    name = data['name'].tolist()

    tot_audio = len(name)
    correct = 0
    for i in range(tot_audio):
        if data['labels'][i] == 0:  # take natural audio

            # create the name of the corresponding generated audio
            n = name[i]  # BASIC5000_0940.wav
            n = n[:-4]  # BASIC5000_0940
            n = n + '_gen.wav'  # BASIC5000_0940_gen.wav

            # extract the corresponding rows of the df
            natural = data.loc[data['name'] == name[i]]  # natural
            generated = data.loc[data['name'] == n]  # generated

            # extract corresponding mse, kl, reny and tsallis
            mse_natural = float(natural['mse'])
            kl_natural = float(natural['kl'])
            reny_natural = float(natural['reny'])
            tsallis_natural = float(natural['tsallis'])

            mse_generated = float(generated['mse'])
            kl_generated = float(generated['kl'])
            reny_generated = float(generated['reny'])
            tsallis_generated = float(generated['tsallis'])

            is_natural = 0 # counts the number of times that a divergence of the natural audio is lower than the one of the generated audio
            if mse_natural < mse_generated:
                is_natural = is_natural + 1

            if kl_natural < kl_generated:
                is_natural = is_natural + 1

            if reny_natural < reny_generated:
                is_natural = is_natural + 1

            if tsallis_natural < tsallis_generated:
                is_natural = is_natural + 1

            if is_natural >= 2:  # if at least two divergences of the natural audio is lower than one of the corresponding generated audio
                correct = correct + 1

    tot_natural = tot_audio/2
    return correct / tot_natural


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # importing the data
    data1 = pd.read_csv("datasets/df_jsut_multiband_melgan_q1.csv")
    data2 = pd.read_csv("datasets/df_jsut_multiband_melgan_q2.csv")
    data3 = pd.read_csv("datasets/df_jsut_multiband_melgan_q3.csv")
    data4 = pd.read_csv("datasets/df_jsut_multiband_melgan_q4.csv")

    # computing the accuracies
    accuracy1 = natural_vs_generated_audio(data1)
    accuracy2 = natural_vs_generated_audio(data2)
    accuracy3 = natural_vs_generated_audio(data3)
    accuracy4 = natural_vs_generated_audio(data4)

    print("-------------Comparisons between natural and generated audio of the same sentence-------------")
    print("q = 1:", accuracy1)
    print("q = 2:", accuracy2)
    print("q = 3:", accuracy3)
    print("q = 4:", accuracy4)
































