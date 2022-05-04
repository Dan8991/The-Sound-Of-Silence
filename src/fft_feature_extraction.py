from scipy.io import wavfile
from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from scipy.stats import entropy
import pandas as pd

def fft_feature_extraction(audio_path):
    """
    This function extracts mfcc features.
    """

    # read .wav file
    sr, signal = wavfile.read(audio_path)

    fft_spectrum = np.fft.rfft(signal)  # take the right data
    fft_spectrum_abs = np.abs(fft_spectrum)  # take the abs - shape (76561,)

    return fft_spectrum_abs


def first_digit_gen(d, base):
    """
    Compute first digit vector.
    """

    return np.floor(np.abs(d) / base ** np.floor(np.log(np.abs(d)) / np.log(base))) # np.log is the natural logarithm


def compute_histograms(audio, base, n_freq):
    """
    Return the pmf of the first digit vector.
    """

    h_audio = []
    for k in range(n_freq):
        try:
            h, _ = np.histogram(audio[k], range=(np.nanmin(audio[k]), np.nanmax(audio[k])),
                                bins=np.arange(0.5, base + 0.5, 1), density=True)
            # range of the histogram: (1, 9); bins: [0.5 1.5 2.5 3.5 4.5 5.5 6.5 7.5 8.5 9.5]

        except ValueError:
            h = np.zeros(base - 1, dtype=np.float64)

        h_audio += [h]

    return np.asarray(h_audio)


def first_digit_call(dataset_path, base):

    ff_list = []
    name_list = []

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        for f in filenames:

            name_list.append(f)

            audio_path = os.path.join(dirpath, f).replace("\\", "/")

            # extract mfcc features from the audio
            audio_fft = fft_feature_extraction(audio_path)

            # remove DC (zero frequency component)
            audio_fft = audio_fft[1:]

            # actually compute first digit vector
            fd = first_digit_gen(audio_fft, base)

            # computing histograms
            nfreq = np.shape(audio_fft)[0]
            ff = compute_histograms(fd, base, nfreq) # matrix with shape (frequencies, probabilities) = (1023, 9)

            # take the mean across the frequencies (the rows)
            ff = ff.mean(0)

            ff_list.append(ff)
            print(f)

    return ff_list, name_list


def gen_benford(m, k, a, b):
    base = len(m)
    return k * (np.log10(1 + (1 / (a + m ** b))) / np.log10(base))


def plot_statistics(natural, generated, base):

    digits = np.arange(1, base)

    benford = gen_benford(digits, 1, 0, 1) # Benford's law

    fig, ax = plt.subplots(figsize=(12, 6))

    plt.plot(digits, natural)
    plt.plot(digits, generated)
    plt.plot(digits, benford, '--')
    plt.legend(["Natural", "GAN", "Benford"])
    plt.xlabel("First Digit")
    plt.ylabel("p(d)")
    plt.show()


def renyi_div(pk, qk, alpha):
    r = np.log2(np.nansum((pk ** alpha) * (qk ** (1 - alpha)))) / (alpha - 1)
    return r


def tsallis_div(pk, qk, alpha):
    r = (np.nansum((pk ** alpha) * (qk ** (1 - alpha))) - 1) / (alpha - 1)
    return r


def feature_extraction(ff):
    base = len(ff) + 1

    mse_img = []
    popt_img = []
    kl_img = []
    renyi_img = []
    tsallis_img = []

    ff_zeroes_idx = ff == 0
    try:
        # Compute regular features
        popt_k, _ = curve_fit(gen_benford, np.arange(1, base, 1), ff,  bounds=(0, np.inf)) # popt_k = (k, a, b)
        h_fit = gen_benford(np.arange(1, base, 1), *popt_k)

        h_fit_zeroes_idx = h_fit == 0

        zeroes_idx = np.logical_or(ff_zeroes_idx, h_fit_zeroes_idx)

        ff_no_zeroes = ff[~zeroes_idx]
        h_fit_no_zeroes = h_fit[~zeroes_idx]

        popt_img += [popt_k]
        mse_img += [np.mean((ff - h_fit) ** 2)]

        kl_img += [entropy(pk=ff_no_zeroes, qk=h_fit_no_zeroes, base=2) +
                   entropy(pk=h_fit_no_zeroes, qk=ff_no_zeroes, base=2)]
        renyi_img += [renyi_div(pk=ff_no_zeroes, qk=h_fit_no_zeroes, alpha=0.3) +
                      renyi_div(pk=h_fit_no_zeroes, qk=ff_no_zeroes, alpha=0.3)]
        tsallis_img += [tsallis_div(pk=ff_no_zeroes, qk=h_fit_no_zeroes, alpha=0.3) +
                        tsallis_div(pk=h_fit_no_zeroes, qk=ff_no_zeroes, alpha=0.3)]

    except (RuntimeError, ValueError):
        mse_img += [np.nan]
        popt_img += [(np.nan, np.nan, np.nan)]
        kl_img += [np.nan]
        renyi_img += [np.nan]
        tsallis_img += [np.nan]

    mse = np.asarray(mse_img)
    popt = np.asarray(popt_img)
    kl = np.asarray(kl_img)
    renyi = np.asarray(renyi_img)
    tsallis = np.asarray(tsallis_img)

    return mse, popt[:, 0], popt[:, 1], popt[:, 2], kl, renyi, tsallis


def create_df(ff_natural, ff_generated, names_natural, names_generated):

    data = {
        "name":[],
        "ff": [],
        "mse": [],
        "kl": [],
        "reny": [],
        "tsallis": [],
        "labels": []
    }

    for i in range(len(ff_natural)):
        data["name"].append(names_natural[i])
        data["ff"].append(ff_natural[i])
        mse, popt_0, popt_1, popt_2, kl, reny, tsallis = feature_extraction(ff_natural[i])
        data["mse"].append(float(mse))
        data["kl"].append(float(kl))
        data["reny"].append(float(reny))
        data["tsallis"].append(float(tsallis))
        data["labels"].append(0)  # label = 0 for natural audio

    for i in range(len(ff_generated)):
        data["name"].append(names_generated[i])
        data["ff"].append(ff_generated[i])
        mse, popt_0, popt_1, popt_2, kl, reny, tsallis = feature_extraction(ff_generated[i])
        data["mse"].append(float(mse))
        data["kl"].append(float(kl))
        data["reny"].append(float(reny))
        data["tsallis"].append(float(tsallis))
        data["labels"].append(1)  # label = 1 for generated audio

    df = pd.DataFrame.from_dict(data)

    # shuffle DataFrame rows
    df = df.sample(frac=1)

    return df

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    base = 10

    # NATURAL AUDIO
    ff_natural, filenames_natural = first_digit_call("datasets/jsut_ver1.1/basic5000/wav", base)
    print(1)
    # GAN-GENERATED AUDIO
    ff_generated, filenames_generated = first_digit_call("datasets/generated_audio/jsut_multi_band_melgan", base)
    print(2)
    df = create_df(ff_natural, ff_generated, filenames_natural, filenames_generated)
    print(3)
    df.to_csv("df_fft.csv")






















