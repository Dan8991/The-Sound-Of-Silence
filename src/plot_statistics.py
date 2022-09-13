import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.io import wavfile
from python_speech_features import mfcc
import soundfile as sf

from scipy.optimize import curve_fit
from scipy.stats import entropy


def mfcc_feature_extraction(audio_path, audio_format, q):
    """
    This function extracts mfcc features from an audio file.

    Parameters:
        audio_path (string): Path of the audio file
        audio_format (string): Can be either '.wav' or '.flac'
        q (int): Quantization step

    Returns:
        mfccs (numpy array): Array of size (numframes, numcep) containing MFCC features.
    """

    if audio_format == '.wav':
        # read .wav file
        sr, signal = wavfile.read(audio_path)

    elif audio_format == '.flac':
        # read .flac file
        signal, sr = sf.read(audio_path)

    else:
        print("Invalid value encountered in 'audio_format'")

    # trim all silence that is longer than 0.1 s
    silence_part = 0
    for i in range(len(signal)):
        if signal[i]==0:
            silence_part = silence_part + 1

    if silence_part/sr > 0.1:
        signal = signal[signal != 0]

    # divide the signal into frames of 1024 samples, with an overlap of 512 samples (~50%)
    winlen = 1024 / sr  # convert into seconds
    winstep = winlen / 2

    # number of coefficients to return
    numcep = 14

    # number of filters in the filterbank
    nfilt = 26

    # FFT size
    nfft = 1024

    # Sampling rate used to compute the higher frequency that respects the Shannon sampling theorem
    highfreq = sr / 2

    # get mfcc coefficients of shape (numframes, numcep)
    mfccs = mfcc(signal, samplerate=sr, winlen=winlen, winstep=winstep, nfft=nfft, numcep=numcep, nfilt=nfilt, highfreq=highfreq)

    # quantization
    mfccs = (mfccs / q)

    return mfccs


def first_digit_gen(d, base):
    """
        This function computes the first digit vector.

        Parameters:
            d (float or numpy array): number on which the FD is calculated
            base (int)

        Returns:
            the first digit vector of d
        """

    return np.floor(np.abs(d) / base ** np.floor(np.log(np.abs(d)) / np.log(base)))


def compute_histograms(audio, base, n_freq):
    """
    This function returns the pmf of the first digit vector.

    Parameters:
        audio (numpy array): first digit vector
        base (int)
        n_freq (int): length of fd

    Returns:
        histogram of fd.
    """

    h_audio = []
    for k in range(n_freq):
        try:
            h, _ = np.histogram(audio[:, k], range=(np.nanmin(audio[:, k]), np.nanmax(audio[:, k])),
                                bins=np.arange(0.5, base + 0.5, 1), density=True)
            # range of the histogram: (1, 9); bins: [0.5 1.5 2.5 3.5 4.5 5.5 6.5 7.5 8.5 9.5]

        except ValueError:
            h = np.zeros(base - 1, dtype=np.float64)

        h_audio += [h]

    return np.asarray(h_audio)


def first_digit_call(audio_path, audio_format, n_freq, q, base):
    """
        This function computes the MFCCs and the corresponding first digit vector and histogram of each .wav file stored in a given directory.

        Parameters:
            audio_path (string): Path of the audio file
            audio_format (string): Can be either '.wav' or '.flac'
            q (int): quantization step
            base (int)

        Returns:
            ff_list, name_list (lists): list of pmfs and corresponding file names of all the audio of the dataset

    """

    # extract mfcc features from the audio
    audio_mfccs = mfcc_feature_extraction(audio_path, audio_format, q)

    # remove DC (zero frequency component)
    audio_mfccs = audio_mfccs[:, 1:] # (numframes, 13)

    # actually compute first digit vector
    fd = first_digit_gen(audio_mfccs, base) # (numframes, 13)

    # computing histograms
    ff = compute_histograms(fd, base, n_freq)  # matrix with shape (frequencies, probabilities) = (13, base - 1)

    return ff


def renyi_div(pk, qk, alpha):
    r = np.log2(np.nansum((pk ** alpha) * (qk ** (1 - alpha)))) / (alpha - 1)
    return r


def tsallis_div(pk, qk, alpha):
    r = (np.nansum((pk ** alpha) * (qk ** (1 - alpha))) - 1) / (alpha - 1)
    return r


def gen_benford(m, k, a, b):
    base = len(m)
    return k * (np.log10(1 + (1 / (a + m ** b))) / np.log10(base))


def exponential(x, p, a, b):
    p_x = a*x + b*x**2
    return p + np.exp(-p_x)


def feature_extraction(ff):
    """
    This function fits the pmf of an audio with a given function. The fitness is measured by some divergence functions.

    Parameters:
        ff (numpy array): pmf of an audio

    Returns:
        mse (int) Mean Square Error
        popt[:, 0], popt[:, 1], popt[:, 2] (int): fitting parameters of the Benford's law
        kl (int) Kullback_Leibler divergence
        renyi (int): Renyi divergence
        tsallis (int): Tsallis divergence
    """

    base = len(ff) + 1

    mse_img = []
    popt_img = []
    kl_img = []
    renyi_img = []
    tsallis_img = []

    ff_zeroes_idx = ff == 0
    try:
        # Compute regular features
        popt_k, _ = curve_fit(gen_benford, np.arange(1, base, 1), ff, bounds=(0, np.inf),
                              maxfev=1000)  # popt_k = (k, a, b)
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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # ASV SPOOF 2019 dataset
    generated_path = "datasets/raw/ASVspoof-LA/ASVspoof2019_LA_eval/flac/LA_E_1080447.flac"
    natural_path = "datasets/raw/ASVspoof-LA/ASVspoof2019_LA_train/flac/LA_T_1271820.flac"

    base = 10
    q = 1
    n_freq = 13
    digits = np.arange(1, base)

    ff_gen_tot = first_digit_call(generated_path, '.flac', n_freq, q, base)
    ff_nat_tot = first_digit_call(natural_path, '.flac', n_freq, q, base)

    n = 10 # frequency number
    ff_gen = ff_gen_tot[n, :]
    ff_nat = ff_nat_tot[n, :]

    mse_gen, popt_0_gen, popt_1_gen, popt_2_gen, kl_gen, reny_gen, tsallis_gen = feature_extraction(ff_gen)
    mse_nat, popt_0_nat, popt_1_nat, popt_2_nat, kl_nat, reny_nat, tsallis_nat = feature_extraction(ff_nat)

    fitted_function = gen_benford(np.arange(1, base, 1), popt_0_nat, popt_1_nat, popt_2_nat)

    plt.figure(figsize=(12,5))
    plt.plot(digits, ff_nat)
    plt.plot(digits, ff_gen)
    plt.plot(digits, fitted_function, '--', color='tab:olive')
    plt.legend(["Bonafide", "Fake", "Benford"])
    plt.savefig("b10_f10_q1.png")
    plt.xlabel("First Digit")
    plt.ylabel("p(d)")
    plt.show()

    for n in range(1, n_freq):
        plt.subplot(3, 5, n)

        ff_gen = ff_gen_tot[n, :]
        ff_nat = ff_nat_tot[n, :]

        mse_gen, popt_0_gen, popt_1_gen, popt_2_gen, kl_gen, reny_gen, tsallis_gen = feature_extraction(ff_gen)
        mse_nat, popt_0_nat, popt_1_nat, popt_2_nat, kl_nat, reny_nat, tsallis_nat = feature_extraction(ff_nat)

        fitted_function = gen_benford(np.arange(1, base, 1), popt_0_nat, popt_1_nat, popt_2_nat)


        plt.plot(digits, ff_nat)
        plt.plot(digits, ff_gen)
        plt.plot(digits, fitted_function, '--', color='tab:olive')
        plt.legend(["Bonafide", "Fake", "Benford"])

        plt.title("Frequency number:{}".format(n))
    #plt.savefig("b10_q1_tot.png")
    plt.show()








