import os
import numpy as np
import pandas as pd

import soundfile as sf
import scipy.signal as sps
from scipy.io import wavfile
from python_speech_features import mfcc

from scipy.optimize import curve_fit
from scipy.stats import entropy


def mfcc_feature_extraction(audio_path, audio_format, resample, sr_new, q):
    """
        This function extracts mfcc features from an audio file.

        Parameters:
            audio_path (string): Path of the audio file
            audio_format (string): Can be either '.wav' or '.flac'
            resample (boolean): if True, changes the sampling rate to sr_new
            sr_new (int): New sampling rate
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

    # change the sampling rate
    if resample == True:
        duration = len(signal) / sr  # duration of the signal in seconds
        number_of_samples = int(duration * sr_new)  # number of samples with the new sampling rate
        signal = sps.resample(signal, number_of_samples) # now signal has sr = sr_new
        sr = sr_new

    # divide the signal into frames of 1024 samples, with an overlap of 512 samples (~50%)
    winlen = 1024/sr  # convert into seconds
    winstep = winlen/2

    # number of coefficients to return
    numcep = 13

    # number of filters in the filterbank
    nfilt = 26

    # FFT size
    nfft = 1024

    # Sampling rate used to compute the higher frequency that respects the Shannon sampling theorem
    highfreq = sr / 2

    # get mfcc coefficients of shape (numframes, numcep), where numframes = ( (number_of_samples - frame_samples)/step_samples ) + 1
    mfccs = mfcc(signal, samplerate=sr, winlen=winlen, winstep=winstep, nfft=nfft,numcep=numcep, nfilt=nfilt, highfreq=highfreq)

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

    return np.floor(np.abs(d) / base ** np.floor(np.log(np.abs(d)) / np.log(base))) # np.log is the natural logarithm


def compute_histograms(fd, base, n_freq):
    """
    This function returns the pmf of the first digit vector.

    Parameters:
        fd (numpy array): first digit vector
        base (int)
        n_freq (int): length of fd

    Returns:
        histogram of fd.
    """

    h_audio = []
    for k in range(n_freq):
        try:
            h, _ = np.histogram(fd[:, k], range=(np.nanmin(fd[:, k]), np.nanmax(fd[:, k])), bins=np.arange(0.5, base + 0.5, 1), density=True)
            # range of the histogram: (1, 9); bins: [0.5 1.5 2.5 3.5 4.5 5.5 6.5 7.5 8.5 9.5]

        except ValueError:
            h = np.zeros(base - 1, dtype=np.float64)

        h_audio += [h]

    return np.asarray(h_audio)


def first_digit_call(dataset_path, audio_format, resample, sr_new, q, base):
    """
        This function computes the MFCCs and the corresponding first digit vector and histogram of each .wav file stored in a given directory.

        Parameters:
            dataset_path (string): Path of the dataset
            audio_format (string): Can be either '.wav' or '.flac'
            resample (boolean): if True, changes the sampling rate to sr_new
            sr_new (int): New sampling rate
            q (int): quantization step
            base (int)

        Returns:
            ff_list, name_list (lists): list of pmfs and corresponding file names of all the audio of the dataset

    """
    ff_list = []
    name_list = []

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        for f in filenames:
            name_list.append(f)

            audio_path = os.path.join(dirpath, f).replace("\\", "/")

            # extract mfcc features from the audio
            audio_mfccs = mfcc_feature_extraction(audio_path, audio_format, resample, sr_new, q)

            # vectorize
            x, y = np.shape(audio_mfccs)
            audio_mfccs = audio_mfccs.reshape(-1, x*y)

            # remove DC (zero frequency component)
            audio_mfccs = audio_mfccs[:, 1:]

            # actually compute first digit vector
            fd = first_digit_gen(audio_mfccs, base)

            # computing histograms
            n_freq = np.shape(fd)[1]
            ff = compute_histograms(fd, base, n_freq) # matrix with shape (frequencies, probabilities)

            # take the mean across the frequencies (the rows)
            ff = ff.mean(0)

            ff_list.append(ff)

    return ff_list, name_list


def gen_benford(m, k, a, b):
    base = len(m)
    return k * (np.log10(1 + (1 / (a + m ** b))) / np.log10(base))


def exponential(x, p, a, b):
    p_x = a*x + b*x**2
    return p + np.exp(-p_x)


def renyi_div(pk, qk, alpha):
    r = np.log2(np.nansum((pk ** alpha) * (qk ** (1 - alpha)))) / (alpha - 1)
    return r


def tsallis_div(pk, qk, alpha):
    r = (np.nansum((pk ** alpha) * (qk ** (1 - alpha))) - 1) / (alpha - 1)
    return r


def feature_extraction(ff, function):
    """
    This function fits the pmf of an audio with a given function. The fitness is measured by some divergence functions.

    Parameters:
    ff (numpy array): pmf of an audio
    function (function): function for the fitting

    Returns:
    mse (int) Mean Square Error
    popt[:, 0], popt[:, 1], popt[:, 2] (int): fitting parameters of the function
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
        popt_k, _ = curve_fit(gen_benford, np.arange(1, base, 1), ff,  bounds=(0, np.inf)) # popt_k = (k, a, b)
        h_fit = function(np.arange(1, base, 1), *popt_k)

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


def create_df(ff_natural, ff_generated, names_natural, names_generated, function):

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
        mse, popt_0, popt_1, popt_2, kl, reny, tsallis = feature_extraction(ff_natural[i], function)
        data["mse"].append(float(mse))
        data["kl"].append(float(kl))
        data["reny"].append(float(reny))
        data["tsallis"].append(float(tsallis))
        data["labels"].append(0)  # label = 0 for natural audio

    for i in range(len(ff_generated)):
        data["name"].append(names_generated[i])
        data["ff"].append(ff_generated[i])
        mse, popt_0, popt_1, popt_2, kl, reny, tsallis = feature_extraction(ff_generated[i], function)
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

    # JSUT dataset (japanese)
    jsut_natural_path = "datasets/jsut/basic5000/wav/"
    jsut_generated_path = "datasets/generated_audio/jsut_multi_band_melgan/"

    # LJSPEECH dataset (english)
    ljspeech_natural_path = "datasets/LJSpeech/wavs"
    ljspeech_generated_path = "datasets/generated_audio/ljspeech_multi_band_melgan"

    base = 10
    n_freq = 13
    function = gen_benford

    # Test intra:
    resample = False
    sr_jsut = 24000
    sr_ljspeech = 22050

    for q in range(1,5):

        # Original dataset
        ff_natural, filenames_natural = first_digit_call(dataset_path=jsut_natural_path, audio_format='.wav', resample=resample, sr_new=sr_jsut, q=q, base=base)

        # Generated dataset
        ff_generated, filenames_generated = first_digit_call(dataset_path=jsut_generated_path, audio_format='.wav', resample=resample, sr_new=sr_jsut, q=q, base=base)

        # fitting with the Benford's law
        df = create_df(ff_natural, ff_generated, filenames_natural, filenames_generated, function)
        df.to_csv("datasets/df_intra_jsut_mean_q{}.csv".format(q))

        # Original dataset
        ff_natural, filenames_natural = first_digit_call(dataset_path=ljspeech_natural_path, audio_format='.wav', resample=resample, sr_new=sr_ljspeech, q=q, base=base)

        # Generated dataset
        ff_generated, filenames_generated = first_digit_call(dataset_path=ljspeech_generated_path, audio_format='.wav', resample=resample, sr_new=sr_ljspeech, q=q, base=base)

        # fitting with the Benford's law
        df = create_df(ff_natural, ff_generated, filenames_natural, filenames_generated, function)
        df.to_csv("datasets/df_intra_ljspeech_mean_q{}.csv".format(q))

    # Test inter:
    resample = True
    sr_jsut = 16000
    sr_ljspeech = 16000

    for q in range(1,5):

        # Original dataset
        ff_natural, filenames_natural = first_digit_call(dataset_path=jsut_natural_path, audio_format='.wav', resample=resample, sr_new=sr_jsut, q=q, base=base)

        # Generated dataset
        ff_generated, filenames_generated = first_digit_call(dataset_path=jsut_generated_path, audio_format='.wav', resample=resample, sr_new=sr_jsut, q=q, base=base)

        # fitting with the Benford's law
        df = create_df(ff_natural, ff_generated, filenames_natural, filenames_generated, function)
        df.to_csv("datasets/df_inter_jsut_mean_q{}.csv".format(q))

        # Original dataset
        ff_natural, filenames_natural = first_digit_call(dataset_path=ljspeech_natural_path, audio_format='.wav', resample=resample, sr_new=sr_ljspeech, q=q, base=base)

        # Generated dataset
        ff_generated, filenames_generated = first_digit_call(dataset_path=ljspeech_generated_path, audio_format='.wav', resample=resample, sr_new=sr_ljspeech, q=q, base=base)

        # fitting with the Benford's law
        df = create_df(ff_natural, ff_generated, filenames_natural, filenames_generated, function)
        df.to_csv("datasets/df_inter_ljspeech_mean_q{}.csv".format(q))





































