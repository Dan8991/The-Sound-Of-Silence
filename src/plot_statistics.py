from scipy.io import wavfile
from python_speech_features import mfcc
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import entropy


def mfcc_feature_extraction(audio_path, sr_min, q):
    """
    This function extracts mfcc features from a .wav file.
    """

    # read .wav file
    sr, signal = wavfile.read(audio_path)

    # divide the signal into frames of 1024 samples, with an overlap of 512 samples (~50%)
    winlen = 1024/sr # convert into seconds
    winstep = winlen/2

    # number of coefficients to return
    numcep = 13

    # number of filters in the filterbank
    nfilt = 1024

    # FFT size
    nfft = 1024

    # get mfcc coefficients of shape (numframes, numcep), where numframes = ( (number_of_samples - frame_samples)/step_samples ) + 1
    mfccs = mfcc(signal, samplerate=sr, winlen=winlen, winstep=winstep, nfft=nfft,numcep=numcep, nfilt=nfilt, highfreq=sr_min/2)

    # quantization
    mfccs = (mfccs / q)

    return mfccs


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
            h, _ = np.histogram(audio[:, k], range=(np.nanmin(audio[:, k]), np.nanmax(audio[:, k])),
                                bins=np.arange(0.5, base + 0.5, 1), density=True)
            # range of the histogram: (1, 9); bins: [0.5 1.5 2.5 3.5 4.5 5.5 6.5 7.5 8.5 9.5]

        except ValueError:
            h = np.zeros(base - 1, dtype=np.float64)

        h_audio += [h]

    return np.asarray(h_audio)


def first_digit_call(audio_path, sr_min, q, base):

    # extract mfcc features from the audio
    audio_mfccs = mfcc_feature_extraction(audio_path, sr_min, q)

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
        popt_k, _ = curve_fit(function, np.arange(1, base, 1), ff,  bounds=(0, np.inf)) # popt_k = (k, a, b)
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

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    base = 10

    sr = 24000 # for jsut dataset
    #sr = 22050 # for ljspeech dataset

    q = 1

    digits = np.arange(1, base)

    function = exponential

    print("------------------------------------first_audio------------------------------------")
    ff_nat1 = first_digit_call("datasets/jsut_ver1.1/basic5000/wav/BASIC5000_0001.wav", sr, q, base)
    ff_gen1 = first_digit_call("datasets/generated_audio/jsut_multi_band_melgan/BASIC5000_0001_gen.wav", sr, q, base)
    #ff_nat1 = first_digit_call("datasets/LJSpeech-1.1/wavs/LJ001-0001.wav", sr, q, base)
    #ff_gen1 = first_digit_call("datasets/generated_audio/ljspeech_multi_band_melgan/LJ001-0001_gen.wav", sr, q, base)


    mse_nat, popt_0_nat, popt_1_nat, popt_2_nat, kl_nat, reny_nat, tsallis_nat = feature_extraction(ff_nat1, function)
    mse_gen, popt_0_gen, popt_1_gen, popt_2_gen, kl_gen, reny_gen, tsallis_gen = feature_extraction(ff_gen1, function)

    function_nat1 = function(np.arange(1, base, 1), popt_0_nat, popt_1_nat, popt_2_nat)
    function_gen1 = function(np.arange(1, base, 1), popt_0_gen, popt_1_gen, popt_2_gen)

    print("NATURAL - mse, kl, reny, tsallis", mse_nat, kl_nat, reny_nat, tsallis_nat)
    print("GENERATED - mse, kl, reny, tsallis", mse_gen, kl_gen, reny_gen, tsallis_gen)

    print("------------------------------------second_audio------------------------------------")
    ff_nat2 = first_digit_call("datasets/jsut_ver1.1/basic5000/wav/BASIC5000_0002.wav", sr, q, base)
    ff_gen2 = first_digit_call("datasets/generated_audio/jsut_multi_band_melgan/BASIC5000_0002_gen.wav", sr, q, base)
    #ff_nat2 = first_digit_call("datasets/LJSpeech-1.1/wavs/LJ001-0002.wav", sr, q, base)
    #ff_gen2 = first_digit_call("datasets/generated_audio/ljspeech_multi_band_melgan/LJ001-0002_gen.wav", sr, q, base)

    mse_nat, popt_0_nat, popt_1_nat, popt_2_nat, kl_nat, reny_nat, tsallis_nat = feature_extraction(ff_nat2,function)
    mse_gen, popt_0_gen, popt_1_gen, popt_2_gen, kl_gen, reny_gen, tsallis_gen = feature_extraction(ff_gen2,function)

    function_nat2 = function(np.arange(1, base, 1), popt_0_nat, popt_1_nat, popt_2_nat)
    function_gen2 = function(np.arange(1, base, 1), popt_0_gen, popt_1_gen, popt_2_gen)

    print("NATURAL - mse, kl, reny, tsallis", mse_nat, kl_nat, reny_nat, tsallis_nat)
    print("GENERATED - mse, kl, reny, tsallis", mse_gen, kl_gen, reny_gen, tsallis_gen)

    print("------------------------------------third_audio------------------------------------")
    ff_nat3 = first_digit_call("datasets/jsut_ver1.1/basic5000/wav/BASIC5000_0003.wav", sr, q, base)
    ff_gen3 = first_digit_call("datasets/generated_audio/jsut_multi_band_melgan/BASIC5000_0003_gen.wav", sr, q, base)
    #ff_nat3 = first_digit_call("datasets/LJSpeech-1.1/wavs/LJ001-0003.wav", sr, q, base)
    #ff_gen3 = first_digit_call("datasets/generated_audio/ljspeech_multi_band_melgan/LJ001-0003_gen.wav", sr, q, base)

    mse_nat, popt_0_nat, popt_1_nat, popt_2_nat, kl_nat, reny_nat, tsallis_nat = feature_extraction(ff_nat3, function)
    mse_gen, popt_0_gen, popt_1_gen, popt_2_gen, kl_gen, reny_gen, tsallis_gen = feature_extraction(ff_gen3, function)

    function_nat3 = function(np.arange(1, base, 1), popt_0_nat, popt_1_nat, popt_2_nat)
    function_gen3 = function(np.arange(1, base, 1), popt_0_gen, popt_1_gen, popt_2_gen)

    print("NATURAL - mse, kl, reny, tsallis", mse_nat, kl_nat, reny_nat, tsallis_nat)
    print("GENERATED - mse, kl, reny, tsallis", mse_gen, kl_gen, reny_gen, tsallis_gen)

    print("------------------------------------fourth_audio------------------------------------")
    ff_nat4 = first_digit_call("datasets/jsut_ver1.1/basic5000/wav/BASIC5000_0004.wav", sr, q, base)
    ff_gen4 = first_digit_call("datasets/generated_audio/jsut_multi_band_melgan/BASIC5000_0004_gen.wav", sr, q, base)
    #ff_nat4 = first_digit_call("datasets/LJSpeech-1.1/wavs/LJ001-0004.wav", sr, q, base)
    #ff_gen4 = first_digit_call("datasets/generated_audio/ljspeech_multi_band_melgan/LJ001-0004_gen.wav", sr, q, base)

    mse_nat, popt_0_nat, popt_1_nat, popt_2_nat, kl_nat, reny_nat, tsallis_nat = feature_extraction(ff_nat4, function)
    mse_gen, popt_0_gen, popt_1_gen, popt_2_gen, kl_gen, reny_gen, tsallis_gen = feature_extraction(ff_gen4, function)

    function_nat4 = function(np.arange(1, base, 1), popt_0_nat, popt_1_nat, popt_2_nat)
    function_gen4 = function(np.arange(1, base, 1), popt_0_gen, popt_1_gen, popt_2_gen)


    print("NATURAL - mse, kl, reny, tsallis", mse_nat, kl_nat, reny_nat, tsallis_nat)
    print("GENERATED - mse, kl, reny, tsallis", mse_gen, kl_gen, reny_gen, tsallis_gen)


    fig = plt.figure()

    plt.subplot(2, 2, 1)
    plt.plot(digits, ff_nat1)
    plt.plot(digits, ff_gen1)
    plt.plot(digits, function_nat1, '--')
    plt.plot(digits, function_gen1, '--')
    plt.legend(["Natural", "GAN", "fit on natural", "fit on generated"])
    plt.title("First audio")
    plt.xlabel("First Digit")
    plt.ylabel("p(d)")

    plt.subplot(2, 2, 2)
    plt.plot(digits, ff_nat2)
    plt.plot(digits, ff_gen2)
    plt.plot(digits, function_nat2, '--')
    plt.plot(digits, function_gen2, '--')
    plt.legend(["Natural", "GAN", "fit on natural", "fit on generated"])
    plt.title("Second audio")
    plt.xlabel("First Digit")
    plt.ylabel("p(d)")

    plt.subplot(2, 2, 3)
    plt.plot(digits, ff_nat3)
    plt.plot(digits, ff_gen3)
    plt.plot(digits, function_nat3, '--')
    plt.plot(digits, function_gen3, '--')
    plt.legend(["Natural", "GAN", "fit on natural", "fit on generated"])
    plt.title("Third audio")
    plt.xlabel("First Digit")
    plt.ylabel("p(d)")

    plt.subplot(2, 2, 4)
    plt.plot(digits, ff_nat4)
    plt.plot(digits, ff_gen4)
    plt.plot(digits, function_nat4, '--')
    plt.plot(digits, function_gen4, '--')
    plt.legend(["Natural", "GAN", "fit on natural", "fit on generated"])
    plt.title("Fourth audio")
    plt.xlabel("First Digit")
    plt.ylabel("p(d)")
    plt.show()

















