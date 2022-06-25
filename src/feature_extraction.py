import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.io import wavfile
from python_speech_features import mfcc
import soundfile as sf
from multiprocessing import Pool

from scipy.optimize import curve_fit
from scipy.stats import entropy
from torch.utils.data import Dataset, DataLoader
from librosa.effects import trim
import torch
import argparse

def parse_args():

    parser = argparse.ArgumentParser(description="Preprocess the ASVSpoof dataset")
    parser.add_argument("--type", default="full", help="type of processed signal can be:\n"
            "full: the whole signal is processed\n"
            "sound: processes only parts with high enough energy\n"
            "silence: processes only parts with low enough energy\n"
    )
    return parser.parse_args()


class AudioDataset(Dataset):

    def __init__(self, dataset_path, audio_format, n_freq, base, signal_type):

        super().__init__()
        self.audio_format = audio_format
        self.n_freq = n_freq
        self.base = base
        self.signal_type = signal_type

        self.files = []

        for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
            for f in filenames:

                audio_path = os.path.join(dirpath, f).replace("\\", "/") # for example 'datasets/raw/ASVspoof-LA/LA_D_1000752.flac'
                self.files.append(audio_path)
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        audio_path = self.files[idx]
        divergences_list = []

        ffs, len_signals = first_digit_call(
            audio_path=audio_path,
            audio_format=self.audio_format,
            n_freq=self.n_freq,
            base=self.base,
            signal_type=self.signal_type
        ) # (13, base - 1)

        # for each frequency i = {0, 1, ..., 13}
        for ff in ffs:
            for i in range(0, n_freq):

                # ff_temp_sound = ff_sound[i, :] # take one frequency at time (1, base - 1)
                ff_temp = ff[i, :] # take one frequency at time (1, base - 1)

                mse, popt_0, popt_1, popt_2, kl, reny, tsallis = feature_extraction(ff_temp)
                divergences_list += [float(mse), float(kl), float(reny), float(tsallis)] 

        name = os.path.basename(audio_path)

        return name, np.array(divergences_list), np.array(len_signals)



def create_df(dataset_path, audio_format, splitting_path, n_freq, base, signal_type="full"):
    """
    This function creates a Pandas DataFrame in which each row corresponds to an audio sample. In particular, each row
    contains the four divergences computed for each audio's frequency (4*13 = 52) computed with different quantization steps (52*4 = 208 columns).

    Parameters:
        dataset_path (string): Path of the dataset
        audio_format (string): Can be either '.wav' or '.flac'
        splitting_path (string): Path of the file that contains the train-development-evaluation split.
        n_freq (int): number of frequencies
        base (list)
    """

    df = pd.DataFrame()

    name_list = []
    dataset = AudioDataset(dataset_path, audio_format, n_freq, base, signal_type)
    names = []
    samples = []
    signal_lengths = []

    with Pool(os.cpu_count(), maxtasksperchild=1) as p:
        names, samples, signal_lengths = list(tqdm(p.imap(dataset.__getitem__, np.arange(len(dataset))), total=len(dataset)))
    # for name, data, signal_len in tqdm(dataloader):
        # samples.append(data)
        # signal_lengths += list(signal_len.numpy())
        # names += name


    df = pd.DataFrame(torch.cat(samples, axis=0).numpy())
    df.insert(loc=0, column='name', value=names)
    df["Audio file name"] = df.apply(lambda x: x["name"][:-5], axis=1)
    df["length"] = signal_lengths

    # assign the corresponding label
    data = pd.read_csv(splitting_path)
    df = pd.merge(df, data, on=["Audio file name"]).drop(columns="Audio file name")
    df["label"] = df.apply(map_labels, axis=1)
    df = df.drop(columns=["Label", "Speaker ID", "Unnamed: 0"])
    df.rename(columns={"System ID": "system ID"}, inplace=True)

    # df_list = (df[df.isna().any(axis=1)]['name'])
    # df = df[df.name.isin(df_list) == False]

    # sorted dataframe by 'name' column
    df = df.sort_values(by=['name'], ascending=True)

    return df


def split_silence(signal):
    power = signal ** 2
    filt = torch.ones((1, 1, 101))
    data = torch.tensor(power).unsqueeze(0).unsqueeze(0).float()
    window_power = torch.nn.functional.conv1d(data, filt, stride = 1, padding=50) + 1e-10
    window_power = window_power / 101
    window_power_db = -10 * np.log10(window_power)
    return window_power_db


def mfcc_feature_extraction(audio_path, audio_format, signal_type="full"):
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
    if signal_type != "full":
        window_power = split_silence(signal)
    if signal_type == "silence":
        silence = signal[torch.where(window_power[0, 0] > 40)]
        signal = silence[np.where(silence != 0)]
    elif signal_type == "sound":
        signal = signal[torch.where(window_power[0, 0] <= 40)]

    signal_lengths = len(signal)
    # this can only happen for silence signals since they are usually shorter
    # the noise is just a placeholder since it will be removed form training/testing
    if signal_lengths < 5000:
        signal = np.random.rand(10000)

    slices = []
    last = 0

    # divide the signal into frames of 1024 samples, with an overlap of 512 samples (~50%)
    winlen = 1024 / sr  # convert into seconds
    winstep = winlen / (8 if signal_type == "silence" else 2)

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

    return mfccs, signal_lengths


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


def first_digit_call(audio_path, audio_format, n_freq, base, signal_type):
    """
        This function computes the MFCCs and the corresponding first digit vector and histogram of each .wav file stored in a given directory.

        Parameters:
            audio_path (string): Path of the audio file
            audio_format (string): Can be either '.wav' or '.flac'
            n_freq (int): Frequency number
            q (int): quantization step
            base (list)

        Returns:
            ff_list, name_list (lists): list of pmfs and corresponding file names of all the audio of the dataset

    """

    # extract mfcc features from the audio
    mfccs, len_signals = mfcc_feature_extraction(audio_path, audio_format, signal_type=signal_type)

    # remove DC (zero frequency component)
    mfccs = mfccs[:, 1:] # (numframes, 13)

    # actually compute first digit vector
    ffs = []
    for q in range(1, 5):
        for b in base:
            fd = first_digit_gen(mfccs / q, b) # (numframes, 13)

            # computing histograms
            ff = compute_histograms(fd, b, n_freq)  # matrix with shape (frequencies, probabilities) = (13, base - 1)
            ffs.append(ff)

    return ffs, len_signals


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
    p_x = a * x + b * x ** 2
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

def map_labels(x):
    if x["Label"] == "bonafide":
        return 0
    if x["Label"] == "spoof":
        return 1
    return np.nan


def concatenate_df(df_base10, df_base20):

    df_base20 = df_base20.iloc[:, 1:]  # remove 'name' column from base 20
    df_base10 = df_base10.iloc[:, :-2]  # remove 'label' and 'system id' column from base 10

    df = pd.concat([df_base10, df_base20], axis=1)  # concatenate by columns, now (numsamples, 419) i.e., 208*2 divergences + 'name', 'label', system id' columns

    return df

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    args = parse_args()
    signal_type = args.type
    assert signal_type in ["full", "sound", "silence"]
    train_path = 'datasets/raw/ASVspoof-LA/ASVspoof2019_LA_train/flac/'
    dev_path = 'datasets/raw/ASVspoof-LA/ASVspoof2019_LA_dev/flac/'
    eval_path = 'datasets/raw/ASVspoof-LA/ASVspoof2019_LA_eval/flac/'

    train_splitting_path = "datasets/processed/ASVspoof-LA/asv_training_set.csv"
    dev_splitting_path = "datasets/processed/ASVspoof-LA/asv_development_set.csv"
    eval_splitting_path = "datasets/processed/ASVspoof-LA/asv_evaluation_set.csv"

    #required since sometimes there are errors with torch
    # pool = torch.multiprocessing.Pool(torch.multiprocessing.cpu_count(), maxtasksperchild=1)
    # torch.multiprocessing.set_sharing_strategy('file_system')

    n_freq = 13

    print("Train")
    # Training
    df_train = create_df(dataset_path=train_path, audio_format='.flac', splitting_path=train_splitting_path,n_freq=n_freq, base=[10, 20], signal_type=signal_type)

    pd.DataFrame(df_train).to_csv(f"datasets/processed/ASVspoof-LA/df_train_{signal_type}.csv", index=False)

    print("Dev")
    # Development
    df_dev = create_df(dataset_path=dev_path, audio_format='.flac', splitting_path=dev_splitting_path, n_freq=n_freq, base=[10, 20], signal_type=signal_type)

    pd.DataFrame(df_dev).to_csv(f"datasets/processed/ASVspoof-LA/df_dev_{signal_type}.csv", index=False)

    print("Eval")
    # Evaluation
    df_eval = create_df(dataset_path=eval_path, audio_format='.flac', splitting_path=eval_splitting_path,n_freq=n_freq, base=[10, 20], signal_type=signal_type)

    pd.DataFrame(df_eval).to_csv(f"datasets/processed/ASVspoof-LA/df_eval_{signal_type}.csv", index=False)
