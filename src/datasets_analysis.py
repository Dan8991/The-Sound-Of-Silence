import os

from scipy.io import wavfile
import scipy.signal as sps


def check_sampling_rate(dataset_path, sr_dataset):
    """
    This function checks that all the audio files of the same dataset have the same sampling rate.

    Parameters:
    dataset_path (string): Path of the dataset
    sr_dataset (int): Sampling rate of the dataset

    Returns:
        sr_is_correct (boolean): is True if all the audio files have sr_dataset as sampling rate
    """
    sr_is_correct = True

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        for f in filenames:
            # get audio path
            audio_path = os.path.join(dirpath, f).replace("\\", "/")

            # read the .wav file
            sr, signal = wavfile.read(audio_path)

            # check the sampling rate
            if sr != sr_dataset:
                sr_is_correct = False

    return sr_is_correct


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # LJSPEECH dataset (english) original
    sr, signal = wavfile.read("datasets/LJSpeech-1.1/wavs/LJ001-0001.wav") # sr = 22050 Hz
    print(check_sampling_rate("datasets/LJSpeech-1.1/wavs/", sr)) # True

    # generated
    sr, signal = wavfile.read("datasets/generated_audio/ljspeech_multi_band_melgan/LJ001-0001_gen.wav") # sr = 22050 Hz
    print(check_sampling_rate("datasets/generated_audio/ljspeech_multi_band_melgan/", sr))  # True


    # JSUT dataset (japanese) original
    sr, signal = wavfile.read("datasets/jsut_ver1.1/basic5000/wav/BASIC5000_0001.wav")  # sr = 48000 Hz
    print(check_sampling_rate("datasets/jsut_ver1.1/basic5000/wav/", sr))  # True

    # generated
    sr, signal = wavfile.read("datasets/generated_audio/jsut_multi_band_melgan/BASIC5000_0001_gen.wav")  # sr = 24000 Hz
    print(check_sampling_rate("datasets/generated_audio/jsut_multi_band_melgan/", sr))  # True


    # Example of downsampling: from 48kHz to 24kHz
    sr, signal = wavfile.read("datasets/jsut_ver1.1/basic5000/wav/BASIC5000_0001.wav")
    sr_min = 24000

    duration = len(signal) / sr  # duration of the signal in seconds
    number_of_samples = duration * sr_min  # number of samples to downsample
    signal = sps.resample(signal, number_of_samples) # now signal has sr = 24000 Hz










