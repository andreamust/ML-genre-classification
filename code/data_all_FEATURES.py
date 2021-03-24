import json
import os
import math
import librosa
import numpy as np

DATASET_PATH = "genres"
JSON_PATH = "data_chroma-10.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 29.7  # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


def save_mfcc(dataset_path, json_path, num_mfcc=40, n_fft=512, hop_length=256, num_segments=5):
    """Extracts MFCCs from music dataset and saves them into a json file along witgh genre labels.
        :param dataset_path (str): Path to dataset
        :param json_path (str): Path to json file used to save MFCCs
        :param num_mfcc (int): Number of coefficients to extract
        :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
        :param hop_length (int): Sliding window for FFT. Measured in # of samples
        :param: num_segments (int): Number of segments we want to divide sample tracks into
        :return:
        """

    # dictionary to store mapping, labels, and MFCCs
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)
    print(samples_per_segment)

    # loop through all genre sub-folder
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're processing a genre sub-folder level
        if dirpath is not dataset_path:

            # save genre label (i.e., sub-folder name) in the mapping
            semantic_label = dirpath.split("/")[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))

            # process all audio files in genre sub-dir
            for f in filenames:

                # load audio file
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                # process all segments of audio file
                for d in range(num_segments):
                    # calculate start and finish sample for current segment
                    start = samples_per_segment * d
                    finish = start + samples_per_segment

                    # extract mfcc
                    mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                                hop_length=hop_length)
                    mfcc = mfcc.T

                    spectral_center = librosa.feature.spectral_centroid(
                        signal[start:finish], sample_rate, hop_length=hop_length
                    )

                    spectral_center = spectral_center.T

                    chroma = librosa.feature.chroma_stft(signal[start:finish], sample_rate, n_chroma=num_mfcc,
                                                         hop_length=hop_length)

                    chroma = chroma.T

                    spectral_contrast = librosa.feature.spectral_contrast(
                        signal[start:finish], sample_rate, hop_length=hop_length
                    )

                    spectral_contrast = spectral_contrast.T

                    # store only mfcc feature with expected number of vectors
                    if len(chroma) == num_mfcc_vectors_per_segment:
                        print(len(mfcc))
                        data["mfcc"].append(chroma.tolist())


                        # arr = np.zeros((len(mfcc), 33))
                        #
                        # arr[:, 0:13] = mfcc
                        # arr[:, 13:14] = spectral_center
                        # arr[:, 14:26] = chroma
                        # arr[:, 26:33] = spectral_contrast
                        # data["mfcc"].append(arr.tolist())


                        data["labels"].append(i - 1)
                        print("{}, segment:{}".format(file_path, d + 1))

    # save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


def retrieve_labels(dataset_path):
    genre_list = []
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're processing a genre sub-folder level
        if dirpath is not dataset_path:
            # save genre label (i.e., sub-folder name) in the mapping
            directory, genre = os.path.split(dirpath)
            genre_list.append(genre)
    print(genre_list)


if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)
    # retrieve_labels(DATASET_PATH)
