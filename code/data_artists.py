import json
import os
import math
import librosa

DATASET_PATH = "artist10"
JSON_PATH = "data_13.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 30  # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
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

    # loop through all genre sub-folder
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if i is 0:
            artists = dirnames
        artist, album = os.path.split(dirpath)
        # ensure we're processing a genre sub-folder level
        if dirpath is not dataset_path and album not in artists:
            # print(i, dirpath, "artist", artist, "album", album)

            # save genre label (i.e., sub-folder name) in the mapping
            unus_dir, semantic_label = os.path.split(artist)
            if semantic_label not in data["mapping"]:
                data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))
            num_album = artists.index(semantic_label)
            print("num_artist:", num_album + 1)

            # pieces_dict = {}
            # process all audio files in genre sub-dir
            for f in filenames:
                title, extension = os.path.splitext(f)
                if extension is ".mp3" or ".wav" or ".wave" or ".flac" or ".wma":
                    print("EXT", extension)
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

                        # store only mfcc feature with expected number of vectors
                        if len(mfcc) == num_mfcc_vectors_per_segment:
                            data["mfcc"].append(mfcc.tolist())
                            data["labels"].append(num_album)
                            print("{}, segment:{}".format(file_path, d + 1))

    # save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


def retrieve_classes(dataset_path):

    num_songs = {}
    classes = []
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if i is 0:
            artists = dirnames
        artist, album = os.path.split(dirpath)
        # ensure we're processing a genre sub-folder level
        if dirpath is not dataset_path and album not in artists:
            # print(i, dirpath, "artist", artist, "album", album)
            # save genre label (i.e., sub-folder name) in the mapping
            unus_dir, semantic_label = os.path.split(artist)
            if semantic_label not in classes:
                classes.append(semantic_label)
                num_songs[semantic_label] = 1
            else:
                for f in filenames:
                    num_songs[semantic_label] += 1

    print(classes)
    print(num_songs)


if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)
    # retrieve_classes(DATASET_PATH)