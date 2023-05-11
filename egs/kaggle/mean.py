# calculate the averaged mean and std for wav file in the folder
import os
import numpy as np
import librosa
import json
import math


folder = "../../../data/NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV/train"

def get_mean_std(folder):
    mean = 0
    std = 0
    count = 0
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            y, sr = librosa.load(folder + "/" + file, sr=44100)
            mean += np.mean(y)
            std += np.std(y)
            count += 1
    mean = mean / count
    std = std / count
    return mean, std

if __name__ == "__main__":
    mean, std = get_mean_std(folder)
    print("mean: ", mean)
    print("std: ", std)