import opensmile
import numpy as np


def opensmile_feature(audio_file_path, feature_type):
    if feature_type == "emobase":
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.emobase,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
    elif feature_type == "ComParE":
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
    else:
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
    return np.array(smile.process_file(audio_file_path))


if __name__ == "__main__":
    audio_file_path = "/home/ultraz/Project/FedSpeech22/data/speech_commands/audio/bed/0a7c2a8d_nohash_0.wav"
    feature_type = "emobase"
    features = opensmile_feature(audio_file_path, feature_type)
    print(features)
