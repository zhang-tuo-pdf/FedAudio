import torchaudio, torch
import s3prl.hub as hub
import os
import sys


def load_model(feature_type):
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print("GPU available, use GPU")
    model = getattr(hub, feature_type)().to(device)
    return device, model


def pretrained_feature(audio_file_path, feature_type, device, model):

    # read audio
    audio, sample_rate = torchaudio.load(audio_file_path)
    transform_model = torchaudio.transforms.Resample(sample_rate, 16000)
    audio = transform_model(audio).to(device)

    if (
        feature_type == "distilhubert"
        or feature_type == "wav2vec2"
        or feature_type == "vq_wav2vec"
        or feature_type == "cpc"
    ):
        features = model([audio[0]])["last_hidden_state"].detach().cpu().numpy()
    else:
        features = model(audio)["last_hidden_state"].detach().cpu().numpy()
    return features[0]


if __name__ == "__main__":
    if sys.platform == "darwin":
        os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    audio_file_path = "/home/ultraz/Project/FedSpeech22/data/speech_commands/audio/bed/0a7c2a8d_nohash_0.wav"
    feature_type = "apc"
    device, model = load_model(feature_type)
    features = pretrained_feature(audio_file_path, feature_type, device, model)
    print(features)
