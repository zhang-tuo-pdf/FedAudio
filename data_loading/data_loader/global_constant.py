""" Feature dim dictionary """
audio_feat_dim_dict = {
    "pretrain": {
        "apc": 512,
        "tera": 768,
    },
    "opensmile": {"emobase": 968},
    "raw": {"mel_spec": 128},
}

""" Label dim dictionary """
label_dim_dict = {
    "iemocap": 4,
    "gcommand": 36,
    "cream-d": 4,
    "meld": 3,
    "urban-sound": 10,
}
