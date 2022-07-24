# IEMOCAP Loaders
```
python3 iemocap_loader.py --raw_data_path iemocap_data_path \
                          --output_data_path output_data_path \
                          --process_method pretrain/raw/opensmile \
                          --feature_type apc/tera/emobase \
                          --test_session Session1/Session2/Session3/Session4/Session5

```
Output data format:
[key, wav_iemocap, emotion_id, data]


# MSP-Improv Loaders
```
python3 msp_improv_loader.py --raw_data_path DATA_PATH \
                          --output_data_path OUTPUT_PATH \
                          --process_method pretrain/raw/opensmile \
                          --feature_type apc/tera/emobase \
                          --test_session session1/session2/session3/session4/session5/session6

```
Output data format:
[key, wav_msp_improv, emotion_id, data]

# Crema-D Loaders
```
python3 crema_d_loader.py --raw_data_path DATA_PATH \
                          --output_data_path OUTPUT_PATH \
                          --process_method pretrain/raw/opensmile \
                          --feature_type apc/tera/emobase \
                          --test_fold 1/2/3/4/5

```
Output data format:
[key, wav_crema_d, emotion_id, data]
