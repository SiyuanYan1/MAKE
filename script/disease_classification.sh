# MAKE Zero-Shot
python src/test.py \
    --val-data=""  \
    --dataset-type "csv" \
    --batch-size 2048 \
    --zeroshot-eval1=data/PAD/MAKE_PAD.csv \
    --zeroshot-eval2=data/F17K/MAKE_F17K.csv \
    --zeroshot-eval3=data/SNU/MAKE_SNU.csv \
    --zeroshot-eval4=data/SD-128/MAKE_SD-128.csv \
    --csv-label-key label \
    --csv-img-key filename \
    --model 'hf-hub:xieji-x/MAKE'

# Cross-Modality Retreival
python src/main.py \
    --val-data="data/skin_cap/skin_cap_meta.csv"  \
    --dataset-type "csv" \
    --batch-size=2048 \
    --csv-img-key filename \
    --csv-caption-key 'caption_zh_polish_en' \
    --model 'hf-hub:xieji-x/MAKE'

# Diagnosis Zero Shot: Infer one image at once time (Classes from F17K.)
python src/infer.py \
    --image_path 'data/F17K/images/9ca1e7ab5fa2261ecd1a938b635a228f.jpg' \
    --k 5 \
    --model 'hf-hub:xieji-x/MAKE'