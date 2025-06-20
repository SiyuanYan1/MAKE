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
    --model ViT-B-16 \
    --resume 'model_weight/MAKE.pt' 

# OPENAI CLIP-B-16 Zero-Shot
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
    --model ViT-B-16 \
    --pretrained OPENAI

# Cross-Modality Retreival
python src/main.py \
    --val-data="data/skin_cap/skin_cap_meta.csv"  \
    --dataset-type "csv" \
    --batch-size=2048 \
    --csv-img-key filename \
    --csv-caption-key 'caption_zh_polish_en' \
    --model ViT-B-16 \
    --resume 'model_weight/MAKE.pt' 

# Diagnosis Zero Shot: Infer one image at once time (Classes from F17K.)
python src/infer.py \
    --image_path 'data/F17K/images/0a0e21f413499ad85018f7fa0df3efe2.jpg' \
    --model ViT-B-16 \
    --resume 'model_weight/MAKE.pt'