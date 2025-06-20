# Test for Clinical annotation
python concept_annotation/automatic_concept_annotation.py \
    --model_api open_clip_hf-hub:xieji-x/MAKE \
    --data_dir "data/skincon" \
    --batch_size 32 \
    --concept_list "data/skincon/concept_list.txt" \
    --concept_terms_json "concept_annotation/term_lists/ConceptTerms.json"

# Test for Dermascopic annotation
python concept_annotation/automatic_concept_annotation.py \
    --model_api open_clip_hf-hub:xieji-x/MAKE \
    --data_dir "data/derm7pt" \
    --batch_size 32 \
    --concept_list "data/derm7pt/concept_list.txt" \
    --concept_terms_json "concept_annotation/term_lists/ConceptTerms.json"

# Clinical concept annotation
python concept_annotation/infer.py \
    --image_path data/skincon/final_images/000004.png \
    --model_api open_clip_hf-hub:xieji-x/MAKE \
    --concept_list "data/skincon/concept_list.txt" \
    --concept_terms_json "concept_annotation/term_lists/ConceptTerms.json"

# Dermascopic concept annotation
python concept_annotation/infer.py \
    --image_path data/derm7pt/final_images/Aal002.jpg \
    --model_api open_clip_hf-hub:xieji-x/MAKE \
    --concept_list data/derm7pt/concept_list.txt \
    --concept_terms_json concept_annotation/term_lists/ConceptTerms.json
