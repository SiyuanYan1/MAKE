# Test for Clinical annotation
python concept_annotation/experiments/concept-annotation/automatic_concept_annotation.py \
    --model_api open_clip_hf-hub:xieji-x/MAKE \
    --data_dir "concept_annotation/data/concept-annotation/skincon" \
    --batch_size 32 \
    --concept_list "concept_annotation/data/concept-annotation/skincon/concept_list.txt" \
    --concept_terms_json "concept_annotation/term_lists/ConceptTerms.json"

# Test for Dermascopic annotation
python concept_annotation/experiments/concept-annotation/automatic_concept_annotation.py \
    --model_api open_clip_hf-hub:xieji-x/MAKE \
    --data_dir "concept_annotation/data/concept-annotation/derm7pt" \
    --batch_size 32 \
    --concept_list "concept_annotation/data/concept-annotation/derm7pt/concept_list.txt" \
    --concept_terms_json "concept_annotation/term_lists/ConceptTerms.json"

# Clinical concept annotation
python concept_annotation/experiments/concept-annotation/infer.py \
    --image_path concept_annotation/data/concept-annotation/skincon/final_images/000004.png \
    --model_api open_clip_hf-hub:xieji-x/MAKE \
    --concept_list "concept_annotation/data/concept-annotation/skincon/concept_list.txt" \
    --concept_terms_json "concept_annotation/term_lists/ConceptTerms.json"

# Dermascopic concept annotation
python concept_annotation/experiments/concept-annotation/infer.py \
    --image_path concept_annotation/data/concept-annotation/derm7pt/final_images/Aal002.jpg \
    --model_api open_clip_hf-hub:xieji-x/MAKE \
    --concept_list concept_annotation/data/concept-annotation/derm7pt/concept_list.txt \
    --concept_terms_json concept_annotation/term_lists/ConceptTerms.json
