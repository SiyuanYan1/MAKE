## Creating a Custom Training Dataset

MAKE uses a structured CSV file to load training data. Each row corresponds to one image and its associated multi-level textual annotations. To create your own dataset, follow the format described below.

The public Derm1M dataset corresponds to a cleaner Version, while MAKE was originally trained on the original version. Therefore, minor preprocessing is required to ensure compatibility.

### Required Columns

Your CSV file must contain the following columns:

```
filename,
caption,
truncated_caption,
visual_concept_caption,
ontology_caption,
sub_captions,
subcaption_1,
subcaption_2,
subcaption_3,
subcaption_4,
subcaption_5,
subcaption_6,
subcaption_7,
subcaption_8,
sub_caption_mask,
knowledge_masks
```

### Column Description

* **filename**
  Path to the image file (relative or absolute). The file must be accessible during training.

* **caption**
  Full natural language description of the image.

* **truncated_caption**
  Usually identical to `caption`. Can be used for text truncation experiments.

* **visual_concept_caption**
  Description of visible features only (e.g., erythema, scaling, vesicles).
  Do not include diagnostic labels.

* **ontology_caption**
  Structured diagnostic description using a consistent template.
  Example:
  `This is a skin photo diagnosed as inflammatory, non-infectious, atopic dermatitis.`

* **sub_captions**
  Number of semantic sub-sentences (1–8).

* **subcaption_1 – subcaption_8**
  Caption split into up to 8 semantic units.
  Leave unused fields empty.

* **sub_caption_mask**
  Binary list of length 8 indicating which subcaptions are valid.
  Example:
  `[1, 1, 0, 0, 0, 0, 0, 0]`

* **knowledge_masks**
  Binary list of length 3 controlling supervision levels:
  `[caption_level, visual_level, ontology_level]`

  Example:
  `[1, 1, 1]` means all three supervision signals are used.

---

### Minimal Example (Single Subcaption)

```
images/img_001.png,
Chronic atopic dermatitis in the antecubital fossa.,
Chronic atopic dermatitis in the antecubital fossa.,
This skin photo shows erythema and scaling.,
This is a skin photo diagnosed as inflammatory, non-infectious, atopic dermatitis.,
1,
Chronic atopic dermatitis in the antecubital fossa,
,,,,,,,
[1,0,0,0,0,0,0,0],
[1,1,1]
```

As long as the column order is consistent and masks have correct lengths (8 for `sub_caption_mask`, 3 for `knowledge_masks`), the dataset will be compatible with MAKE training.
