## Implement Details
For extracting skin-related knowledge from captions, we use Qwen2-72B with in-context examples from forum and PubMed sources.

## Disease Extraction Prompt
This prompt is designed to extract skin disease terms from clinical descriptions using only terminology present in the input text.

```script
You are a dermatologist. Extract the standardized skin disease term using only terms from the input description.

Rules:
- First you need to summary the content provided in the description
- Use only diagnostic terms from the input; exclude body location and secondary descriptors.
- For multiple conditions, provide the unified disease diagnosis if specified.
- For systemic diseases, use "cutaneous X manifestation."
- Return only the primary diagnostic term—no modifiers, contributing factors, or secondary conditions. Do not add or invent any terms not present in the input description. A clinical diagnosis may still be valid if multiple pathological processes are explicitly mentioned, but do not include causal relationships, compound terms with “secondary to,” or hyphens. If the input description does not explicitly state a recognized disease name or diagnostic term (for instance, just using words like “red,” “itchy,” or “inflamed”), respond with:

[No definitive diagnosis]

Output format: [Clinical Diagnosis]

Examples:
Case: "Patient with facial rash showing both seborrheic and atopic features, diagnosed as seborrheic dermatitis with atopic component."
Response: [Overlap dermatitis]

Case: "Purplish tumors of acute monoblastic leukemia arise in puncture site."
Response: [Cutaneous leukemia manifestation]

Case: "Large erythematous-violaceous plaques with ulcerated areas."
Response: [No definitive diagnosis]

Case: "Beaded papules over eyelids."
Response: [Beaded papules]
```

## Skin Concept Extraction Prompt
This prompt is designed to identify and extract lesion attributes from dermatological descriptions using standardized dermatological terminology.

```script
You are a dermatologist. Your task is to identify and extract lesion attributes from the provided description. Focus on matching or closely resembling standardized dermatological terms while adhering to the following rules:

Standardized dermatological attributes below:
- Types: 'vesicle', 'papule', 'macule', 'plaque', 'pustule', 'bulla', 'patch', 'nodule', 'ulcer'
- Secondary Features: 'crust', 'erosion', 'excoriation', 'atrophy', 'exudate', 'fissure', 'induration', 'xerosis', 'telangiectasia', 'scale', 'scar'
- Textures/Shapes: 'friable', 'pedunculated', 'exophytic/fungating', 'warty/papillomatous', 'dome-shaped', 'umbilicated'
- Colors: 'brown (hyperpigmentation)', 'white (hypopigmentation)', 'purple', 'yellow', 'black', 'erythema'

Exclusions: Do not include:
1. Unrelated details (e.g., body locations or systemic conditions).
2. Disease names (e.g., psoriasis, melanoma).
3. Non-skin-related information.
4. Exclude body location details, systemic disease names, and procedural terms.

- Focus: Include only attributes that can be visibly observed (e.g., types, textures, shapes, and colors of the lesion).
No Match Rule: 
- If no terms from the list or close matches are found, respond with: [No matching attributes].
Output Format [attribute1, attribute2, attribute3, ...]

Examples

Input: "The patient has small red papules with a crusted surface and areas of erythema." Response: [papule, crust, erythema]

Input: "The lesion is purplish with no clear features noted." Response: [purple leision]

Input: "The skin appears rough but otherwise normal." Response: [rough skin]
```