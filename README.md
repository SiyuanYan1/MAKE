# MAKE: Multi-Aspect Knowledge-Enhanced Vision-Language Pretraining for Zero-shot Dermatological Assessment

![GitHub stars](https://img.shields.io/github/stars/SiyuanYan1/MAKE?style=social)
![GitHub forks](https://img.shields.io/github/forks/SiyuanYan1/MAKE?style=social)
![License](https://img.shields.io/badge/License-MIT-blue)

## Abstract

Dermatological diagnosis represents a complex multimodal challenge that requires integrating visual features with specialized clinical knowledge. While vision-language pretraining (VLP) has advanced medical AI, its effectiveness in dermatology is limited by text length constraints and the lack of structured texts.

In this paper, we introduce **MAKE**, a **M**ulti-**A**spect **K**nowledge-**E**nhanced vision-language pretraining framework for zero-shot dermatological tasks. Recognizing that comprehensive dermatological descriptions require multiple knowledge aspects that exceed standard text constraints, our framework introduces:

1. A **multi-aspect contrastive learning strategy** that decomposes clinical narratives into knowledge-enhanced sub-texts through large language models
2. A **fine-grained alignment mechanism** that connects subcaptions with diagnostically relevant image features
3. A **diagnosis-guided weighting scheme** that adaptively prioritizes different sub-captions based on clinical significance prior

Through pretraining on 403,563 dermatological image-text pairs collected from education resources, MAKE significantly outperforms state-of-the-art VLP models on eight datasets across zero-shot skin disease classification, concept annotation, and cross-modal retrieval tasks.

## Framework Overview

![MAKE Framework](https://raw.githubusercontent.com/SiyuanYan1/MAKE/main/assets/framework.png)

MAKE addresses key limitations of conventional VLP frameworks like CLIP, which typically limit text input to a fixed token length (e.g., 77 tokens) and truncate longer descriptions. Our approach:

- Decomposes complex dermatological descriptions into multiple sub-captions, each capturing distinct aspects of clinical knowledge (morphology, distribution patterns, symptoms)
- Associates multiple sub-captions with diagnostically relevant image patches of skin lesions
- Adaptively prioritizes different aspects of knowledge based on their diagnostic relevance in dermatology practice

This approach enables precise alignment between visual features of skin lesions and various aspects of clinical knowledge, which is critical for differential diagnosis.

## Resources

- 📦 **Code and pretrained model** will be released in this repository
- 🔍 **Pretraining data** available at [Derm1M Repository](https://github.com/SiyuanYan1/Derm1M)
- 🧠 **Vision encoder foundation model** available at [PanDerm Repository](https://github.com/SiyuanYan1/PanDerm)

## Key Innovations

| Feature | Description |
|---------|-------------|
| **Multi-aspect Knowledge-Image Contrastive Learning** | Decomposes clinical narratives into multiple sub-captions, each capturing distinct aspects of clinical knowledge |
| **Fine-grained Alignment Mechanism** | Associates sub-captions with diagnostically relevant image patches of skin lesions |
| **Diagnosis-guided Weighting Scheme** | Adaptively prioritizes different aspects of knowledge based on their diagnostic relevance |

## Performance

MAKE significantly outperforms state-of-the-art VLP models on:

- Zero-shot skin disease classification
- Concept annotation
- Cross-modal retrieval tasks

Evaluated across eight comprehensive dermatological datasets.

## Citation

If you find our work useful in your research, please consider citing our papers:

```bibtex
@misc{yan2025makemultiaspectknowledgeenhancedvisionlanguage,
      title={MAKE: Multi-Aspect Knowledge-Enhanced Vision-Language Pretraining for Zero-shot Dermatological Assessment}, 
      author={Siyuan Yan and Xieji Li and Ming Hu and Yiwen Jiang and Zhen Yu and Zongyuan Ge},
      year={2025},
      eprint={2505.09372},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.09372}, 
}

@misc{yan2025derm1mmillionscalevisionlanguagedataset,
      title={Derm1M: A Million-scale Vision-Language Dataset Aligned with Clinical Ontology Knowledge for Dermatology}, 
      author={Siyuan Yan and Ming Hu and Yiwen Jiang and Xieji Li and Hao Fei and Philipp Tschandl and Harald Kittler and Zongyuan Ge},
      year={2025},
      eprint={2503.14911},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.14911}, 
}

@misc{yan2025multimodalvisionfoundationmodel,
      title={A Multimodal Vision Foundation Model for Clinical Dermatology}, 
      author={Siyuan Yan and Zhen Yu and Clare Primiero and Cristina Vico-Alonso and Zhonghua Wang and Litao Yang and Philipp Tschandl and Ming Hu and Lie Ju and Gin Tan and Vincent Tang and Aik Beng Ng and David Powell and Paul Bonnington and Simon See and Elisabetta Magnaterra and Peter Ferguson and Jennifer Nguyen and Pascale Guitera and Jose Banuls and Monika Janda and Victoria Mar and Harald Kittler and H. Peter Soyer and Zongyuan Ge},
      year={2025},
      eprint={2410.15038},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.15038}, 
}
