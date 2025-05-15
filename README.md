# [MICCAI 2025] MAKE: Multi-Aspect Knowledge-Enhanced Vision-Language Pretraining for Zero-shot Dermatological Assessment 

## Abstract

Dermatological diagnosis represents a complex multimodal challenge that requires integrating visual features with specialized clinical knowledge. We introduce MAKE, a Multi-Aspect Knowledge-Enhanced vision-language pretraining framework for zero-shot dermatological tasks. Our approach addresses the limitations of existing vision-language models by decomposing clinical narratives into knowledge-enhanced subcaptions, connecting subcaptions with relevant image features, and adaptively prioritizing different knowledge aspects. Through pretraining on 403,563 dermatological image-text pairs, MAKE significantly outperforms state-of-the-art VLP models on eight datasets across zero-shot skin disease classification, concept annotation, and cross-modal retrieval tasks.

## Resources

- 📦 **Code and pretrained model** will be released in this repository
- 🔍 **Pretraining data** will be available at [Derm1M Repository](https://github.com/SiyuanYan1/Derm1M)


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
