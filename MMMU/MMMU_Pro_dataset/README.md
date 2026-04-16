---
language:
- en
license: apache-2.0
size_categories:
- 10K<n<100K
task_categories:
- question-answering
- visual-question-answering
- multiple-choice
dataset_info:
- config_name: standard (10 options)
  features:
  - name: id
    dtype: string
  - name: question
    dtype: string
  - name: options
    dtype: string
  - name: explanation
    dtype: string
  - name: image_1
    dtype: image
  - name: image_2
    dtype: image
  - name: image_3
    dtype: image
  - name: image_4
    dtype: image
  - name: image_5
    dtype: image
  - name: image_6
    dtype: image
  - name: image_7
    dtype: image
  - name: img_type
    dtype: string
  - name: answer
    dtype: string
  - name: topic_difficulty
    dtype: string
  - name: subject
    dtype: string
  splits:
  - name: test
    num_bytes: 691464721.52
    num_examples: 1730
  download_size: 677992993
  dataset_size: 691464721.52
- config_name: standard (4 options)
  features:
  - name: id
    dtype: string
  - name: question
    dtype: string
  - name: options
    dtype: string
  - name: explanation
    dtype: string
  - name: image_1
    dtype: image
  - name: image_2
    dtype: image
  - name: image_3
    dtype: image
  - name: image_4
    dtype: image
  - name: image_5
    dtype: image
  - name: image_6
    dtype: image
  - name: image_7
    dtype: image
  - name: img_type
    dtype: string
  - name: answer
    dtype: string
  - name: topic_difficulty
    dtype: string
  - name: subject
    dtype: string
  splits:
  - name: test
    num_bytes: 691172846.48
    num_examples: 1730
  download_size: 677854263
  dataset_size: 691172846.48
- config_name: vision
  features:
  - name: id
    dtype: string
  - name: image
    dtype: image
  - name: options
    dtype: string
  - name: answer
    dtype: string
  - name: subject
    dtype: string
  splits:
  - name: test
    num_bytes: 1719633315.3
    num_examples: 1730
  download_size: 1632115576
  dataset_size: 1719633315.3
configs:
- config_name: standard (10 options)
  data_files:
  - split: test
    path: standard (10 options)/test-*
- config_name: standard (4 options)
  data_files:
  - split: test
    path: standard (4 options)/test-*
- config_name: vision
  data_files:
  - split: test
    path: vision/test-*
tags:
- chemistry
- biology
- music
- art
- medical
- math
- science
- engineering
---


# MMMU-Pro (A More Robust Multi-discipline Multimodal Understanding Benchmark)

[**🌐 Homepage**](https://mmmu-benchmark.github.io/) | [**🏆 Leaderboard**](https://mmmu-benchmark.github.io/#leaderboard) | [**🤗 Dataset**](https://huggingface.co/datasets/MMMU/MMMU_Pro) | [**🤗 Paper**](https://huggingface.co/papers/2409.02813) | [**📖 arXiv**](https://arxiv.org/abs/2409.02813) | [**GitHub**](https://github.com/MMMU-Benchmark/MMMU)

## 🔔News

- **🛠️🛠️ [2025-03-08] Fixed mismatch between inner image labels and shuffled options in Vision and Standard (10 options) settings. (test_Chemistry_5,94,147,216,314,345,354,461,560,570; test_Materials_450; test_Pharmacy_198; validation_Chemistry_12,26,29; validation_Materials_10,28; validation_Psychology_1)**
- **🛠️[2024-11-10] Added options to the Vision subset.**
- **🛠️[2024-10-20] Uploaded Standard (4 options) cases.**
- **🔥[2024-09-05] Introducing [MMMU-Pro](https://arxiv.org/abs/2409.02813), a robust version of MMMU benchmark for multimodal AI evaluation! 🚀**

# Introduction

MMMU-Pro is an enhanced multimodal benchmark designed to rigorously assess the true understanding capabilities of advanced AI models across multiple modalities. It builds upon the original MMMU benchmark by introducing several key improvements that make it more challenging and realistic, ensuring that models are evaluated on their genuine ability to integrate and comprehend both visual and textual information.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/64de37ee5e192985054be575/LN8IQGwUJI4NYtQo1wav8.png)

## Key Features
- **Multimodal Understanding:** The dataset includes a diverse set of questions that require models to interpret and integrate both visual and textual information, reflecting real-world scenarios where users often interact with embedded content.
- **Increased Complexity:** MMMU-Pro introduces a vision-only input setting and increases the number of candidate options from 4 to 10, making it significantly harder for models to rely on guessing or exploiting shortcuts.
- **Real-World Simulation:** The vision-only questions are derived from screenshots or photos captured within a simulated display environment. These variations include different backgrounds, font styles, and sizes, closely mimicking real-world conditions where users might provide integrated visual-textual content.

# Dataset Details
The dataset is organized into two subsets:

- **Standard:** This subset increases the number of candidate answers to 10, making it more challenging for models to guess the correct answer.
  
  - `id`: Unique identifier for each question.
  - `question`: The textual question that needs to be answered.
  - `options`: A list of 10 possible answers for the question.
  - `explanation`: A detailed explanation of the correct answer, useful for understanding the reasoning behind it.
  - `image_[num]`: Associated images relevant to the question, where `[num]` is a placeholder for image numbering (e.g., image_1, image_2).
  - `image_type`: Describes the type of images included (e.g., chart, diagram, map).
  - `answer`: The correct answer from the list of options.
  - `topic_difficulty`: A measure of the difficulty of the topic.
  - `subject`: The academic subject or field to which the question belongs.

- **Vision:** In this subset, questions are embedded within screenshots or photos, and models must integrate visual and textual information to answer correctly. No separate text is fed into the model.
  
  - `id`: Unique identifier for each question.
  - `image`: The image containing both the question and information needed to answer it.
  - `answer`: The correct answer to the question.
  - `subject`: The academic subject or field to which the question belongs.

## Usage

```
from datasets import load_dataset

mmmu_pro_vision = load_dataset("MMMU/MMMU_Pro", "vision")
mmmu_pro_standard_4 = load_dataset("MMMU/MMMU_Pro", "standard (4 options)")
mmmu_pro_standard_10 = load_dataset("MMMU/MMMU_Pro", "standard (10 options)")
```

# Methods
- **Filtering Questions:** Initially, questions answerable by text-only models were filtered out. Four strong open-source LLMs were tasked with answering the MMMU questions without images. Questions consistently answered correctly were excluded, resulting in a refined dataset.
- **Augmenting Candidate Options:** To reduce the reliance on option-based guessing, the number of candidate answers was increased from four to ten, making the task significantly more complex.
- **Enhancing Evaluation with Vision-Only Input Setting:** To further challenge models, a vision-only input setting was introduced. Questions are embedded in screenshots or photos, demanding integration of visual and textual information without separate text input.

# Overall Results
- **Comparison with MMMU:** The combined challenges of additional candidate options and vision-only input resulted in a substantial performance decrease from the original MMMU.

|Model                |MMMU-Pro|MMMU (Val)|
|---------------------|--------|----------|
|GPT-4o (0513)        |51.9    |69.1      |
|Claude 3.5 Sonnet    |51.5    |68.3      |
|Gemini 1.5 Pro (0801)|46.9    |65.8      |
|Gemini 1.5 Pro (0523)|43.5    |62.2      |
|InternVL2-Llama3-76B |40.0    |58.3      |
|GPT-4o mini          |37.6    |59.4      |
|InternVL2-40B        |34.2    |55.2      |
|LLaVA-OneVision-72B  |31.0    |56.8      |
|InternVL2-8B         |29.0    |51.2      |
|MiniCPM-V 2.6        |27.2    |49.8      |
|VILA-1.5-40B         |25.0    |51.9      |
|Llava-NEXT-72B       |25.1    |49.9      |
|LLaVA-OneVision-7B   |24.1    |48.8      |
|LLaVA-NeXT-34B       |23.8    |48.1      |
|Idefics3-8B-Llama3   |22.9    |46.6      |
|Phi-3.5-Vision       |19.7    |43.0      |
|LLaVA-NeXT-7B        |17.0    |35.3      |
|LLaVA-NeXT-13B       |17.2    |36.2      |

*Table 1: Overall results of different models on MMMU-Pro and MMMU (Val).*

## Disclaimers
The guidelines for the annotators emphasized strict compliance with copyright and licensing rules from the initial data source, specifically avoiding materials from websites that forbid copying and redistribution. 
Should you encounter any data samples potentially breaching the copyright or licensing regulations of any site, we encourage you to [contact](#contact) us. Upon verification, such samples will be promptly removed.

## Contact
- Xiang Yue: xiangyue.work@gmail.com

# Citation
**BibTeX:**
```bibtex
@article{yue2024mmmu,
  title={MMMU-Pro: A More Robust Multi-discipline Multimodal Understanding Benchmark},
  author={Xiang Yue and Tianyu Zheng and Yuansheng Ni and Yubo Wang and Kai Zhang and Shengbang Tong and Yuxuan Sun and Botao Yu and Ge Zhang and Huan Sun and Yu Su and Wenhu Chen and Graham Neubig},
  journal={arXiv preprint arXiv:2409.02813},
  year={2024}
}
```