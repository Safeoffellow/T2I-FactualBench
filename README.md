
<p align="center">
  <h2 align="center">[ACL 2025 Main] T2I-FactualBench: Benchmarking the Factuality of Text-to-Image Models with Knowledge-Intensive Concepts
</h2>
  <p align="center">
    <a><strong>Ziwei Huang<sup>1</sup> , </strong></a>
    <a><strong>Wanggui He<sup>2</sup> , </strong></a>
    <a><strong>Quanyu Long<sup>3</sup> , </strong></a>
    <a><strong>Yandi Wang<sup>1</sup>  </strong></a>
    <a><strong>Haoyuan Li<sup>2</sup> ,  </strong></a>
    <br>
    <a><strong>Zhelun Yu<sup>2</sup> , </strong></a>
    <a><strong>Fangxun Shu<sup>2</sup> ,  </strong></a>
    <a><strong>Long Chan<sup>2</sup> , </strong></a>
    <a><strong>Hao Jiang<sup>2</sup>   </strong></a>
    <a><strong>Fei Wu<sup>1</sup>   </strong></a>
    <a><strong>Leilei Gan<sup>1*</sup>   </strong></a>
    <br>
    <sup>1</sup> Zhejiang University&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<sup>2</sup> Alibaba Group&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<sup>3</sup> Nanyang Technological University&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp
    <br>
    <sup>*</sup> Corresponding author &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp
    </br>
    </br>
        <a href="https://arxiv.org/abs/2412.04300">
        <img src='https://img.shields.io/badge/T2I--FactualBench-Arxiv-red' alt='Paper PDF'></a>
        <a href="https://huggingface.co/datasets/Sakeoffellow001/T2i_Factualbench">
        <img src='https://img.shields.io/badge/Dataset-HuggingFace-yellow' alt='Dataset'></a>
  </p>
</p>

## Introduction

This repository contains code and links to the paper "T2I-FactualBench: Benchmarking the Factuality of Text-to-Image Models with Knowledge-Intensive Concepts". T2I-FactualBench is the benchmark to evaluate the factuality of text-to-image models when generating images that involves knowledge-intensive concepts. We propose a three-tiered knowledge-intensive text-to-image generation framework, spanning from the basic memorization of individual knowledge concepts to the more complex composition of multiple knowledge concepts. To conduct an effective and efficient evaluation, we also introduce a multi-round visual question answering (VQA)-based evaluation framework aided by advanced multi-modal LLMs.

## Multi-Round VQA

This Multi-Round VQA framework consists of three VQA tasks: (1) Concept Factuality Evaluation; (2) Instantiation Completeness Evaluation and (3) Composition Factuality Evaluation

<img src="assets/evaluation_short.png" alt="multi-round_vqa" width="500" />

## Start
### 1. Dataset Download

To get started with T2I-FactualBench, you first need to download the dataset. This includes a collection of concept images, which will be used for evaluating the factuality of text-to-image generation models.

The concept image dataset can be downloaded from Hugging Face. To facilitate this, we have provided a Python script `download.py` that will automatically download and extract the dataset.

#### Instructions:
1. Clone or download this repository to your local machine.
2. Navigate to the `data` directory:
   ```bash
   cd data
   ```
3. Run the download.py script:
  ```bash
  python download.py
  ```

4. The script will download the concept_image.tar.gz file from Hugging Face and extract it to the data directory.

### 2. Image Generation

Once the dataset is downloaded, you can proceed to generate images using the provided models. We have included a shell script `generate_images.sh` which runs the image generation process using multiple GPUs.

The `generate_images.sh` script utilizes the `torchrun` command to run a distributed image generation process across multiple GPUs. The command runs the `generate_all.py` script, which generates images using various models based on prompts provided in the `data/prompts` directory.

To generate the images, run the following command:

```bash
bash scripts/generate_images.sh
```

This will trigger the following command:

```bash
torchrun \
  --nproc_per_node=4 \
  --master_port=29501 \
  src/generate_all.py \
  --model sdxl \
  --prompt_dir data/prompts \
  --output_dir results/ \
  --level SKCM \
  --knowledge_injection text
```

--model sdxl: The model to be used for image generation. In this case, sdxl (Stable Diffusion XL) is chosen. Other available models include:
  - dalle3
  - flux
  - sd3.5
  - sdxl
  - sd1.5
  - playground2.5
  - pixart

To switch models, simply replace sdxl with any of the listed models. If you want to add a new model, you can add it to the src/generators directory.
--knowledge_injection text: This flag specifies that the model should utilize text-based knowledge injection. You can change this to other types of knowledge injection depending on your experiment.

### 3. Evaluation

Once the images have been generated, the next step is to evaluate their factuality using either the Multi-Round VQA framework or traditional metrics. We provide scripts for both evaluation methods: Multi-Round VQA and traditional metrics (CLIP & DINO).

#### 3.1 Multi-Round VQA Evaluation

The Multi-Round VQA evaluation framework involves asking multiple rounds of visual questions to assess the factuality of generated images. It contains three evaluation tasks:

1. **Concept Factuality Evaluation**
2. **Instantiation Completeness Evaluation**
3. **Composition Factuality Evaluation**

To run the Multi-Round VQA evaluation, use the following script:

```bash
bash bash/eval_multi_round/eval.sh
```

#### 3.2 Traditional Metrics Evaluation

For a more traditional evaluation approach, we also provide scripts that use pre-trained models like CLIP and DINO for evaluating the generated images' factuality.

```bash
bash ./eval/eval_clip.sh
```

Or

```bash
bash ./eval/eval_dino.sh
```

## Leaderboard

Want to submit results on the leaderboard? Please email the authors.

<img src="assets/model_performance.png" alt="result_1" width="500" />

<img src="assets/category_performance_whole.png" alt="result_2" width="650" />

## Example Cases of Diverse Generations by Models on T2I-FactualBench 

### SKCM
<img src="assets/SKCM.png" alt="SKCM" width="900" />

### SKCI
<img src="assets/SKCI.png" alt="SKCI" width="900" />

### MKCC
<img src="assets/MKCC.png" alt="MKCC" width="900" />

## Release

- [x] Release the code for generating prompts and evaluation.
- [ ] Release images generated by different models.
- [x] Release 1600 knowledge-intensive concepts and 3000 prompts of T2I-FactualBench.
- [x] Release the paper of T2I-FactualBench on arXiv.
