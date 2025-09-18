# SAIL-VL2

<div align="center">
  <img src="assets/logo/logo_with_name.jpeg" width="80%" alt="SAIL-VL2 Logo">
</div>

<font size=3><div align='center' >  
[[📖 Technique Report](https://arxiv.org/abs/2509.14033)] 
[[🤗 HuggingFace](https://huggingface.co/BytedanceDouyinContent)]
</div></font>

We are very excited to introduce **SAIL-VL2** 🚀, a state-of-the-art visual language model that significantly outperforms existing models in various visual language tasks.

## 🔥 Updates

- **`2025.09.18`** 🌟 **SAIL-VL2 Technical Report** is now available at [arxiv](https://arxiv.org/abs/2509.14033).


## 🌟 Highlights
- SAIL-VL2 is powerful, efficient, and achieves top results under 2B parameters.
- SAIL-VL2-Thinking boosts complex reasoning, matching larger models.
- SAIL-VL2 excels in fine-grained visual tasks beyond similar-scale models.

<div align="center">
  <img src="assets/figures/performance.png" width="100%" alt="SAIL-VL2 Performance">
</div>

## Contents
- [🔥 Updates](#-updates)
- [🌟 Highlights](#-highlights)
- [🎬 Quick Start](#-quick-start)
- [👀 Introduction](#-introduction)
- [📚 Training Strategy](#-training-strategy)
    - [🌟 Data construction](#-data-construction)
    - [🌟 Pre-Train](#-pre-train)
    - [🌟 Post-Train](#-post-train)
- [📈 Experimental Results](#-experimental-results)
    - [🌟 Performance of 2B series](#-performance-of-2b-series)
    - [🌟 Performance of 8B series](#-performance-of-8b-series)
    - [🌟 Performance of Thinking-mode models](#-performance-of-thinking-mode-models)
- [🙏 Acknowledge](#-acknowledge)
- [✒️ Citation](#️-citation)
- [📜 License](#-license)
- [📧 Contact](#-contact)


## 🎬 Quick Start


```python
import torch
from transformers import AutoTokenizer, AutoModel, AutoProcessor
from PIL import Image


model_path = "your model path"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
device = torch.cuda.current_device()
model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,).to(device)

print("##### with images")
messages = [
    {"role": "user", "content": [{"type": "image", "image": 'image_path'}, 
    {"type": "text", "text": "describe the image"}]}
]
text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

image_path = 'your image path'
image = Image.open(image_path)
inputs = processor(images=image, text=text, return_tensors="pt", padding=True, truncation=True).to(model.device).to(torch.bfloat16)

generated_ids = model.generate(**inputs, max_new_tokens=512)
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
response = response.split('<|im_end|>')[0].strip()
print(response)


print("##### without images")
messages = [
    {
        "role": "user",
        "content": [{"type": "text", "text": "中国的首都是哪里？"}]
    }
]
text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = processor(images=None, text=text, return_tensors="pt", padding=True, truncation=True).to(model.device).to(torch.bfloat16)
generated_ids = model.generate(**inputs, max_new_tokens=512)
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
response = response.split('<|im_end|>')[0].strip()
print(response)

```

## 👀 Introduction
- **SAIL-VL2 is powerful yet efficient:** With training on 776B tokens, SAIL-VL2 has verified its effectiveness across 106 datasets, achieving state-of-the-art results on a broad spectrum of influential benchmarks under the 2B-parameter scale. Remarkably, even without specialized prompting, the base SAIL-VL2 model delivers highly competitive performance on challenging reasoning benchmarks such as MMMU and MathVista, demonstrating strong out-of-the-box capabilities.

- **SAIL-VL2 as a deep thinker:** Many real-world tasks demand sophisticated reasoning and multi-step thought processes, which remain challenging for standard LVMs. To address this, we develop SAIL-VL2-Thinking, a specialized variant trained with advanced Chain-of-Thought (CoT) and reinforcement learning (RL) strategies. This design substantially improves performance on complex reasoning benchmarks, often matching or even surpassing models with far larger parameter scales, thereby setting a new standard for efficient architectures in high-level reasoning.

- **SAIL-VL2 perceives with clarity:** Fine-grained visual understanding is a critical challenge for multimodal models. SAIL-VL2 delivers high-fidelity perception in tasks such as OCR, high-resolution document layout analysis, and complex chart interpretation, achieving detailed visual grounding beyond models of similar scale.


<div align="center">
  <img src="assets/figures/framework.png" width="100%" alt="SAIL-VL2 Framework">
  <i> Overview of the SAIL-VL2 framework. The architecture is composed of a vision encoder that aligns visual inputs into the representation space of the LLM. A lightweight adapter further transforms visual embeddings into tokenized representations, which are jointly processed with linguistic embeddings for multimodal reasoning and prediction. SAIL-VL2 accommodates multiple LLM backbones, ensuring flexibility and scalability across model configurations.</i>
</div>


## 📚 Training Strategy
### 🌟 Data construction

<div align="center">
  <img src="assets/figures/data.png" width="100%" alt="SAIL-VL2 Data">
  <i> Data construction pipeline for SAIL-VL2 training. High-quality multimodal corpora are construted by curating and filtering open-source datasets, and generating synthetic data, with both components systematically organized to meet the requirements of different training stages.</i>
</div>

### 🌟 Pre-Train

- **Basic Multimodal Pre-Training:** develops SAIL-VL2’s multimodal alignment via SAIL-ViT, LLM and a random MLP adapter, using 64M samples, AdaLRS and 2048 batch size.

- **Multi-task Pre-Training:** strengthens SAIL-VL2’s visual and instruction-following abilities, unfreezes all params, adds instruction-tuning data, uses 180M samples, and skips AdaLRS.

<div align="center">
  <img src="assets/figures/mtpt_scaling.png" width="100%" alt="SAIL-VL2 MTPT Scaling">
  <i> Scaling curves of SAIL-VL2-2B during the multi-task pre-training stage. Results are reported on overall benchmarks, natural-scene VQA datasets, and OCR VQA tasks. ’BMK Score’ denotes the average benchmark score.</i>
</div>


### 🌟 Post-Train

- **Basic Supervised Fine-Tuning:** 4 phases; Model Soup merges homogeneous models.

- **LongCoT Supervised Fine-tuning:** enhances the model’s step-by-step reasoning capabilities for complex problems.

- **RL with Verifiable Rewards:** refines the model by optimizing it against a reward system focused on two primary objectives: the correctness of the final answer and adherence to the specified output format

- **Think-Fusion Supervised Fine-tuning:** enhances the model’s reasoning capabilities while maintaining its broad general understanding.

- **RL with a Mixed Reward System:** enhances the model’s reasoning capabilities through a RL stage

## 📈 Experimental Results
### 🌟 Performance of 2B series
<div align="center">
  <img src="assets/figures/performance_table_2b.png" width="100%" alt="SAIL-VL2 Performance">
  <i> Overall comparison of the SAIL-VL2 series and existing open-source MLLMs (<4B).</i>
</div>

### 🌟 Performance of 8B series
<div align="center">
  <img src="assets/figures/performance_table_8b.png" width="100%" alt="SAIL-VL2 Performance">
  <i> Overall comparison of the SAIL-VL2 series with existing open-source 8B MLLMs and closed-source models.</i>
</div>

### 🌟 Performance of Thinking-mode models
<div align="center">
  <img src="assets/figures/performance_rl.png" width="100%" alt="SAIL-VL2 Performance">
  <i> Evaluation results on OpenCompass multimodal reasoning benchmarks.</i>
</div>

## 🙏 Acknowledge

Our model is built upon numerous outstanding open-source projects, and we are grateful for their contributions. We extend special thanks to the InternVL team, Qwen team, and Apple team for their great base models, and to the BAAI team (Infinity-MM), MAmmoTH-VL team(MAmmoTH-VL-Instruction-12M) for their generous release of data, and to the OpenCompass team for their valuable benchmarks.

## ✒️ Citation

If you find our work helpful for your research, please consider citing our work.   

```
@article{dong2025scalable,
  title={Scalable vision language model training via high quality data curation},
  author={Dong, Hongyuan and Kang, Zijian and Yin, Weijie and Liang, Xiao and Feng, Chao and Ran, Jiao},
  journal={arXiv preprint arXiv:2501.05952},
  year={2025}
}
```
```
@misc{
  sailvl,
  title = {SAIL-VL: Scalable Vision Language Model Training with High Quality Data Curation},
  url = {https://huggingface.co/BytedanceDouyinContent/SAIL-VL-2B/},
  author = {Bytedance Douyin Content Team},
  month = {December},
  year = {2024}
}
```


## 📜 License

This project is licensed under [Apache License 2.0](LICENSE).

## 📧 Contact

If you have any question, please feel free to contact us: BytedanceDouyinContent@bytedance.com

