# R1-Omni: Explainable Omni-Multimodal Emotion Recognition with Reinforcement Learning

[![ModelScope](https://img.shields.io/badge/ModelScope-R1Omni-blue)](https://modelscope.cn/models/iic/R1-Omni-0.5B)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-R1Omni-yellow)](https://huggingface.co/StarJiaxing/R1-Omni-0.5B)
[![arXiv](https://img.shields.io/badge/arXiv-2503.05379-red)](https://arxiv.org/abs/2503.05379)


## 📖 Introduction
**R1-Omni** is the industry’s first application of Reinforcement Learning with Verifiable Reward (RLVR) to an Omni-multimodal large language model. We focus on emotion recognition, a task where both visual and audio modalities play crucial roles, to validate the potential of combining RLVR with Omni model. Our findings reveal several key insights:
1) **Enhanced Reasoning Capability**: R1-Omni demonstrate superior reasoning abilities, enabling a clearer understanding of how visual and audio information contribute to emotion recognition.
2) **Improved Understanding Capability**: Compared to SFT, RLVR significantly boosts performance on emotion recognition tasks.
3) **Stronger Generalization Capability**: RLVR models exhibit markedly better generalization capabilities, particularly excelling in out-of-distribution scenarios.



## 📦 Model Download
We chose the open-source Omni model HumanOmni-0.5B as our base model. We have open-sourced the following: the base model HumanOmni-0.5B, the cold-start model EMER-SFT, the model MAFW-DFEW-SFT fine-tuned directly on the MAFW and DFEW training sets, and our final model R1-Omni.
<div align="center">

| **Model**              | **HuggingFace**                                                                 | **ModelScope**                                                          |
|------------------------|---------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| `HumanOmni-0.5B`      |  [![HF](https://img.shields.io/badge/🤗-Download-yellow)](https://hf.co/StarJiaxing/HumanOmni-0.5B) | [![MS](https://img.shields.io/badge/ModelScope-Download-blue)](https://modelscope.cn/models/myroot/HumanOmni-0.5B) |
| `EMER-SFT`      |  [![HF](https://img.shields.io/badge/🤗-Download-yellow)](https://hf.co/StarJiaxing/EMER-SFT-0.5B)  | [![MS](https://img.shields.io/badge/ModelScope-Download-blue)](https://modelscope.cn/models/myroot/EMER-SFT-0.5B)  |
| `MAFW-DFEW-SFT`       | [![HF](https://img.shields.io/badge/🤗-Download-yellow)](https://hf.co/StarJiaxing/MAFW-DFEW-0.5B)         | [![MS](https://img.shields.io/badge/ModelScope-Download-blue)](https://modelscope.cn/models/myroot/MAFW-DFEW-0.5B)         |
| `R1-Omni`       | [![HF](https://img.shields.io/badge/🤗-Download-yellow)](https://hf.co/StarJiaxing/R1-Omni-0.5B)         | [![MS](https://img.shields.io/badge/ModelScope-Download-blue)](https://modelscope.cn/models/myroot/R1-Omni-0.5B)         |
</div>



## 🏆 Performance

Below are the performance on emotion recognition datasets. We use symbols to indicate whether the data is **in-distribution (⬤)** or **out-of-distribution (△)**.

| Method                          | DFEW (WAR) ⬤ | DFEW (UAR) ⬤ | MAFW (WAR) ⬤ | MAFW (UAR) ⬤ | RAVDESS (WAR) △ | RAVDESS (UAR) △ |
|----------------------------------|---------------|---------------|---------------|---------------|------------------|------------------|
| HumanOmni-0.5B                   | 22.64         | 19.44         | 20.18         | 13.52         | 7.33             | 9.38             |
| EMER-SFT                         | 38.66       | 35.31         | 38.39         | 28.02         | 29.00            | 27.19            |
| MAFW-DFEW-SFT                    | 60.23         | 44.39         | 50.44         | 30.39         | 29.33            | 30.75            |
| R1-Omni                          | 65.83         | 56.27         | 57.68         | 40.04         | 43.00            | 44.69            |

![image](https://github.com/user-attachments/assets/f0239753-8a70-4e8b-9088-35c420595978)

### Legend
- **⬤**: Indicates **in-distribution data** (DFEW and MAFW).
- **△**: Indicates **out-of-distribution data** (RAVDESS).




## 📺  Demo
https://github.com/user-attachments/assets/8c73cbe6-5f24-49a9-bef9-bff6c50e4580
```
User: As an emotional recognition expert; throughout the video, which emotion conveyed by the characters is the most obvious to you?  Output the thinking process in <think> </think> and final emotion in <answer> </answer> tags.

HumanOmni-0.5B: fear
EMER-SFT: <think>In the video, a man in a brown jacket stands in front of a vibrant mural, his face showing clear signs of anger. His furrowed brows and open mouth express his dissatisfaction. From his expressions and vocal traits, it can be inferred that he is experiencing intense emotional turmoil. The audio content mentions "fear" and "lowering one's voice," suggesting that he is facing an urgent situation or expressing strong emotions during an emotionally tense moment. Overall, the analysis indicates that the man is experiencing a deep sense of fear and anxiety.</think>
<answer>fear</answer>
MAFW-DFEW-SFT: angry
R1-Omni: <think>In the video, a man in a brown jacket stands in front of a vibrant mural. He is wearing a pink shirt underneath his brown jacket, and his hair is dark and curly. His facial expression is complex, with wide eyes, slightly open mouth, raised eyebrows, and furrowed brows, revealing surprise and anger. Speech recognition technology suggests that his voice contains words like "you", "lower your voice", "freaking out", indicating strong emotions and agitation. Overall, he displays an emotional state of confusion, anger, and excitement.</think>
<answer>angry</answer>
GT:angry

```

https://github.com/user-attachments/assets/1ee5f969-a13d-4706-a835-9790faf61407
```
HumanOmni-0.5B: happy
EMER-SFT: <think>In the video, we see a woman in an indoor setting that appears to be a personal space or a study room. She is wearing a red headband and a beige blazer over a white top, with her hair neatly tied back. Her facial expression shows slight confusion and surprise, as if she has encountered something unexpected. As time goes on, her emotions seem to calm down, but the specific reasons for this are not explained. In the audio, there is a long silence between "but" and "it was," indicating hesitation and uncertainty. In the text, the subtitle reads: "But it was interesting." This sentence may express the confusion and surprise of the person. Based on the video clues of the woman's slightly confused and surprised expression, as well as her overall appearance and facial expression showing minor confusion and surprise, we can infer that this sentence may have a hesitant or uncertain tone. Additionally, the audio clue describing a long silence between "but" and "it was" also indicates some hesitation and uncertainty. Therefore, this sentence expresses the woman's interest and curiosity in something, while also showing a certain level of hesitation and uncertainty.</think>
<answer>surprise</answer>
MAFW-DFEW-SFT: happy
R1-Omni: <think>In the video, in the opening scene, we see a woman with her eyes slightly closed and mouth slowly opening as if she is laughing. Her facial expression appears somewhat joyful, which may indicate that she is experiencing some pleasant or amusing situation at that moment. In the audio, there are no pauses between sentences, they flow continuously, and the tone is light and cheerful. Combined with the text content, it can be felt that the character is in a very happy and positive emotional state. In the text, the subtitle reads: "It was interesting." This sentence may express the woman's satisfaction and curiosity towards something or someone. Based on the laughter and joyful emotion displayed by the woman in the video clues, as well as the description of continuous flowing, light and cheerful tone in the audio clues, we can infer that this sentence expresses the woman's interest and curiosity towards something. The joy in this sentence aligns with the overall display of happiness and positivity.</think>
<answer>happy</answer>
GT:happy
```

## 🛠️ Environment Setup
Our code is built on the R1-V framework. To set up the environment, please follow the installation instructions in the [R1-V repository](https://github.com/Deep-Agent/R1-V/)

## 🔍 Inference
Our inference code is based on the implementation from **HumanOmni**. To ensure the model runs inference smoothly, follow these steps:

1. **Download the Required Models**：
   - [siglip-224](https://huggingface.co/google/siglip-base-patch16-224).
   - [whisper-large-v3](https://huggingface.co/openai/whisper-large-v3).

2. **Update the Configuration File**：
   - In the directory where you downloaded the R1-Omni model, locate the config.json file.
   - Update the paths on line 23 and line 31 to point to the local folders where you saved the models.


#### Example: Updating config.json
If you saved the models to the following local paths:：
- `/path/to/local/models/siglip-base-patch16-224`
- `/path/to/local/models/whisper-large-v3`

Update the relevant lines in config.json as follows：
```json
 "mm_audio_tower": "/path/to/local/models/whisper-large-v3",
 "mm_vision_tower": "/path/to/local/models/siglip-base-patch16-224"
```

We provide inference.py for singe video inference. 
```
python inference.py --modal video_audio \
  --model_path ./R1-Omni-0.5B \
  --video_path video.mp4 \
  --instruct "As an emotional recognition expert; throughout the video, which emotion conveyed by the characters is the most obvious to you?  Output the thinking process in <think> </think> and final emotion in <answer> </answer> tags."
```



## 🧠 Training
### Cold Start
we initialize the HumanOmni-0.5B by fine-tuning it on a combined dataset consisting of 232 samples from the [Explainable Multimodal Emotion Reasoning](https://github.com/zeroQiaoba/AffectGPT) dataset and 348 samples from HumanOmni dataset.
An example json file of the training data:
```
[
    {
        "video": "MER24/sample_00000967.mp4",
        "conversations": [
            {
                "from": "human",
                "value": "<video>\n<audio>\nAs an emotional recognition expert; throughout the video, which emotion conveyed by the characters is the most obvious to you? Output the thinking process in <think> </think> and final emotion in <answer> </answer> tags."
            },
            {
                "from": "gpt",
                "value": "<think>The video depicts a bright and tranquil indoor setting, where a man in a white Polo shirt stands by the window, engaged in a phone call. His furrowed brow and open mouth suggest he is experiencing tension and anxiety. According to the audio content of the video, his speech is fast-paced, and his tone is filled with confusion and stress. A comprehensive analysis reveals that the man is facing a moderate level of anxiety, closely linked to the challenging phone conversation he is having. Consequently, the entire emotional analysis report emphasizes his anxiety and nervousness in handling challenging situations.</think>\n<answer>anxious</answer>"
            }
        ]
    },
  ...
]
```
All of the cold-start data will be released as soon as possible.

### RLVR
In this stage, we utilize the training sets from the [MAFW](https://mafw-database.github.io/MAFW/) and DFEW(https://dfew-dataset.github.io/) datasets, comprising a total of 15,306 video sample. 
An example json file of the training data:
```
[
    {
        "video": "DFEW/videos/1.mp4",
        "conversations": [
            {
                "from": "human",
                "value": "<video>\n<audio>\nAs an emotional recognition expert; throughout the video, which emotion conveyed by the characters is the most obvious to you?"
            },
            {
                "from": "gpt",
                "value": "sad"
            }
        ],
    },
  ...
]
```

### wandb
![image](https://github.com/user-attachments/assets/3395bafa-aaba-4212-902d-91067a1cd19a)


## 🤝 Related Work
- [R1-V](https://github.com/Deep-Agent/R1-V)
- [HumanOmni](https://github.com/HumanMLLM/HumanOmni)
- [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)

## 📚 Citation
If you find our work helpful, feel free to give us a cite.
```
{zhao2025r1omniexplainableomnimultimodalemotion,
      title={R1-Omni: Explainable Omni-Multimodal Emotion Recognition with Reinforcement Learning}, 
      author={Jiaxing Zhao and Xihan Wei and Liefeng Bo},
      journal={arXiv preprint arXiv:2503.05379},
      year={2025}
}
```
