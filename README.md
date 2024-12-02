# [Lottery Rank Pruning Adaptation](https://www.mdpi.com/2227-7390/12/23/3744)

## Abstract

Recent studies on parameter-efficient fine-tuning (PEFT) have introduced effective and efficient methods for fine-tuning large language models (LLMs) on downstream tasks using fewer parameters than required by full fine-tuning. Low-rank decomposition adaptation (LoRA) significantly reduces the parameter count to 0.03\% of that in full fine-tuning, maintaining satisfactory performance when training only two low-rank parameters. However, limitations remain due to the lack of task-specific parameters involved in training. To mitigate these issues, we propose the Lottery Rank-Pruning Adaptation (LoRPA) method, which utilizes the Lottery Ticket Hypothesis to prune less significant parameters based on their magnitudes following initial training. Initially, LoRPA trains with a relatively large rank size and then applies pruning to enhance performance in subsequent training with fewer parameters. We conducted experiments to compare LoRPA with LoRA baselines, including a setting with a relatively large rank size. Experimental results on the GLUE dataset with RoBERTa demonstrate that LoRPA achieves comparable results on the base scale while outperforming LoRA with various rank sizes by 0.04\% to 0.74\% on a large scale across multiple tasks. Additionally, on generative summarization tasks using BART-base on the CNN/DailyMail and XSum datasets, LoRPA outperformed LoRA at the standard rank size and other PEFT methods in most of the metrics. These results validate the efficacy of lottery pruning for LoRA in downstream natural-language understanding and generation tasks.

![figure1](https://github.com/user-attachments/assets/0dd907a4-d724-4f57-a0ae-f866a294f4cd)

## Process of Magnitude Pruning

![image](https://github.com/user-attachments/assets/7e1b322e-4e9e-418a-9149-d2292b94e026)

Process of magnitude-based pruning. (a) Load rank matrices A, B trained from initial training. (b) Calculate the importance score S for each rank based on magnitude by the absolute value of the product. (c) Sort scores in ascending order and select the bottom p\% ranks to prune. (d) Concatenate ranks that are not within prune indices to obtain pruned rank matrices.

## Setup

### Install Dependencies

Install all the required Python packages listed in the `requirements.txt` file:

```
pip install -r requirements.txt
conda activate lorpa
```

### Run the Training Script

Execute the training script to start fine-tuning the model using LoRPA

Sample Training Script
```
bash train.sh
```
To reproduce the experimental results presented in this paper, please ensure that each task's run script includes the same hyperparameter values as those provided in the appendix of the paper.

### License

This project is licensed under the MIT License. You are free to use, modify, and distribute this software in accordance with the terms of the license.
