# Detecting Arabic Sarcasm with Multi-Task BERT

**Author:**  
Mohammad Jamal AlKhatib  

---

## Abstract

This project develops a robust machine learning model to detect sarcasm in Arabic YouTube comments. Leveraging a multi-task learning strategy with BERT, the model is trained to simultaneously perform sarcasm detection, sentiment analysis, and speech act classification. This approach enhances the model's ability to capture intricate linguistic nuances and contextual cues inherent in informal Arabic text.

---

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Training and Testing Process](#training-and-testing-process)
- [Results and Discussion](#results-and-discussion)
- [Conclusion](#conclusion)
- [References](#references)

---

## Introduction

Sarcasm detection poses unique challenges in natural language processing, especially for Arabic due to its rich morphology and diverse dialects. With the rapid expansion of online content, accurately identifying sarcasm in YouTube comments is vital for improving sentiment analysis and content moderation. This project harnesses a pre-trained BERT model and adopts a multi-task training strategy to address these challenges by jointly learning sarcasm detection, sentiment analysis, and speech act classification.

---

## Dataset

### Chosen Dataset

- **Source:**  
  YouTube comments collected from channels such as "BidonWaraq," "mmr_sa1," "POWR-Esports," and "thmanyahPodcasts."

- **Annotations:**  
  Each comment includes multiple annotations:
  - **Sarcasm:** Sarcastic or non-sarcastic
  - **Sentiment:** Positive, negative, neutral, or mixed
  - **Speech Act:** Categories such as expression, assertion, question, recommendation, or request
  - Additional metadata such as video title, unique video/channel identifiers, dangerous content flags, and annotation rationales.

This dataset was curated and annotated by Khalaya to support advanced NLP research.

---

## Methodology

### Multi-Task BERT Approach

- **Model:**  
  A pre-trained BERT-based model fine-tuned for Arabic text using multi-task learning to handle:
  - Sarcasm Detection
  - Sentiment Analysis
  - Speech Act Classification

- **Data Preprocessing:**  
  - Removal of hashtags, mentions, emojis, punctuation, numbers, and non-Arabic characters.
  - Stop-word removal and stemming to reduce noise and focus on essential linguistic features.

- **Training Strategy:**  
  - The model is trained on TPUs with iterative experiments varying hyperparameters (epochs, learning rates, batch sizes, and max token lengths) to optimize performance.
  - Early stopping based on validation performance is employed to avoid overfitting.

---

## Training and Testing Process

The project experiments involved several trials:

1. **Baseline Experiments:**  
   - Initial training with the MARBERTv2 model using raw data (no preprocessing) and varying hyperparameters.
   
2. **Hyperparameter Tuning:**  
   - Subsequent experiments adjusted epochs, learning rates, and batch sizes to evaluate their impact on task performance.
   
3. **Preprocessing Implementation:**  
   - Comprehensive text cleaning was introduced (e.g., removal of non-essential characters and stop-words), significantly enhancing performance.
   
4. **Specialized Sarcasm Detection:**  
   - An experiment using a pre-trained BERT model from the American University in Beirut (AUB) focused solely on sarcasm detection (ongoing work).

The code for these experiments is detailed in the appendix for full reproducibility.

---

## Results and Discussion

### Overview of Experimental Results

| Experiment | Task                     | Accuracy | Macro F1-Score | Weighted F1-Score |
|------------|--------------------------|----------|----------------|-------------------|
| **1st**    | Sentiment Analysis       | 86%      | 61%            | 86%               |
|            | Speech Act Classification| 81%      | 68%            | 82%               |
|            | Sarcasm Detection        | 88%      | 74%            | 90%               |
| **2nd**    | Sentiment Analysis       | 68%      | 39%            | 63%               |
|            | Speech Act Classification| 67%      | 23%            | 62%               |
|            | Sarcasm Detection        | 58%      | 48%            | 67%               |
| **3rd**    | Sentiment Analysis       | 79%      | 54%            | 78%               |
|            | Speech Act Classification| 74%      | 51%            | 73%               |
|            | Sarcasm Detection        | 65%      | 52%            | 73%               |
| **4th**    | Sentiment Analysis       | 85%      | 68%            | 85%               |
|            | Speech Act Classification| 82%      | 68%            | 82%               |
|            | Sarcasm Detection        | 91%      | 77%            | 92%               |
| **5th**    | Sarcasm Detection        | Ongoing  | N/A            | N/A               |

- **Key Insights:**
  - Initial experiments provided a strong baseline, particularly in sentiment and sarcasm detection.
  - Hyperparameter tuning in the 2nd and 3rd experiments highlighted the sensitivity of the model to training configurations.
  - The introduction of advanced preprocessing in the 4th experiment significantly boosted performance across all tasks.
  - The 5th experiment focuses on further optimizing sarcasm detection using a specialized pre-trained model.

---

## Conclusion

This project demonstrates that a multi-task learning strategy with BERT can effectively address the challenges of detecting sarcasm in Arabic YouTube comments. By integrating sentiment analysis and speech act classification, the model benefits from a richer contextual understanding, leading to improved performance—especially after applying comprehensive preprocessing techniques. These results pave the way for more nuanced NLP systems tailored to the complexities of informal Arabic language.

---

## References

1. Abdul-Mageed, et al. – Contextualized models for Arabic sarcasm detection.
2. Hengle, et al. – Combining context-free and contextualized representations for Arabic sarcasm detection and sentiment analysis.
3. Farha and Magdy – Development of the ArSarcasm dataset for Arabic tweets.
4. Alahmdi – "Arabic YouTube Comments by Khalaya" dataset, available on Kaggle.

---

This README provides an overview of the project's objectives, methodology, experiments, and outcomes. For detailed code and further documentation, please refer to the appendix and supplementary materials. Feel free to reach out for any questions or collaboration opportunities!

---
