# NLP Assignment 2 - Fall 2024

## Overview

This assignment covers two fundamental tasks in Natural Language Processing (NLP), each focusing on different approaches to text analysis and classification:

1. **N-Gram Language Model with Perplexity Calculation**
2. **Naive Bayes Classifier for Sentiment Analysis**

Each task emphasizes hands-on implementation from scratch to build practical understanding of NLP concepts and algorithms.

---

## Assignment Questions

### Q1: Implementing an N-Gram Language Model and Testing Perplexity

**Objective:**  
Develop an N-Gram Language Model that calculates the probability of words based on their preceding words in a sequence. Measure the model’s performance using **Perplexity** on a test set.

**Instructions:**
- Load and preprocess the provided `train.txt` file, including text normalization and tokenization.
- Implement an N-Gram model with configurable `n` (e.g., unigram, bigram, trigram).
- Apply smoothing (e.g., add-one smoothing) to handle unseen words.
- Calculate **Perplexity** on a separate test set and analyze performance for different values of `n`.

### Q2: Implementing a Naive Bayes Classifier for Sentiment Analysis

**Objective:**  
Build a **Naive Bayes Classifier** from scratch to classify IMDB movie reviews into positive or negative sentiments.

**Instructions:**
1. **Dataset Preparation:**
   - Load a subset of IMDB movie reviews (500 positive and 500 negative samples for training, 100 each for testing).
   - Preprocess text by removing punctuation, stop words, and converting to lowercase.
2. **Classifier Implementation:**
   - Calculate **prior probabilities** for each class.
   - Estimate **likelihoods** for each word in the vocabulary, using Laplace smoothing.
   - Compute posterior probabilities to classify test reviews.
3. **Evaluation:** Calculate accuracy, confusion matrix, precision, recall, and F1-score on the test set.


## Learning Objectives

This assignment allowed me to:
- Gain practical experience with N-Gram, and Naive Bayes for NLP.
- Enhance skills in data preprocessing, probabilistic and neural model building, and model evaluation.
- Develop a foundational understanding of NLP and its applications in text classification.

