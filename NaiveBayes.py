import os, random, math, re
import numpy as np
from collections import defaultdict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def load_samples(folder_path, num, seed=None):

    if seed is not None:
        random.seed(seed)

    pos_folder = os.path.join(folder_path, 'pos')
    neg_folder = os.path.join(folder_path, 'neg')

    pos_files = [os.path.join(pos_folder, f) for f in os.listdir(pos_folder) if f.endswith('.txt')]
    neg_files = [os.path.join(neg_folder, f) for f in os.listdir(neg_folder) if f.endswith('.txt')]

    # selected_pos_files = random.sample(pos_files, min(num, len(pos_files)))
    # selected_neg_files = random.sample(neg_files, min(num, len(neg_files)))
    selected_pos_files = pos_files[:min(num // 2, len(pos_files))]
    selected_neg_files = neg_files[:min(num // 2, len(neg_files))]

    samples = []
    for file in selected_pos_files:
        with open(file, 'r', encoding='utf-8') as f:
            samples.append((f.read().strip(), 1))  
    for file in selected_neg_files:
        with open(file, 'r', encoding='utf-8') as f:
            samples.append((f.read().strip(), 0)) 
    return samples


train_samples = load_samples('aclImdb/train', 1000, 0)   
random.shuffle(train_samples)       #used random seed part and shuffle just to make we have bit well rounded data
test_samples = load_samples('aclImdb/test', 200, 0)
random.shuffle(test_samples)

def preprocess_sentence(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r'[^\w\s]', '', sentence)
    tokens = sentence.split()
    return tokens

train_processed_samples = [(preprocess_sentence(sentence), label) for sentence, label in train_samples]
test_processed_samples = [(preprocess_sentence(sentence), label) for sentence, label in test_samples]
vocab = set()

for data, _ in train_processed_samples:
    for word in data:
        vocab.add(word)

# print(train_processed_samples)

def extract_features(reviews, vocabulary):
    feature_vectors = []
    labels = []
    for tokens, label in reviews:
        word_freq = defaultdict(int)
        for word in tokens:
            word_freq[word] += 1
        feature_vector = {word: word_freq[word] for word in vocabulary}
        feature_vectors.append(feature_vector)
        labels.append(label)
    return feature_vectors, labels

train_features, train_labels = extract_features(train_processed_samples, vocab)
# print(train_features[0])
test_features, test_labels = extract_features(test_processed_samples, vocab)

# print(train_features[0])
# print(len(train_labels))

def calculate_prior_probabilities(labels):
    class_counts = defaultdict(int)
    total_reviews = len(labels)
    for label in labels:
        class_counts[label] += 1
    priors = {label: class_counts[label] / total_reviews for label in class_counts}
    return priors

# this part seems reduntant cosidering we know that we have taken 500 pos and 500 neg; so it will be 0.5
probs = calculate_prior_probabilities(train_labels)
pos_prob = probs[0]
neg_prob = probs[1]
# print(pos_prob, neg_prob) 

def calculate_likelihoods(reviews, labels, vocab):
    word_counts = {0: defaultdict(int), 1: defaultdict(int)} 
    total_word_count = {0: 0, 1: 0} 
    likelihoods = {0: {}, 1: {}}  

    for i in range(len(reviews)):
        tokens = reviews[i]  
        label = labels[i] 

        for word in tokens:
            if word in vocab:
                word_counts[label][word] += tokens[word] 
                total_word_count[label] += tokens[word] 

    for word in vocab:
        likelihoods[0][word] = (word_counts[0][word] + 1) / (total_word_count[0] + len(vocab))  
        likelihoods[1][word] = (word_counts[1][word] + 1) / (total_word_count[1] + len(vocab))  

    return likelihoods

# Example of using the function
likelihoods = calculate_likelihoods(train_features, train_labels, vocab)
# print(likelihoods)

def calculate_posterior(review_tokens, class_label, prob, likelihoods, vocab):
    log_prob = math.log(prob)
    for word, count in review_tokens.items(): 
        # print(word)
        # break
        if word in vocab: 
            if word in likelihoods[class_label]:
                word_likelihood = likelihoods[class_label][word]
            else:
                word_likelihood = 1 / len(vocab)
        else:
            word_likelihood = 1 / len(vocab)  
        log_prob += count * math.log(word_likelihood)
        
    return log_prob

def classify_review(review_tokens, pos_prob, neg_prob, likelihoods, vocab):
    a_prob = calculate_posterior(review_tokens, 1, pos_prob, likelihoods, vocab)
    n_prob = calculate_posterior(review_tokens, 0, neg_prob, likelihoods, vocab)
    if a_prob > n_prob:
        return 1 
    else:
        return 0
    
def test_naive_bayes(test_features, test_labels, pos_prob, neg_prob, likelihoods, vocab):
    correct_predictions = 0
    total_reviews = len(test_features)
    preds = []
    for i in range(total_reviews):
        tokens = test_features[i] 
        actual_label = test_labels[i] 
        predicted_label = classify_review(tokens, pos_prob, neg_prob, likelihoods, vocab)
        preds.append(predicted_label)
        if predicted_label == actual_label:
            correct_predictions += 1

    accuracy = correct_predictions / total_reviews
    return accuracy, preds

accuracy, preds = test_naive_bayes(test_features, test_labels, pos_prob, neg_prob, likelihoods, vocab)
conf_matrix = confusion_matrix(test_labels, preds)
precision = precision_score(test_labels, preds)
recall = recall_score(test_labels, preds)
f1 = f1_score(test_labels, preds)

print(f"Accuracy: {accuracy*100}")
print("Confusion Matrix:")
print(conf_matrix)
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")