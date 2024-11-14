import csv
import numpy as np
from collections import defaultdict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def process_csv(file_path):
    all_words = []
    all_labels = []
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        i = 0
        for row in reader:
            if row and i > 0: 
                # print(row)
                words_part = row[0]
                label_part = row[1]
                words = words_part.strip().split()
                label = int(label_part)
                all_words.append(words)
                all_labels.append(label)
            i += 1

    return all_words, all_labels

words_list, labels_list = process_csv('sentiment_train_dataset.csv')
test_words, test_labels = process_csv('sentiment_test_dataset.csv')
# print(labels_list)
vocab = set()

for data in words_list:
    for word in data:
        vocab.add(word)     # 300 length

word_to_index = {word: idx for idx, word in enumerate(vocab)}

def extract_features(reviews, vocabulary):     #modified it to contain 1 and 0 only
    feature_vectors = []
    for tokens in reviews:
        word_freq = defaultdict(int)
        for word in tokens:
            word_freq[word] += 1
        feature_vector = []
        for word in vocabulary:
            if word_freq[word] > 0:
                feature_vector.append(1)
            else:
                feature_vector.append(0)
        feature_vectors.append(feature_vector)
    return feature_vectors

train_features = extract_features(words_list, vocab)
x_train = np.array(train_features)
x_labels = np.array(labels_list)

test_words = extract_features(test_words, vocab)
test_words = np.array(test_words)
test_labels = np.array(test_labels)

# print(len(labels_list))

def initialize_parameters(input_size, hidden_size):
    w1 = np.random.randn(hidden_size, input_size) * 0.01
    b1 = np.zeros((hidden_size, 1))
    w2 = np.random.randn(1, hidden_size) * 0.01
    b2 = np.zeros((1, 1))
    return w1, b1, w2, b2

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return z > 0

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

def forward_propagation(x, w1, b1, w2, b2):
    z1 = np.dot(w1, x) + b1
    a1 = relu(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = sigmoid(z2)
    return a1, a2

def backward_propagation(x, Y, a1, a2, w1, w2):
    m = x.shape[1]
    dz2 = a2 - Y
    dw2 = (1 / m) * np.dot(dz2, a1.T)
    db2 = (1 / m) * np.sum(dz2)

    da1 = np.dot(w2.T, dz2)
    dz1 = da1 * relu_derivative(a1)
    dw1 = (1 / m) * np.dot(dz1, x.T)
    db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

    return dw1, db1, dw2, db2

def update_parameters(w1, b1, w2, b2, dw1, db1, dw2, db2, learning_rate):
    w1 -= learning_rate * dw1
    b1 -= learning_rate * db1
    w2 -= learning_rate * dw2
    b2 -= learning_rate * db2
    return w1, b1, w2, b2

def train(x_train, y_train, input_size, hidden_size, epochs, learning_rate):
    w1, b1, w2, b2 = initialize_parameters(input_size, hidden_size)

    for epoch in range(epochs):
        for i in range(x_train.shape[0]):
            x = x_train[i].reshape(-1, 1)
            y = y_train[i]

            a1, a2 = forward_propagation(x, w1, b1, w2, b2)
            dw1, db1, dw2, db2 = backward_propagation(x, y, a1, a2, w1, w2)
            w1, b1, w2, b2 = update_parameters(w1, b1, w2, b2, dw1, db1, dw2, db2, learning_rate)

        print(f'Epoch {epoch+1}/{epochs} completed.')

    return w1, b1, w2, b2

def predict(x, w1, b1, w2, b2):
    _ , a2 = forward_propagation(x, w1, b1, w2, b2)
    return 1 if a2 >= 0.5 else 0

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

def evaluate(x_test, y_test, w1, b1, w2, b2):
    y_pred = []
    for i in range(x_test.shape[0]):
        x = x_test[i].reshape(-1, 1)
        pred = predict(x, w1, b1, w2, b2)
        y_pred.append(pred)

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    precision_pos = precision_score(y_test, y_pred, pos_label=1)
    recall_pos = recall_score(y_test, y_pred, pos_label=1)
    f1_pos = f1_score(y_test, y_pred, pos_label=1)

    precision_neg = precision_score(y_test, y_pred, pos_label=0)
    recall_neg = recall_score(y_test, y_pred, pos_label=0)
    f1_neg = f1_score(y_test, y_pred, pos_label=0)

    print(f'accuracy: {accuracy*100}')
    print(f'Confusion Matrix: \n{conf_matrix}')
    print(f"Precision (Positive Class): {precision_pos:.2f}")
    print(f"Recall (Positive Class): {recall_pos:.2f}")
    print(f"F1-Score (Positive Class): {f1_pos:.2f}")
    print(f"Precision (Negative Class): {precision_neg:.2f}")
    print(f"Recall (Negative Class): {recall_neg:.2f}")
    print(f"F1-Score (Negative Class): {f1_neg:.2f}")



print("Variation 1")
input_size = len(vocab)
hidden_size = 256
epochs = 10
learning_rate = 0.01
w1, b1, w2, b2 = train(x_train, x_labels, input_size, hidden_size, epochs, learning_rate)
evaluate(test_words, test_labels, w1, b1, w2, b2)
print()

# Variation 2
print("Variation 2")
input_size = len(vocab)
hidden_size = 256
epochs = 20
learning_rate = 0.01
w1, b1, w2, b2 = train(x_train, x_labels, input_size, hidden_size, epochs, learning_rate)
evaluate(test_words, test_labels, w1, b1, w2, b2)
print()

# Variation 3
print("Variation 3")
input_size = len(vocab)
hidden_size = 256
epochs = 10
learning_rate = 0.015
w1, b1, w2, b2 = train(x_train, x_labels, input_size, hidden_size, epochs, learning_rate)
evaluate(test_words, test_labels, w1, b1, w2, b2)
print()

# Variation 4
print("Variation 4")
input_size = len(vocab)
hidden_size = 256
epochs = 20
learning_rate = 0.015
w1, b1, w2, b2 = train(x_train, x_labels, input_size, hidden_size, epochs, learning_rate)
evaluate(test_words, test_labels, w1, b1, w2, b2)
print()

# Variation 5
print("Variation 5")
input_size = len(vocab)
hidden_size = 128
epochs = 10
learning_rate = 0.01
w1, b1, w2, b2 = train(x_train, x_labels, input_size, hidden_size, epochs, learning_rate)
evaluate(test_words, test_labels, w1, b1, w2, b2)
print()

# Variation 6
print("Variation 6")
input_size = len(vocab)
hidden_size = 128
epochs = 20
learning_rate = 0.01
w1, b1, w2, b2 = train(x_train, x_labels, input_size, hidden_size, epochs, learning_rate)
evaluate(test_words, test_labels, w1, b1, w2, b2)
print()

# Variation 7
print("Variation 7")
input_size = len(vocab)
hidden_size = 128
epochs = 10
learning_rate = 0.015
w1, b1, w2, b2 = train(x_train, x_labels, input_size, hidden_size, epochs, learning_rate)
evaluate(test_words, test_labels, w1, b1, w2, b2)
print()

# Variation 8
print("Variation 8")
input_size = len(vocab)
hidden_size = 128
epochs = 20
learning_rate = 0.015
w1, b1, w2, b2 = train(x_train, x_labels, input_size, hidden_size, epochs, learning_rate)
evaluate(test_words, test_labels, w1, b1, w2, b2)
print()
