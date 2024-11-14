import re, math
from collections import defaultdict

def preprocess(file_path):
    with open(file_path, 'r') as file:
        text = file.read().lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split() 
    return tokens

train_data = preprocess("C:/Users/Lab One/Desktop/NLP HW2/train.txt")
test_data = preprocess("C:/Users/Lab One/Desktop/NLP HW2/test.txt")

def generate_ngrams(tokens, n):
    ngram = []
    for i in range(len(tokens) + 1 - n):
        ngram.append(tuple(tokens[i:i + n]))
    return ngram

def train_ngrams(tokens, n):
    ngram_counts = defaultdict(int)
    context_counts = defaultdict(int)
    vocab = set()
    ngrams = generate_ngrams(tokens, n)
    # print(ngrams)
    for ngram in ngrams:
        context = ngram[:-1]
        # print(context)
        ngram_counts[ngram] += 1
        context_counts[context] += 1
        for word in ngram:
            vocab.add(word)
    return ngram_counts, context_counts, vocab


def calculate_probs(ngram_counts, context_counts, vocab, context, ngram):
    # print()
    # print(ngram)
    # print(context)
    # print()
    numerator = ngram_counts[ngram] + 1 
    denominator = context_counts[context] + len(vocab)
    return numerator/denominator

def perplexity(ngram_counts, context_counts, vocab, test_data, n):
    test_ngrams = generate_ngrams(test_data, n)
    log_prob_sum = 0
    for ngram in test_ngrams:
        context = ngram[:-1]
        prob = calculate_probs(ngram_counts, context_counts, vocab, context, ngram)
        log_prob_sum += math.log(prob)
    return math.exp(-log_prob_sum / len(test_ngrams))

for n in range(1, 4):
    ngram_counts, context_counts, vocab = train_ngrams(train_data, n)
    perplexity_val = perplexity(ngram_counts, context_counts, vocab, test_data, n)
    # print(ngram_counts)
    print(f"Perplexity for {n}gram: {perplexity_val}")