import re
from collections import Counter

def preprocess_corpus(filepath):
    
    with open(filepath, encoding='utf-8') as file:
        data = file.read()

    # Replace certain special characters with spaces
    data = data.replace('\n', ' ').replace('\t', ' ').replace('-', ' ').replace('—', ' ')
    
    # Remove all non-alphabetical characters (numbers, alphanumerics, special symbols)
    data = re.sub(r'[^a-zA-Z\s]', '', data)
    
    # Lowercase and split into words
    data = data.lower().strip().split(' ')
    
    # Remove any empty strings and non-alphabetical tokens
    data = list(filter(lambda word: word.isalpha() and not word.startswith('http') and len(word) <= 20, data))

    # Word frequency count
    word_counter = Counter(data)
    
    return data, word_counter

# Call preprocess_corpus when the module is imported
corpus_path = "cancer_new.txt"
data, word_counter = preprocess_corpus(corpus_path)
