
#part 1 of the assignment

import os
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import ne_chunk, pos_tag
from collections import Counter


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')

#initailize tools
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

#define file directory
directory_path = "s:\downloads\HW 2" 
files = ["Text_1.txt", "Text_2.txt", "Text_3.txt"]

#function to load text
def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

#function for tokenization stemming and lemmatization
def preprocess_text(text):
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
    
    stemmed = [stemmer.stem(word) for word in words]
    lemmatized = [lemmatizer.lemmatize(word) for word in words]
    return words, stemmed, lemmatized


def get_named_entities(text):
    sentences = sent_tokenize(text)
    named_entities = []
    for sent in sentences:
        words = word_tokenize(sent)
        tags = pos_tag(words)
        chunked = ne_chunk(tags)
        for chunk in chunked:
            if hasattr(chunk, 'label') and chunk.label() == 'NE':
                named_entities.append(' '.join(c[0] for c in chunk))
    return named_entities


results = []

for file in files:
    file_path = os.path.join(directory_path, file)
    print(f"Processing: {file_path}")
    
    text = load_text(file_path)
    words, stemmed, lemmatized = preprocess_text(text)
    
    #top 20 most common tokens
    word_freq = Counter(lemmatized)
    common_tokens = word_freq.most_common(20)
    
    #named entities
    named_entities = get_named_entities(text)
    named_entity_count = len(named_entities)
    
    
    results.append({
        'File': file,
        'Common Tokens': common_tokens,
        'Named Entity Count': named_entity_count,
        'Sample Named Entities': named_entities[:10]
    })


for result in results:
    print("\nFile:", result['File'])
    print("Top 20 Common Tokens:", result['Common Tokens'])
    print("Named Entity Count:", result['Named Entity Count'])
    print("Sample Named Entities:", result['Sample Named Entities'])

    

#part 2 of the assignment

from nltk import ngrams


nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

directory_path = "s:\downloads\HW 2"  
files = ["Text_1.txt", "Text_2.txt", "Text_3.txt", "Text_4.txt"]

def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def generate_trigrams(text, n=3):
    
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
    
    
    trigrams = list(ngrams(words, n))
    return trigrams


def most_common_ngrams(trigrams, top_n=10):
    trigram_freq = Counter(trigrams)
    return trigram_freq.most_common(top_n)


results = {}

for file in files:
    file_path = os.path.join(directory_path, file)
    print(f"Processing: {file_path}")
    
    
    text = load_text(file_path)
    trigrams = generate_trigrams(text, n=3)
    common_trigrams = most_common_ngrams(trigrams, top_n=10)
    
    
    results[file] = common_trigrams


print("\nTrigram Analysis Results:")
for file, trigrams in results.items():
    print(f"\n{file} - Top 10 Trigrams:")
    for trigram, freq in trigrams:
        print(f"{' '.join(trigram)}: {freq}")


text_4_trigrams = [trigram for trigram, freq in results["Text_4.txt"]]
similarities = {}

for file in ["Text_1.txt", "Text_2.txt", "Text_3.txt"]:
    other_trigrams = [trigram for trigram, freq in results[file]]
    common_count = len(set(text_4_trigrams) & set(other_trigrams))
    similarities[file] = common_count


print("\nSimilarity Between Text_4 and Other Texts:")
for file, count in similarities.items():
    print(f"Common Trigrams with {file}: {count}")
