from nltk.util import ngrams
from collections import Counter
from nltk.stem import WordNetLemmatizer
import string
import re

def preprocess_speeches(speeches, bigrams=True):
    #Concatenates the speeches into one string of text
    text = '\n'.join(speeches)
    #Removes all punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    #Removes text inside parenthesis
    text = re.sub(r'\d+', '', text)
    #Converts text into lowercase, and splits by whitespace
    words = text.lower().split()
    words = [w for w in words if len(w) > 1]
    lemmatizer = WordNetLemmatizer()
    #Reduce each word to its root
    words = [lemmatizer.lemmatize(w, pos='v') for w in words]
    if bigrams == False:
        return words
    bigrams = ngrams(words, 2)
    return words, bigrams

def preprocess_speeches_for_embeddings(speeches, stopwords=[], min_length=25):
    out = []
    for i in range(len(speeches)):
        text = speeches[i]['text']
        #Removes all punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        #Removes text inside parenthesis
        text = re.sub(r'\d+', '', text)
        #Lower case
        text = text.lower()
        #Remove stopwords
        text = " ".join(w for w in text.split() if not w in stopwords)
        if len(text.split()) >= min_length:
            out.append({'idx' : i, 'timestamp' : speeches[i]['timestamp'], 'text' : text})
    return out


def bigrams_frequency_count(words, bigrams, stopwords = []):
    #Counts words as before, excluding stopwords
    words_counter = Counter([w for w in words if not w in stopwords])
    #Counts bigrams, excluding any that are in the list of stopwords
    bigrams_counter = Counter([' '.join(w) for w in bigrams if not w[0] in stopwords and not w[1] in stopwords])
    #Gets a dictionary of the 250 most common words
    words_counter = words_counter.most_common(250)
    words_counter = {w[0] : w[1] for w in words_counter}
    #Gets a dictionary with the 250 most common bigrams
    bigrams_counter = bigrams_counter.most_common(250)
    bigrams_counter = {w[0] : w[1] for w in bigrams_counter}
    #Removes counts from single words, if they appear in the list of bigrams
    for w in words_counter.keys():
        for b in bigrams_counter.keys():
            bg = b.split()
            if w == bg[0] or w == bg[1]:
                words_counter[w] -= bigrams_counter[b]
    return Counter(words_counter) + Counter(bigrams_counter)