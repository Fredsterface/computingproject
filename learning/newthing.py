# loading in all the essentials for data manipulation
print('reading pandas')
import pandas as pd
#load in the NTLK stopwords to remove articles, preposition and other words that are not actionable
print('reading stopwords')
from nltk.corpus import stopwords
# This allows to create individual objects from a bog of words
print('reading tokenize')
# from nltk.tokenize import wordpunct_tokenize
from nltk.tokenize import WhitespaceTokenizer as tokenizer
from nltk.stem import WordNetLemmatizer
# Lemmatizer helps to reduce words to the base formfrom nltk.stem import WordNetLemmatizer
# Ngrams allows to group words in common pairs or trigrams..etc
print('reading ngrams')
from nltk import ngrams
# We can use counter to count the objects from collections
print('reading counter')
from collections import Counter

def word_frequency(sentence):
    # joins all the sentenses
    sentence = " ".join(sentence)
    # creates tokens, creates lower class, removes numbers and lemmatizes the words
    print('starting to tokenize')
    new_tokens = tokenizer().tokenize(sentence)
    print('finished tokenize')
    #new_tokens = sentence.split()
    new_tokens = [t.lower() for t in new_tokens]
    S = set(stopwords.words('english'))
    new_tokens =[t for t in new_tokens if t not in S]
    new_tokens = [t for t in new_tokens if t.isalpha()]
    lemmatizer = WordNetLemmatizer()
    print('startin lemon')
    new_tokens =[lemmatizer.lemmatize(t) for t in new_tokens]
    print('finish lemon')
    #counts the words, pairs and trigrams
    counted = Counter(new_tokens)
    counted_2= Counter(ngrams(new_tokens,2))
    counted_3= Counter(ngrams(new_tokens,3))
    #creates 3 data frames and returns them
    word_freq = pd.DataFrame(counted.items(),columns=['word','frequency']).sort_values(by='frequency',ascending=False)
    word_pairs =pd.DataFrame(counted_2.items(),columns=['pairs','frequency']).sort_values(by='frequency',ascending=False)
    trigrams =pd.DataFrame(counted_3.items(),columns=['trigrams','frequency']).sort_values(by='frequency',ascending=False)
    return word_freq,word_pairs,trigrams

import json 
with open("../notebooks/larry.json", "r") as F:
    data = json.load(F)

word_freq,word_pairs,trigrams = word_frequency(data)
print(word_freq)
print(word_pairs)
word_freq_array = [{'word': word_freq.iloc[i].word, 'value': word_freq.iloc[i].frequency} for i in range(128)]
word_pairs_array = [{'word': word_pairs.iloc[i].pairs, 'value': word_pairs.iloc[i].frequency} for i in range(128)]
trigrams_array = [{'word': trigrams.iloc[i].trigrams, 'value': trigrams.iloc[i].frequency} for i in range(128)]


