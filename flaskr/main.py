from collections import Counter
from nltk import ngrams
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WhitespaceTokenizer as tokenizer
import pandas as pd
import requests
import string
from bs4 import BeautifulSoup
from . hansard import getMP, getHansard, getSpeeches
from . wordcloud import preprocess_speeches, bigrams_frequency_count

from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)

bp = Blueprint('main', __name__, url_prefix='/index')

import logging

log = logging.getLogger('Hansard.main')

# load in the NTLK stopwords to remove articles, preposition and other words that are not actionable
# This allows to create individual objects from a bog of words
# from nltk.tokenize import wordpunct_tokenize
# Lemmatizer helps to reduce words to the base formfrom nltk.stem import WordNetLemmatizer
# Ngrams allows to group words in common pairs or trigrams..etc
# We can use counter to count the objects from collections

mystopwords = None

def get_stopwords():
    print('Getting stopwords')
    global mystopwords
    import os
    with bp.open_resource('static/stopwords.txt', 'r') as F:
        words = F.readlines()
    from wordcloud import STOPWORDS
    STOPWORDS.add('(b)')
    STOPWORDS.add('(a)')
    STOPWORDS.add('(c)')        
    words = [w.strip() for w in words] + list(STOPWORDS)
    words = set(words)
    return words


class HansardMP:
    def __init__(self, postcode_or_constituency, minLength=25):
        self.info = getMP(postcode_or_constituency)
        self.minLength = minLength
        self._wordcloud_freqs = None
        self._speeches = None
        self.stopwords = get_stopwords()
        log.info('Completed MP initialisation for %s', self.full_name)

    @property
    def constituency(self):
        return self.info['constituency']
    
    @property
    def person_id(self):
        return self.info['person_id']

    @property
    def party(self):
        return self.info['party']

    @property
    def image(self):
        return 'https://www.theyworkforyou.com' + self.info['image']

    @property
    def full_name(self):
        return self.info['full_name']

    def get_speeches(self):
        log.info('Getting speeches for %s', self.full_name)
        self._speeches = getSpeeches(self.person_id, self.minLength)
        log.info('Completed getting speeches for %s', self.full_name)

    @property
    def speeches(self):
        if self._speeches is None:
            self.get_speeches()
        return self._speeches

    def get_wordcloud_freqs(self):
        log.info('Preprocessing %d speeches for %s', len(self.speeches), self.full_name)
        words, bigrams = preprocess_speeches(self.speeches, bigrams = True)
        bigrams = list(bigrams)
        log.info('Frequency counting %d words and %d bigrams for %s', len(words), len(bigrams), self.full_name)
        freqs = bigrams_frequency_count(words, bigrams, stopwords=self.stopwords)
        self._wordcloud_freqs = freqs

    @property
    def wordcloud_freqs(self):
        if self._wordcloud_freqs is None:
            self.get_wordcloud_freqs()
        return self._wordcloud_freqs
    
    

def word_frequency(sentence):
    global mystopwords
    # joins all the sentenses
    sentence = " ".join(sentence)
    # creates tokens, creates lower class, removes numbers and lemmatizes the words
    print('starting to tokenize')
    new_tokens = tokenizer().tokenize(sentence)
    print('finished tokenize')
    #new_tokens = sentence.split()
    new_tokens = [t.lower().strip() for t in new_tokens]
    #S = set(stopwords.words('english'))
    if mystopwords is None:
        mystopwords = get_stopwords()
    S = mystopwords
    new_tokens = [t for t in new_tokens if t not in S]
    new_tokens = [t for t in new_tokens if t.isalpha()]
    lemmatizer = WordNetLemmatizer()
    #print('startin lemon')
    #new_tokens = [lemmatizer.lemmatize(t) for t in new_tokens]
    #print('finish lemon')
    # counts the words, pairs and trigrams
    counted = Counter(new_tokens)
    counted_2 = Counter(ngrams(new_tokens, 2))
    counted_3 = Counter(ngrams(new_tokens, 3))
    # creates 3 data frames and returns them
    word_freq = pd.DataFrame(counted.items(), columns=[
                             'word', 'frequency']).sort_values(by='frequency', ascending=False)
    word_pairs = pd.DataFrame(counted_2.items(), columns=[
                              'pairs', 'frequency']).sort_values(by='frequency', ascending=False)
    trigrams = pd.DataFrame(counted_3.items(), columns=[
                            'trigrams', 'frequency']).sort_values(by='frequency', ascending=False)
    return word_freq, word_pairs, trigrams


constituencies = None

from .hansard import getConstituencies

@bp.route('/', methods=('GET', 'POST'))
def dropdown(selected_constituency=None, MP=None, wordclouddata=None):
    global constituencies
    if constituencies is None:
        log.info('Requesting constituencies')
        constituencies = getConstituencies()
        log.info('Completed requesting constituencies')
    return render_template('main/main.html',
                           constituencies=constituencies,
                           selected_constituency=selected_constituency,
                           MP=MP,
                           wordclouddata=wordclouddata)


@bp.route('/search', methods=('GET', 'POST'))
def search():
    select = request.form.get('constituency')
    if select == 'Select constituency':
        return redirect(url_for('main.dropdown'))
    constituency = str(select)
    MP = HansardMP(constituency)
    log.info('Getting wordcloud frequencies for %s', MP.full_name)
    freqs = MP.wordcloud_freqs
    log.info('Completed getting wordcloud frequencies for %s', MP.full_name)
    wordclouddata = [{'word' : x[0], 'value' : x[1]} for x in freqs.most_common(128)]
    wordclouddata.sort(key=lambda x: x['value'], reverse=True)
    return dropdown(selected_constituency=constituency,
                    MP=MP, wordclouddata=wordclouddata)
