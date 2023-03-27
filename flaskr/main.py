import logging
log = logging.getLogger('Hansard.main')
log.info('At start of main')
log.info('A')
from .hansard import getConstituencies
log.info('B')
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WhitespaceTokenizer as tokenizer
import pandas as pd
import requests
import string
from bs4 import BeautifulSoup
from . hansard import getMP, getHansard, getSpeeches
from . wordcloud import preprocess_speeches, preprocess_speeches_for_embeddings, bigrams_frequency_count
from flask import Flask, render_template, request, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField
import psutil
import numpy as np
from datetime import datetime

log.info('Importing ngrams')
from . import ngrams
log.info('Importing bertopic')
from . import BERTopic 
log.info('Done importing UMAP')
from . import UMAP
log.info('Importing sentence transformer')
from . import HansardSentenceTransformer

from wtforms.validators import DataRequired, Email, EqualTo

from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)
log.info('Getting Bluprint')
bp = Blueprint('main', __name__, url_prefix='/index')
log.info('Got Blueprint')

# load in the NTLK stopwords to remove articles, preposition and other words that are not actionable
# This allows to create individual objects from a bog of words
# from nltk.tokenize import wordpunct_tokenize
# Lemmatizer helps to reduce words to the base formfrom nltk.stem import WordNetLemmatizer
# Ngrams allows to group words in common pairs or trigrams..etc
# We can use counter to count the objects from collections

class SearchTermForm(FlaskForm):
    searchTerm = StringField('Search Term', validators=[DataRequired()])
    submit = SubmitField()


mystopwords = None


def get_stopwords():
    print('Getting stopwords')
    global mystopwords
    with bp.open_resource('static/stopwords.txt', 'r') as F:
        words = F.readlines()
    from wordcloud import STOPWORDS
    STOPWORDS.add('(b)')
    STOPWORDS.add('(a)')
    STOPWORDS.add('(c)')
    words = [w.strip() for w in words] + list(STOPWORDS)
    words = set(words)
    return words

class HansardSimpleMP:
    def __init__(self, MP):
        self.constituency = MP.constituency
        self.party = MP.party
        self.image = MP.image
        self.full_name = MP.full_name

from numpy import dot
from numpy.linalg import norm
def cosineSimilarity(a, b):
    return dot(a, b)/(norm(a)*norm(b))

class HansardMP:
    def __init__(self, postcode_or_constituency, minLength=25):
        self.info = getMP(postcode_or_constituency)
        self.minLength = minLength
        self._wordcloud_freqs = None
        self._speeches = None
        self.stopwords = get_stopwords()
        self._embeddings = None
        self._topic_model = None
        self._representative_docs = None
        self._sentenceTransformer = HansardSentenceTransformer
        log.info('Completed MP initialisation for %s', self.full_name)
    
    @property
    def sentenceTransformer(self):
        return self._sentenceTransformer


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
        log.info('Preprocessing %d speeches for %s',
                 len(self.speeches), self.full_name)
        text = [x['text'] for x in self.speeches]
        words, bigrams = preprocess_speeches(text, bigrams=True)
        bigrams = list(bigrams)
        log.info('Frequency counting %d words and %d bigrams for %s',
                 len(words), len(bigrams), self.full_name)
        freqs = bigrams_frequency_count(
            words, bigrams, stopwords=self.stopwords)
        self._wordcloud_freqs = freqs

    @property
    def wordcloud_freqs(self):
        if self._wordcloud_freqs is None:
            self.get_wordcloud_freqs()
        return self._wordcloud_freqs

    
    def get_embeddings(self):
        data = preprocess_speeches_for_embeddings(self.speeches, stopwords=self.stopwords, min_length=self.minLength)
        vectors = self.sentenceTransformer.encode([x['text'] for x in data])
        for i in range(len(data)):
            data[i]['vector'] = vectors[i]
        self._embeddings = data
        log.info('Computed emeddings frequencies for %s', self.full_name)

    @property
    def embeddings(self):
        if self._embeddings is None:
            self.get_embeddings()
        return self._embeddings
    
    def find_most_similar(self, sentence):
        log.info('Finding vector for sentence %s', sentence) 
        vector = self.sentenceTransformer.encode([sentence])[0]
        log.info('Found vector sentence')
        log.info(vector)
        scores = [cosineSimilarity(vector, self.embeddings[i]['vector']) for i in range(len(self.embeddings))]
        idxs = np.argsort(scores) [-10:][::-1]
        most_similar = []
        for i in idxs:
            #log.info('score %.3f', scores[i])
            id = self.embeddings[i]['idx']
            text = self.speeches[id]['text']
            t = datetime.fromtimestamp(self.speeches[id]['timestamp'])
            t = t.strftime('%d/%m/%Y')
            most_similar.append({'text' : text,  'date' : t})
            #log.info(text)
        return most_similar






    


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


@bp.route('/', methods=('GET', 'POST'))
def dropdown(selected_constituency=None, MP=None, wordclouddata=None, form=None):
    global constituencies
    log.info('Working')
    display_tab = 'wordcloud'
    if constituencies is None:
        log.info('Requesting constituencies')
        constituencies = getConstituencies()
        log.info('Completed requesting constituencies')
    if not form is None and form.validate_on_submit():
        searchTerm = form.searchTerm.data
        display_tab = 'search'
    else:
        searchTerm = None
    most_similar = None
    if MP is None:
        SimpleMP = None
    else:
        SimpleMP = HansardSimpleMP(MP)
        if not searchTerm is None:
            most_similar = MP.find_most_similar(searchTerm)
    log.info('Rendering template')
    log.info('Using %.2f %% of memory', psutil.virtual_memory().percent)
    return render_template('main/main.html',
                           constituencies=constituencies,
                           selected_constituency=selected_constituency,
                           MP=SimpleMP,
                           wordclouddata=wordclouddata, form=form, most_similar=most_similar, display_tab=display_tab)


@bp.route('/search', methods=('GET', 'POST'))
def search():
    constituency = None
    select = request.form.get('constituency')
    log.info("select=%s", select)
    if select == 'Select constituency':
        return redirect(url_for('main.dropdown'))
    if select is None:
        log.info('Getting MP from session. SELECT IS NONE')
        MP = session["MP"]
        constituency = session["constituency"]
    else:
        constituency = str(select)
        if constituency == session.get("constituency", None):
            log.info('Getting MP from session. CONSTITUENCY UNCHANGED')
            MP = session["MP"]
        else:
            log.info('GETTING MP FROM HANSARD')
            MP = HansardMP(constituency)
            session["MP"] = MP
            session["constituency"] = constituency

    form = SearchTermForm()
    log.info('Getting wordcloud frequencies for %s', MP.full_name)
    freqs = MP.wordcloud_freqs
    log.info('Completed getting wordcloud frequencies for %s', MP.full_name)
    wordclouddata = [{'word': x[0], 'value': x[1]}
                     for x in freqs.most_common(128)]
    wordclouddata.sort(key=lambda x: x['value'], reverse=True)
    return dropdown(selected_constituency=constituency,
                    MP=MP, wordclouddata=wordclouddata, form=form)
