from numpy import dot
from . import BERTopic
from numpy.linalg import norm
from flask import (
    Blueprint, redirect, render_template, request, session, url_for
)
from wtforms.validators import DataRequired
from . import HansardSentenceTransformer
from . import UMAP
from . import ngrams
from datetime import datetime
import numpy as np
import psutil
from wtforms import StringField, SubmitField
from flask_wtf import FlaskForm
from flask import Flask, render_template, request, redirect, url_for
from . wordcloud import preprocess_speeches, preprocess_speeches_for_embeddings, bigrams_frequency_count
from . hansard import getMP, getHansard, getSpeeches
import pandas as pd
from nltk.tokenize import WhitespaceTokenizer as tokenizer
from nltk.stem import WordNetLemmatizer
from collections import Counter
from .hansard import getConstituencies
import logging
log = logging.getLogger('Hansard.main')
log.info('At start of main')

log.info('Importing ngrams')
log.info('Importing bertopic')
log.info('Done importing UMAP')
log.info('Importing sentence transformer')


log.info('Getting Bluprint')
bp = Blueprint('main', __name__, url_prefix='/')
log.info('Got Blueprint')

# load in the NTLK stopwords to remove articles, preposition and other words that are not actionable
# This allows to create individual objects from a bog of words
# from nltk.tokenize import wordpunct_tokenize
# Lemmatizer helps to reduce words to the base formfrom nltk.stem import WordNetLemmatizer
# Ngrams allows to group words in common pairs or trigrams..etc
# We can use counter to count the objects from collections


class SearchTermForm(FlaskForm):
    """
    Form in Search tab using Python Forms (WTF)
    """
    searchTerm = StringField('Search Term', validators=[DataRequired()])
    submit = SubmitField()


mystopwords = None


def get_stopwords():
    """
    Gets the list of stopwords from the stopwords file
    """
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
    """
    Gets the MP data for the webpage (useful stuff)
    """
    def __init__(self, MP):
        self.constituency = MP.constituency
        self.party = MP.party
        self.image = MP.image
        self.full_name = MP.full_name


def cosineSimilarity(a, b):
    """"
    Computes the cosine similarity of two vectors
    """
    return dot(a, b)/(norm(a)*norm(b))

def cosineSimilarityNormalised(a, b):
    """"
    Computes the cosine similarity of two vectors with norm 1
    """
    return dot(a, b)


class HansardMP:
    """
    Gets and stores all the info on the MP
    """
    def __init__(self, postcode_or_constituency, minLength=25):
        """
        Initialises the info about the chosen MP
        """
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
        """
        Returns the function that converts text into vectors
        """
        return self._sentenceTransformer

    @property
    def constituency(self):
        """
        Returns the constituency chosen by the user
        """
        return self.info['constituency']

    @property
    def person_id(self):
        """
        Returns the ID from Hansard for the chosen MP
        """
        return self.info['person_id']

    @property
    def party(self):
        """
        Returns the political party for the chosen MP
        """
        return self.info['party']

    @property
    def image(self):
        """
        Returns the URL of the image of the chosen MP
        """
        return 'https://www.theyworkforyou.com' + self.info['image']

    @property
    def full_name(self):
        """
        Returns the full name of the chosen MP
        """
        return self.info['full_name']

    def get_speeches(self):
        """
        Gets all the speeches for the chosen MP from the Hansard API.
        Sets self._speeches to be a list, and each element of the list, and each entry is a dictionary
        with keys timestamps, and text. 
        Timestamp is a unix timestamp.
        Speeches are presnted in the order of latest speech tp the earliest.
        """
        log.info('Getting speeches for %s', self.full_name)
        self._speeches = getSpeeches(self.person_id, self.minLength)
        log.info('Completed getting speeches for %s', self.full_name)

    @property
    def speeches(self):
        """
       Returns the speeches.
       If they have not yet been retrieved, retrieve them 
        """
        if self._speeches is None:
            self.get_speeches()
        return self._speeches

    def get_wordcloud_freqs(self):
        """
        Computes the wordcloud frequencies.
        Sets self._wordcloud_freqs to be a Counter.
        The Keys are the word/bigrams, and the values are the Counts of each word
        """
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
        """
        Returns the wordcloud frequencies.
        If they have not been retrieved, retrieve them
        """
        if self._wordcloud_freqs is None:
            self.get_wordcloud_freqs()
        return self._wordcloud_freqs

    def get_embeddings(self):
        """
        Computes the embeddings.
        Sets self._embeddings to be a list.
        Each entry of the list is a dictionary with keys
        timestamp - a unix timestamp.
        text - a speech with all the stopwords removed
        vector - a 300 dimensional vector with norm 1.0 that represents the text
        idx - the index of this speech in self._speeches
        """
        data = preprocess_speeches_for_embeddings(
            self.speeches, stopwords=self.stopwords, min_length=self.minLength)
        vectors = self.sentenceTransformer.encode([x['text'] for x in data], normalize_embeddings=True)
        for i in range(len(data)):
            data[i]['vector'] = vectors[i]
        self._embeddings = data
        log.info('Computed emeddings for %s', self.full_name)

    @property
    def embeddings(self):
        """
        Returns the embeddings.
        If they have not been computed, compute them
        """
        if self._embeddings is None:
            self.get_embeddings()
        return self._embeddings

    def find_most_similar(self, sentence):
        """
        The input is a sentence as a text string.
        This function does the following: 
        1. Produces a vector for the sentence
        2. Score the vector against existing embeddings by cosine similarity
        3. Sort by score, to find the most similar 10 vectors
        4. Recover the original most similar speeches

        Returns a list. Each entry of the list is a dictionary with keys
        text - a speech by the MP
        date - the date the MP said the speech in a readable format
        """
        log.info('Finding vector for sentence %s', sentence)
        vector = self.sentenceTransformer.encode([sentence], normalize_embeddings=True)[0]
        log.info('Found vector sentence')
        scores = [cosineSimilarityNormalised(vector, self.embeddings[i]['vector'])
                  for i in range(len(self.embeddings))]
        idxs = np.argsort(scores)[-10:][::-1]
        most_similar = []
        for i in idxs:
            # log.info('score %.3f', scores[i])
            id = self.embeddings[i]['idx']
            text = self.speeches[id]['text']
            t = datetime.fromtimestamp(self.speeches[id]['timestamp'])
            t = t.strftime('%d/%m/%Y')
            most_similar.append({'text': text,  'date': t})
            # log.info(text)
        return most_similar

    def get_topic_model(self):
        """
        Computes a BERTopic topic model
        for example see: https://maartengr.github.io/BERTopic/getting_started/tips_and_tricks/tips_and_tricks.html#speed-up-umap
        """
        # Initiate UMAP
        umap_model = UMAP(n_neighbors=15,
                          n_components=5,
                          min_dist=0.0,
                          metric='cosine',
                          random_state=100)
        # Initiate BERTopic
        self._topic_model = BERTopic(umap_model=umap_model, embedding_model=self.sentenceTransformer,
                                     language="english", calculate_probabilities=True, nr_topics=9, verbose=False)

    @property
    def topic_model(self):
        """
        Returns the topic modelling.
        If they have not been retrieved, get the topic models and return them
        """
        if self._topic_model is None:
            self.get_topic_model()
            self.run_topic_model()
            
        return self._topic_model

    def run_topic_model(self):
        """
        Runs the topic modelling
        Assings a topic to each speech
        """
        text, embeddings = [x['text'] for x in self.embeddings], np.array(
            [x['vector'] for x in self.embeddings])
        log.info('Running topic model for %s on %d extracts',
                 self.full_name, len(text))
        self.topics, self.probabilities = self.topic_model.fit_transform(
            text, embeddings)
        log.info('Completed running topic model %s', self.full_name)

    def get_representative_docs(self):
        """
         Retrieves representative documents for each topic from the topic model.
    
    Returns:
    - tables: List of lists, where each inner list contains the date and text of a representative document 
              for a particular topic.
        """
        tables = []
        log.info('Getting representative docs')
        for i in self.topic_model.get_topics().keys():
            if i == -1:
                continue
            tables.append([])
            reps = self.topic_model.get_representative_docs(i)
            for r in reps:
                j = next(j for j in range(len(self.embeddings))
                         if self.embeddings[j]['text'] == r)
                idx = self.embeddings[j]['idx']
                t = datetime.fromtimestamp(self.speeches[idx]['timestamp'])
                t = t.strftime('%d/%m/%Y')
                tables[-1].append([t, self.speeches[idx]['text'].strip()])
        return tables

    @property
    def representative_docs(self):
        """
        Property that returns the representative documents for each topic from the topic model.
    If the representative documents have not been retrieved yet, it calls the `get_representative_docs()` 
    method to retrieve them and stores them in a private variable for future access.

    Returns:
    - tables: List of lists, where each inner list contains the date and text of a representative document 
              for a particular topic.
        """
        if self._representative_docs is None:
            self._representative_docs = self.get_representative_docs()
        return self._representative_docs


def word_frequency(sentence):
    """
    Computes word frequency, word pairs frequency, and trigrams frequency from a given sentence.

    Args:
    - sentence: List of strings, representing the input sentence.

    Returns:
    - word_freq: Pandas DataFrame, containing the word frequency with columns 'word' and 'frequency',
                 sorted by frequency in descending order.
    - word_pairs: Pandas DataFrame, containing the word pairs frequency with columns 'pairs' and 'frequency',
                  sorted by frequency in descending order.
    - trigrams: Pandas DataFrame, containing the trigrams frequency with columns 'trigrams' and 'frequency',
                sorted by frequency in descending order.
    """
    global mystopwords
    sentence = " ".join(sentence)
    print('starting to tokenize')
    new_tokens = tokenizer().tokenize(sentence)
    print('finished tokenize')
    new_tokens = [t.lower().strip() for t in new_tokens]
    if mystopwords is None:
        mystopwords = get_stopwords()
    S = mystopwords
    new_tokens = [t for t in new_tokens if t not in S]
    new_tokens = [t for t in new_tokens if t.isalpha()]
    lemmatizer = WordNetLemmatizer()
    counted = Counter(new_tokens)
    counted_2 = Counter(ngrams(new_tokens, 2))
    counted_3 = Counter(ngrams(new_tokens, 3))
    word_freq = pd.DataFrame(counted.items(), columns=[
                             'word', 'frequency']).sort_values(by='frequency', ascending=False)
    word_pairs = pd.DataFrame(counted_2.items(), columns=[
                              'pairs', 'frequency']).sort_values(by='frequency', ascending=False)
    trigrams = pd.DataFrame(counted_3.items(), columns=[
                            'trigrams', 'frequency']).sort_values(by='frequency', ascending=False)
    return word_freq, word_pairs, trigrams


constituencies = None

@bp.route('/index')
def index():
    """
    Redirects to the 'main.main' endpoint from the '/index' route.
    """
    return redirect(url_for('main.main'))

@bp.route('/', methods=('GET', 'POST'))
def main(selected_constituency=None, MP=None, wordclouddata=None, form=None):
    """
    Renders the main page with optional parameters for selected_constituency, MP, wordclouddata, and form.
    """
    global constituencies
    log.info('Working')
    display_tab = 'wordcloud'
    if constituencies is None:
        log.info('Requesting constituencies')
        constituencies = getConstituencies()
        log.info('Completed requesting constituencies')
        instructions = True
    else:
        instructions = False
    if not form is None and form.validate_on_submit():
        searchTerm = form.searchTerm.data
        display_tab = 'search'
    else:
        searchTerm = None
    most_similar = None
    if MP is None:
        instructions = True
        representative_docs = None
        topicsData = None
        SimpleMP = None
    else:
        representative_docs = MP.representative_docs
        topicsData = [[{'word': w[0], 'value': 1.0} for w in MP.topic_model.get_topic(
            i)] for i in MP.topic_model.get_topics().keys() if i != -1]
        SimpleMP = HansardSimpleMP(MP)
        selected_constituency = MP.constituency
        instructions = False
        if not searchTerm is None:
            most_similar = MP.find_most_similar(searchTerm)
    if instructions:
        log.info('Rendering initial page with instructions')
    else:
        log.info('Rendering template')
    log.info('Using %.2f %% of memory', psutil.virtual_memory().percent)
    ret = render_template('main/main.html',
                           constituencies=constituencies,
                           selected_constituency=selected_constituency,
                           MP=SimpleMP,
                           wordclouddata=wordclouddata, form=form, most_similar=most_similar, 
                           display_tab=display_tab, topics_data=topicsData, 
                           representative_docs=representative_docs, instructions=instructions)
    log.info('Completed rendering template')
    return ret


@bp.route('/search', methods=('GET', 'POST'))
def search():
    """
    Performs a search based on user-selected constituency, retrieves wordcloud data, and returns the main page.
    """
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
                     for x in freqs.most_common(64)]
    wordclouddata.sort(key=lambda x: x['value'], reverse=True)
    return main(selected_constituency=constituency,
                    MP=MP, wordclouddata=wordclouddata, form=form)
