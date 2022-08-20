import functools

from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)

bp = Blueprint('dropdown', __name__, url_prefix='/dropdown')

from bs4 import BeautifulSoup
import string

import requests
APIkey = 'DN8s9LBm8jMBFZihXEG2gqzx'

def getConstituencies():
    url = 'https://www.theyworkforyou.com/api/getConstituencies'
    response = requests.get(url, params={'key':APIkey})
    return [x['name'] for x in response.json()]

def getMP(constituency):
    params = {'key':APIkey}
    params['constituency'] = constituency 
    url = 'https://www.theyworkforyou.com/api/getMP'
    response = requests.get(url, params=params)
    return response.json()

def getHansard(personID):
    params = {'key':APIkey}
    params['person'] = personID
    params['num'] = 512
    url = 'https://www.theyworkforyou.com/api/getHansard'
    response = requests.get(url, params=params)
    data = response.json()
    params['page'] = 1
    print('Getting %d results' % data['info']['total_results'])
    while True:
        params['page'] += 1
        response = requests.get(url, params=params)
        data0 = response.json()
        if len(data0['rows']) == 0:
            break
        data['rows'].extend(data0['rows'])
        print('%d : %d' % (len(data['rows']), data['info']['total_results']))
    extracts = [BeautifulSoup(x['body'], 'html.parser').text.lower().translate(str.maketrans('', '', string.punctuation)) for x in data['rows']]

    return extracts


import pandas as pd
#load in the NTLK stopwords to remove articles, preposition and other words that are not actionable
from nltk.corpus import stopwords
# This allows to create individual objects from a bog of words
# from nltk.tokenize import wordpunct_tokenize
from nltk.tokenize import WhitespaceTokenizer as tokenizer
from nltk.stem import WordNetLemmatizer
# Lemmatizer helps to reduce words to the base formfrom nltk.stem import WordNetLemmatizer
# Ngrams allows to group words in common pairs or trigrams..etc
from nltk import ngrams
# We can use counter to count the objects from collections
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

constituencies = None

@bp.route('/', methods=('GET', 'POST'))
def dropdown(selected_constituency=None, MP=None, wordclouddata=None):
    global constituencies
    if constituencies is None:
        print('Requesting constituencies')
        constituencies = getConstituencies()
    return render_template('dropdown/dropdown.html', 
    constituencies=constituencies,
     selected_constituency=selected_constituency,
     MP=MP,
     wordclouddata=wordclouddata)

@bp.route('/search', methods=('GET', 'POST'))
def search():
    select = request.form.get('constituency')
    if select=='Select constituency':
        return redirect(url_for('dropdown.dropdown'))


    constituency = str(select)
    MP = getMP(constituency)
    print('getting extracts')
    extracts = getHansard(MP['person_id'])
    print('finished getting extracts')
    print(MP['image'])
    imageurl = 'https://www.theyworkforyou.com'+MP['image']
    print(imageurl)
    MP['image'] = imageurl
    word_freq,word_pairs,trigrams = word_frequency(extracts)
    print(word_freq)
    print(word_pairs)
    word_freq_array = [{'word': word_freq.iloc[i].word, 'value': word_freq.iloc[i].frequency} for i in range(128)]
    word_pairs_array = [{'word': ' '.join(word_pairs.iloc[i].pairs), 'value': word_pairs.iloc[i].frequency} for i in range(128)]
    trigrams_array = [{'word': ' '.join(trigrams.iloc[i].trigrams), 'value': trigrams.iloc[i].frequency} for i in range(128)]

    wordclouddata = word_freq_array + word_pairs_array + trigrams_array
    wordclouddata.sort(key=lambda x : x['value'], reverse=True)
    wordclouddata = wordclouddata[:128]
    return dropdown(selected_constituency=constituency,
    MP=MP, wordclouddata=wordclouddata)


