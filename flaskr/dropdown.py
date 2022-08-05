import functools

from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)

bp = Blueprint('dropdown', __name__, url_prefix='/dropdown')

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
    print(MP['image'])
    imageurl = 'https://www.theyworkforyou.com'+MP['image']
    print(imageurl)
    MP['image'] = imageurl

    wordclouddata = [
        {'word' : 'fish', 'value' : 20},
        {'word' : 'trout', 'value' : 25},
        {'word' : 'salmon', 'value' : 15},
        {'word' : 'bass', 'value' : 30},
        {'word' : 'swordfish', 'value' : 27},
        

    ]

    return dropdown(selected_constituency=constituency,
    MP=MP, wordclouddata=wordclouddata)



