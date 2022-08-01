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

@bp.route('/', methods=('GET', 'POST'))
def dropdown():
    constituencies = getConstituencies()
    return render_template('dropdown/dropdown.html', 
    constituencies=constituencies)




