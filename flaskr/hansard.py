#This file contains all the functions to interact with the Hansard API

import requests
import functools
from bs4 import BeautifulSoup
import postcodes_uk
from tqdm import tqdm
import logging

APIkey = 'DN8s9LBm8jMBFZihXEG2gqzx'

log = logging.getLogger('Hansard.hansard')

def getQuota():
    url = 'https://www.theyworkforyou.com/api/getQuota'
    response = requests.get(url, params={'key': APIkey})
    return response.json()

def getConstituencies():
    log.info('Getting constituencies')
    #log.info(getQuota())
    url = 'https://www.theyworkforyou.com/api/getConstituencies'
    log.info('Created constituencies url')
    response = requests.get(url, params={'key': APIkey})
    log.info('Received response')
    return [x['name'] for x in response.json()]

def is_valid_postcode(postcode):
    """
    This function takes one argument and returns True or False.
    
    Args:
    postcode (str): The postcode
    
    Returns:
    bool: True if a valid postcode, otherwise returns False.
    """
    #making sure the postcode is in uppercase, required for validation
    postcode = postcode.upper()
    # using the postcodes_uk library to validate the postcode
    ret = postcodes_uk.validate(postcode)
    return ret

def getMP(postcode_or_constituency):
    """
    This function takes one argument, either the postcode or the constituency name, and returns the MP information from Hansard.
    
    Args:
    postcode_or_constituency (str): Postcode or a Constituency
    
    Returns:
    dict: The MP information from Hansard
    """
    # checks to see in the input is a valid postcode
    params = {'key':APIkey}
    if is_valid_postcode(postcode_or_constituency):
        params['postcode'] = postcode_or_constituency
    else:
        params['constituency'] = postcode_or_constituency
    
    url = 'https://www.theyworkforyou.com/api/getMP'
    response = requests.get(url, params=params)
    return response.json()

def getMPExtraInfo(person_id):
    """
    This function takes one argument and returns the MP's person ID from Hansard.
    
    Args:
    personID (int): 
    
    Returns:
    dict: Get extra MP information
    """
    params = {'key':APIkey}
    params['id'] = person_id
    
    url = 'https://www.theyworkforyou.com/api/getMPInfo'
    response = requests.get(url, params=params)
    return response.json()


functools.lru_cache(maxsize = 128)
def getHansard(person_id):
    """
    This function takes one argument and returns the MP's person ID from Hansard.
    
    Args:
    personID (int): gets the relevant data from Hansard
    
    Returns:
    dict: The information from Hansard about the person_id
    """
    params = {'key':APIkey}
    params['person'] = person_id
    params['num'] = 512
    url = 'https://www.theyworkforyou.com/api/getHansard'
    response = requests.get(url, params=params)
    data = response.json()
    params['page'] = 1
    # loops until we have all the pages
    with tqdm(total=data['info']['total_results']) as pbar:
        pbar.update(len(data['rows']))
        while True:
            params['page'] += 1
            response = requests.get(url, params=params)
            data0 = response.json()
            pbar.update(len(data0['rows']))
            if len(data0['rows']) == 0:
                break
            data['rows'].extend(data0['rows'])
    return data



def deduplicate(lst):
    deduplicated = []
    for item in lst:
        if item not in deduplicated:
            deduplicated.append(item)
    return deduplicated

import time
def make_timestamp(_date, _time):
    try:
        if not _time is None:
            return time.mktime(time.strptime("%s %s" % (_date, _time), "%Y-%m-%d %H:%M:%S"))
        else:
            return time.mktime(time.strptime("%s" % (_date), "%Y-%m-%d"))
    except:
        if not _time is None:
            return make_timestamp(_date, None)
        raise(ValueError, 'Date and time to not make sense. %s %s\n' % _date, _time)

def getSpeeches(person_id, minLength=25):
    """
    This function takes one argument and returns the MP's person ID from Hansard.
    
    Args:
    person_id (int): gets the relevant data from Hansard
    
    Returns:
    list: Each list entry is one speech given by the MP and a timestamp
    """
    data = getHansard(person_id)
    speeches = []
    for x in data['rows']:
        body = x['body']
        text = BeautifulSoup(body, "html.parser").text
        if len(text.split()) < minLength:
            continue
        if len(speeches) > 0 and text == speeches[-1]['text']:
            continue
        speeches.append({'timestamp' : make_timestamp(x['hdate'], x['htime']), 'text' : text})
    return deduplicate(speeches)