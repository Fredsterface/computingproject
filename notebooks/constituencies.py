import requests
APIkey = 'DN8s9LBm8jMBFZihXEG2gqzx'
def getQuota():
    quotaurl = 'https://www.theyworkforyou.com/api/getQuota'
    response = requests.get(quotaurl, params={'key':APIkey})
    return response.json()['quota']

print(getQuota())

def getConstituencies():
    url = 'https://www.theyworkforyou.com/api/getConstituencies'
    response = requests.get(url, params={'key':APIkey})
    return [x['name'] for x in response.json()]

print(getConstituencies())


