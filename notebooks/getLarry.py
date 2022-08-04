import json
import postcodes_uk
import requests
APIkey = 'DN8s9LBm8jMBFZihXEG2gqzx'

def getMP(postcode):
    postcode = postcode.upper()
    params = {'key':APIkey}
    if postcodes_uk.validate(postcode):
        params['postcode'] = postcode
    else:
        params['constituency'] = postcode
    
    url = 'https://www.theyworkforyou.com/api/getMP'
    response = requests.get(url, params=params)
    return response.json()


#MP = getMP('Tewkesbury')
#MP = getMP('B54BU') #Postcode for bullring
MP = getMP('BN1 1AA') #Postcode for Brighton Pavillion

print(MP)

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
    return data


data = getHansard(MP['person_id'])
from bs4 import BeautifulSoup
import string
extracts = [BeautifulSoup(x['body'], 'html.parser').text.lower().translate(str.maketrans('', '', string.punctuation)) for x in data['rows']]

#with open('larry.json', 'w') as F:
#    json.dump(extracts, F)
#with open('B54BU.json', 'w') as F:
#    json.dump(extracts, F)
with open('Lucas.json', 'w') as F:
    json.dump(extracts, F)