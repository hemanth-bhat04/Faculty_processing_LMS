import json
import re
import requests

def cleanPhraseFirst(phrase):  ######### NLP FUNCTIONS
    phrase = phrase.strip()
    quote1 = "'"
    startPattern = r'^[-^*"~@#$_?!+=|<>%`&\\/\'\;:,.]+'
    endPattern = r'[-^*"~@#$_?!+=|<>%`&\\/\'\;:,.]+$'
    # startPattern = '^[-|^|*|"|~|@|#|$|_|?|!|+|=|\||<|>|%|`|&|\\|\/|\'|;|:|,|.]+'
    # endPattern = '[-|^|*|"|~|@|#|$|_|?|!|+|=|\||<|>|%|`|&|\\|\/|\'|;|:|,|.]+$'
    remove_pattern = '='
    phrase = re.sub(startPattern, '', phrase).strip()
    phrase = re.sub(endPattern, '', phrase).strip()
    phrase = re.sub(remove_pattern, '', phrase).strip()
    phrase = re.sub(quote1, '', phrase).strip()
    phrase = phrase.replace('"', ' ').lower()
    phrase = phrase.replace("-", " ").lower()
    phrase = " ".join(phrase.split())  # remove extra spaces
    if phrase.endswith("'") or phrase.endswith(".") or phrase.endswith("'"):
        phrase = phrase[:-1]
    # print(phrase)
    return phrase


import json
import requests
from nlp_keywords import cleanPhraseFirst

def get_weighted_queries(text, y, subject, level):
    headers = {'Content-type': 'application/json'}
    sections = [{'end_index': y, 'start_index': 0, 'sectionName': '', 'level': [2]}]

    payload = json.dumps({
        'url': 'test',
        'section_title': [''],
        'section_para': text,
        'complete_sections': sections,
        'offlineSubject': subject,
        'level': level,
        'is_nlp_server': True
    })

    try:
        # Send the request to the NLP server
        response = requests.post('http://164.52.192.242:8001/search-nlp-keywords/', data=payload, headers=headers, timeout=100000)
        response.raise_for_status()  # Check if request was successful
        response_data = response.json()

        # Debug: Print the response to check its structure
        print("Response from NLP Server:", json.dumps(response_data, indent=2))

        # Check for error message in the response
        error_message = response_data.get('nlp_response_output', {}).get('ErrorMessage', None)
        if error_message:
            print(f"Error from NLP server: {error_message}")
            return None, [], [], []  # Return empty results if there's an error

        # Process the transcript directly for keywords
        nlp_response_output = response_data.get('nlp_response_output', {})

        # Safely access the data
        weighted_query = nlp_response_output.get('weighted_query', None)
        phrasescorelist = nlp_response_output.get('phraseScorelist', [])
        phrasescorelist = [(cleanPhraseFirst(phrase[0]), phrase[1]) for phrase in phrasescorelist]

        entity_weight = nlp_response_output.get('entity_weight', [])
        entity_weight = [(cleanPhraseFirst(entity[0]), entity[1]) for entity in entity_weight]

        plus_words = nlp_response_output.get('plus_words', [])

        # Return the data
        return weighted_query, phrasescorelist, entity_weight, plus_words

    except requests.exceptions.RequestException as e:
        print(f"Error with the request: {e}")
        return None, [], [], []




