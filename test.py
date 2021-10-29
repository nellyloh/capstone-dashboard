import jellyfish
from fuzzywuzzy import fuzz
import pandas as pd
import numpy as np
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import requests
from newspaper import Article
# import nltk
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
import dateutil.parser as parser
from geopy.geocoders import Nominatim
import pycountry
import time
from datetime import date
from dateutil.relativedelta import relativedelta
import pinyin
import sys

from geopy.exc import GeocoderServiceError

# Sentiment Analysis Packages
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import re
import string
import unicodedata


# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import RegexpTokenizer
# nltk.download('stopwords')

nlp = en_core_web_sm.load()

import warnings
warnings.filterwarnings('ignore')


def preprocess_df_to_dict(df):
    def get_year(date):
        try:
            parser_obj = parser.parse(str(date))
            return parser_obj.year
        except:
            return None

    def get_month(date):
        if len(str(date))>4:
            try:
                return parser.parse(str(date)).month
            except:
                return None
        else:
            return None
            
    def get_day(date):
        if len(str(date))>4:
            try:
                return parser.parse(str(date)).day
            except:
                return None
        else:
            return None
    
    def isEnglish(s):
        try:
            s.encode(encoding='utf-8').decode('ascii')
        except UnicodeDecodeError:
            return False
        else:
            return True    
    
    df_dict_list = df.to_dict('records')
    cleaned_dict_list = []
    for record in df_dict_list:
        
        alias = record['Alias name']
        if alias is not None:
            alias_is_english = isEnglish(alias)
            if alias_is_english is False:
                try:
                    alias = pinyin.get(alias, format='strip', delimiter=' ')
                except:
                    alias = None
        current_record = {
            'name': record['Name to be screened'],
            'alias' : alias,
            'year_of_birth': get_year(record['Date of birth']),
            'month_of_birth': get_month(record['Date of birth']),
            'day_of_birth': get_day(record['Date of birth']),
            'gender': record['Gender'],
            'nationality': record['Nationality'],
            ### delete these later on, for testing only###
            'type_of_error': record['Type of variation (if any)'],
            'actual_name': record['Actual name'],
        }
        cleaned_dict_list.append(current_record)
    return cleaned_dict_list

def ER_name_matching(name1, name2):
    def split_name_list(name):
        name = name.lower()
        output = name.split(" ")
        return output

    def preprocess_name(names_dict, word):
        for key, value in names_dict.items():
            if word in value:
                return key
        else:
            return word

    def stitch_name(list1):
        output = ''
        for x in range(len(list1)):
            if x==0:
                output += list1[x]
            else:
                output += ' ' + list1[x]
        return output

    def phonetic_comparison(list1, list2):
        meta_list1 = []
        meta_list2 = []
        nysiis_list1 = []
        nysiis_list2 = []
        for name_1 in list1:
            meta_list1.append(jellyfish.metaphone(name_1))
            nysiis_list1.append(jellyfish.nysiis(name_1))
        for name_2 in list2:
            meta_list2.append(jellyfish.metaphone(name_2))
            nysiis_list2.append(jellyfish.nysiis(name_2))
        if (set(meta_list1) == set(meta_list2)) or (set(nysiis_list1) == set(nysiis_list2)):
            return True
        else:
            return False
    
    def excel_to_dict(excel_file):
        excel_df = pd.read_excel(excel_file)
        excel_df.value.apply(str)
        before_transformation = dict(zip(excel_df.key, excel_df.value))
        dictionary = {key: [val for val in value.split(',')] for key, value in before_transformation.items()}
        return dictionary
            
    names_dict = excel_to_dict('names_dict.xlsx') 
    
    # START #
    ### Change this if needed ###
    threshold = 89
    #############################
    
    split_list_1 = split_name_list(name1)
    split_list_2 = split_name_list(name2) 
 
    
    for i in range(len(split_list_1)):
        split_list_1[i] = preprocess_name(names_dict, split_list_1[i])        
    for i in range(len(split_list_2)):
        split_list_2[i] = preprocess_name(names_dict, split_list_2[i])
    
    stitched_name1 = stitch_name(split_list_1)
    stitched_name2 = stitch_name(split_list_2)
    
    # 1st layer of testing: Token Sort Ratio with threshold
    score1 = fuzz.token_sort_ratio(stitched_name1, stitched_name2)
    if score1 >= threshold:
        # score_list.append(score1)
        return score1
        # do something
# 4) 2nd layer of testing - Metaphone and NYSIIS phonetic encoding - DONE
    else: 
        matched_phonetic = phonetic_comparison(split_list_1, split_list_2)
        if matched_phonetic:
            return threshold # assumption that phonetic match will give threshold score
        else: 
            return None
    
    try:
        return score1
    except:
        pass

# hlpr func: get country by cities, states name
def get_country(gpe):
    geolocator = Nominatim(user_agent = "geoapiExercises")
    location = geolocator.geocode(gpe)
    if location:
        loc_lst = location.address.split(',')
        return loc_lst[-1]
    return None

# hlpr func: return a list of countries names
def countries():
    return list(map(lambda x: x.name, list(pycountry.countries)))

# hlpr func: return True if name countains country name
def contain_country(word, ctry_lst):
    for ctry in ctry_lst:
        if ctry.lower() in word.lower():
            return True
    return False

# hlpr func: extract entities with tag 'GPE', 'ORG', 'NORP'
def search_target_ent(tags):
    country_lst = countries()
    tag_lst = []
    for i in range(len(tags)):
        if tags[i][1] == 'GPE' or tags[i][1] == 'ORG' or tags[i][1] == 'NORP':
            if contain_country(tags[i][0], country_lst):
                tag_lst.append(tags[i])
    return tag_lst

# hlpr func: return the odd of the person's nationality in the article is nat
def calc_odd_nationality(nat,lst):
    try:
        result = []
        for tag in lst:
            if tag[0] is not None and nat is not None:
                if nat.lower() in tag[0].lower():
                    result.append(tag)
                    continue
            try:
                if tag[1] == 'GPE' and (get_country(tag[0]) is not None and nat is not None):
                    if get_country(tag[0]).lower() == nat.lower():
                        result.append(tag)
            except GeocoderServiceError as e:
                pass
        prob = 1 if ((len(lst) - len(result)) == 0 and len(result) > 0) else (len(result) / (len(lst) - len(result)))
        prob = 1 if prob > 1 else prob
        return prob
    except TypeError as e:
        pass

# hlpr func: return True if name fuzzy matching score > 80
def is_target(name, article_name):
    return fuzz.partial_ratio(name, article_name) > 80

# the main function for nationality matching
# return odd if target tags found else return 0
def nationality_matching(tags, nationality, person):
    
    if nationality is None:
        return None
    
    result = []
    try:
        for i in range(len(tags)):
            #if second item is a name
            if tags[i][1] == 'PERSON':
                
                # check if is target
                if is_target(person, tags[i][0]):
                    search = search_target_ent(tags)
                
                    if len(search) != 0:
                        return calc_odd_nationality(nationality, search)
        return 0
    except IndexError as e:
        pass

# hlpr func: parse text to tags
def parse(text):
        #try:     
        doc = nlp(text)
        tags = [[X.text, X.label_] for X in doc.ents]
        labels = [x.label_ for x in doc.ents]
        items = [x.text for x in doc.ents]

        return tags

# hlpr func: return True if token is a name and subject
def is_name_subj(token):
    return (token.dep_ =='nsubj' or token.dep_ == 'nsubjpass')  and token.pos_ == 'PROPN'

def is_part_of_name(token):
    return (token.dep_ =='nsubj' or token.dep_ =='compound' or token.dep_ == 'nsubjpass') \
        and token.pos_ == 'PROPN'

# hlpr func: return True if the token is a determiner: his, her, hers
def is_det(token):
    return token.pos_ == 'DET' and (token.dep_ == 'poss' or token.dep_ == 'attr')

# hlpr func: return True if the token is a pronoun: he, she, herself, himself
def is_pron(token):
    return token.pos_ == 'PRON' and \
        (token.dep_ == 'nsubj' or token.dep_ == 'nsubjpass' or token.dep_ == 'pobj' or token.dep_ == 'dobj')

# hlpr func: return True if the gender noun is referring to target person
def refer_target(gender, noun, name, text):
    m = ['man', 'boy', 'guy']
    f = ['woman', 'lady', 'girl']
    
    if is_target(name, text):
        return (gender == 'male' and noun in m) or (gender == 'female' and noun in f)
    return 0

# hlpr func: return True if gender noun is follwed by 'is, was, as or comma'
def gender_noun(t1, t2):
    gender_nouns = ['man', 'boy', 'guy', 'woman', 'lady', 'girl']
    verbs = ['was', 'is', 'as', ',']
    return (t1 in gender_nouns) and (t2 in verbs)

# hlpr func: return the probability of the gender in article to the true gender
def calc_prob_gender(pron_lst, gender):
    male_pron = ['he', 'his', 'himself', 'him']
    female_pron = ['she', 'her', 'herself', 'hers']
    n_target = 0
    gdr_pron = []
    
    if gender.lower() == 'male':
        gdr_pron = male_pron
    else:
        gdr_pron = female_pron
        
    for pron in pron_lst:
        if pron in gdr_pron:
            n_target += 1
    return n_target / len(pron_lst) if len(pron_lst) else 0

# the main function in gender matching
def gender_matching(text, gender, name):
    
    if gender is None:
        return None
    
    try:
        pron_lst = ['he', 'his', 'himself', 'him', 'she', 'her', 'herself', 'hers']
        name_str = ''
        target_name = name.replace(" ", "")
        target_found = False
        res_lst = []
        
        # text tagging
        doc = nlp(text)
        
        i = 0
        while i < len(doc):

            # catch text like '...woman is/was/as/, xxx...'
            if gender_noun(doc[i].text, doc[i + 1].text):
                if refer_target(gender.lower(), doc[i].text, name, doc[i + 2].text):
                    return (1)

            # search for target name of subject form
            if is_name_subj(doc[i]):
                end_name = i
                start_name = i
                while is_part_of_name(doc[start_name]):
                    start_name -= 1
                start_name += 1
                while start_name <= end_name:
                    name_str += doc[start_name].text
                    start_name += 1

            if is_target(name_str, target_name):
                target_found = True
            else:
                name_str = ''
                target_found = False
          
            # if target name is found, search for pronouns, break if another name is found
            while target_found:
                i+=1
                if gender_noun(doc[i].text, doc[i + 1].text):
                    if refer_target(gender.lower(), doc[i].text, name, doc[i + 2].text):
                        return (1)
                if is_name_subj(doc[i]):
                    target_found = False
                    name_str = ''
                    break
                if is_det(doc[i]) or is_pron(doc[i]):
                    if (doc[i].text).lower() in pron_lst:
                        res_lst.append((doc[i].text).lower())
                        break

            i += 1
    except IndexError as e:
        pass
    return calc_prob_gender(res_lst, gender)

useless_dates = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday','yesterday','today']

#index is index of person
def forward_searcher(index,tags):
    for i in range(index,len(tags)):
        if tags[i][1] == 'DATE' and tags[i][0] not in useless_dates:
            return tags[i]
    return [None,None]

def backward_searcher(index,tags):
    i = index
    while i >= 0:
        if tags[i][1] == 'DATE' and tags[i][0] not in useless_dates:
            return tags[i]
        else:
            i -=1

def detect_age(age,lst):
    try:
        if lst[1] is not None and lst[2] is not None:
            date1 = lst[1][0]
            date2 = lst[2][0]
            if (str(age) in date1) or (str(age) in date2):
                return True
        else:

            if lst[1] == None:
                if str(age) in lst[2][0]:
                    return True

            if lst[2] == None:
                if str(age) in lst[1][0]:
                    return True
    except TypeError as e:
        pass
    
    
def confirm_age(lst,age,threshold):
    iterating_lst = []
    plus = 1
    minus = -1
    for i in range(threshold):
        iterating_lst.append(age+plus)
        plus += 1
    for i in range(threshold):
        iterating_lst.append(age+minus)
        minus -=1 
    iterating_lst.append(age)
    
    for j in iterating_lst:
        if str(j) in lst[1][0]:
            return 1
    return 0

def age_matching(name_dict,tags,age):
    '''
    tags: parse(text)
    age: desired age to check
    '''
    if age is None:
        return None
    
    for tag in tags:
        #if tag[1] == 'DATE':
            #print(tag)
        if str(age) in tag[0] or str(age+1) in tag[0] or str(age-1) in tag[0]:
            return 1
    result = []
    try:
        for i in range(len(tags)):
            #if second item is a name

            
            if tags[i][1] == 'PERSON':
                if tags[i][0] in name_dict:

                    forward_age = forward_searcher(i,tags)
                    backwards_age = backward_searcher(i,tags)
                    new_list = [tags[i],forward_age,backwards_age]
                    #new_list = [tags[i-1],tags[i],tags[i+1]]
                    #print(new_list)

                    if detect_age(age,new_list) and tags[i][0] in name_list:

                        #print(new_list)
                        #result += new_list

                        if str(age) in new_list[1][0]:
                            #print('****************')
                            #print([tags[i], new_list[1]])
                            return(confirm_age([tags[i],new_list[1]],age,3))


                        elif str(age) in new_list[2][0]:
                            #print('****************')
                            #print([tags[i],new_list[2]])
                            return(confirm_age([tags[i],new_list[2]],age,3))
                        
        return 0
    except IndexError as e:
        pass
    
def entity_recognition_scoring_each_article(input_info, text, names_list):
    output = []
    input_name = input_info['name']


    article_names_list = names_list.most_common() 
    matched = False

    for each_name, each_count in article_names_list: ## as of now checking all names within the article, should we limit to e.g. top 3/5?
        if len(each_name.split()) == 1 and each_name in input_name:
            score = 100 ## if surname matches, default match score 100 
        else: 
            try: 
                score = ER_name_matching(input_name, each_name)
            except ValueError as e:
                pass
        if score is not None:
            matched = True
        if matched:
            break
    conf_score = 0
    if matched:
        name_score = score
        nationality_score = nationality_matching(parse(text), input_info['nationality'], input_info['name'])
        gender_score = gender_matching(text, input_info['gender'], input_info['name'])
        age_score = age_matching(names_list,parse(text),input_info['year_of_birth'])
        
        denom = 0.9071
        if nationality_score is not None:
            denom += 0.049973
        if gender_score is not None:
            denom += 0.030293
        if age_score is not None:
            denom += 0.012634

        conf_score = ((0.9071 / denom) * (name_score/100))

        if nationality_score is not None:
            conf_score += ((0.049973 / denom) * nationality_score)
        if gender_score is not None:
            conf_score += ((0.030293 / denom) * gender_score)
        if age_score is not None:
            conf_score += ((0.012634 / denom) * age_score)
                           
    return conf_score
    
# Main Function
def search_articles_on_individual(individual_dict, no_of_articles=30):
    def generate_link(person_dict, attributes_used = ['name'], keywords=['crimes', 'sentenced']):
        link_start = "https://www.google.com/search?q="
        link_end = "&sxsrf=ALeKk01K1bOuJFHjy4HBARo1cRpUYakYPg:1629640327633&source=lnms&tbm=nws&sa=X&sqi=2&ved=2ahUKEwiu29um48TyAhWGqpUCHYuoAlcQ_AUoAnoECAEQBA&biw=1441&bih=718&dpr=2" 
        link_query = ""

        for attributes in attributes_used:
            temp_attr = person_dict[attributes]
            if temp_attr is not None:
                temp_attr = str(temp_attr)
                link_query += temp_attr.replace(' ', '+') + '+'       
                
        links = []
        for keyword in keywords:
            temp_search_link = link_start + link_query + keyword + link_end + "&num=" + str(no_of_articles)
            links.append(temp_search_link)
        return links
    
    def article_extraction(link):
        article = Article(link)
        article.download()
        try:
            article.parse()
        except:
            pass
        return article.text

    def parse(text):
        #try:     
        doc = nlp(text)
        tags = [[X.text, X.label_] for X in doc.ents]
        labels = [x.label_ for x in doc.ents]
        items = [x.text for x in doc.ents]

        return tags

    def find_names(tags):
        names = []
        for tag in tags:
            if tag[1] == 'PERSON':
                names.append(tag[0])
        return names
    
    def time_to_months(time):
        if 'weeks' in time:
            return 0
        else:
            return int(time.split(' month')[0])

    search_links = generate_link(individual_dict)
    
    unique_links_checker = []
    
    output = []
    for x in search_links:
        req = Request(x, headers = {'User-Agent': 'Mozilla/5.0'})

        webpage = urlopen(req).read()

        with requests.Session() as c:
            soup = BeautifulSoup(webpage, 'html5lib')
            #print(soup)
            for item in soup.find_all('div', attrs = {'class': "ZINbbc xpd O9g5cc uUPGi"}):
                current_dict = {}
                raw_link = (item.find('a', href = True)['href'])
                try:
                    link = (raw_link.split("/url?q=")[1]).split('&sa=U&')[0]
                except IndexError as e1:
                    continue
                if link not in unique_links_checker and item:
                    unique_links_checker.append(link)
                    title = item.find('div',attrs = {'class': 'BNeawe vvjwJb AP7Wnd'})
                    if title == None:
                        continue
                    title = title.get_text()
                    description  = (item.find('div',attrs = {'class': 'BNeawe s3v9rd AP7Wnd'}).get_text())
                    time = description.split(" · ")[0]
                    #print(description)
                    descript = description.split(" · ")[1]
                    
                    # create names_list
                    parsed_description = parse(description)
                    names_in_description = find_names(parsed_description) 
                    parsed_text = parse(article_extraction(link))
                    names_in_text = find_names(parsed_text)
                    names_list = Counter(names_in_description + names_in_text)
                    
                    # extract text
                    text = article_extraction(link)

                    # compute confidence score before accepting the article
                    conf_score = entity_recognition_scoring_each_article(individual_dict, text, names_list)
                    
                    # this is the new part 0.9071
                    overall_threshold = 0.8
                    
                    nationality = individual_dict['nationality']
                    gender = individual_dict['gender']
                    year_of_birth = individual_dict['year_of_birth']
                    
                    if nationality is not None:
                        overall_threshold += 0.049973
                    if gender is not None:
                        overall_threshold += 0.030293
                    if year_of_birth is not None:
                        overall_threshold += 0.012634
                  

                    if conf_score < overall_threshold:
                        continue
            
                    current_dict['title'] = title
                    current_dict['time'] = time
                    try:
                        current_dict['year_of_birth'] = (date.today() - relativedelta(months = time_to_months(time))).year - individual_dict['year_of_birth']
                    except TypeError as e1:
                        current_dict['year_of_birth'] = 0
                    except ValueError as e2:
                        current_dict['year_of_birth'] = 0
                    current_dict['description'] = descript
                    current_dict['link'] = link
                    current_dict['text'] = text
                    current_dict['confidence_score'] = conf_score
                    
                    output.append(current_dict)
                else:
                    pass
    return output

def demo(name,gender,nationality):
    # Web Scraping
    test_record_1 = {'name': name, 'alias': None, 'year_of_birth': None, 'month_of_birth': None, 'day_of_birth': None, 'gender': gender, 'nationality': nationality, 'type_of_error': '-', 'actual_name': name}
    
    print('Test input: \n')
    print('Name: ' + str(test_record_1['actual_name']))
    print('Alias: ' + str(test_record_1['alias']))
    print('Year of Birth: ' + str(test_record_1['year_of_birth']))
    print('Month of Birth: ' + str(test_record_1['month_of_birth']))
    print('Day of Birth: ' + str(test_record_1['day_of_birth']))
    print('Gender: ' + str(test_record_1['gender']))
    print('Nationality: ' + str(test_record_1['nationality']))

    print('\n')

    test_query = search_articles_on_individual(test_record_1, 3)
    test_query = pd.DataFrame(test_query)

    return test_query
