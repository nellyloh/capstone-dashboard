import pandas as pd
import jellyfish
from fuzzywuzzy import fuzz
import numpy as np
from webscraper_confidence_score import ER_name_matching

def sanction_screening(client):
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
    
    def excel_to_dict(excel_file):
        excel_df = pd.read_excel(excel_file)
        excel_df.value.apply(str)
        before_transformation = dict(zip(excel_df.key, excel_df.value))
        dictionary = {key: [val for val in value.split(',')] for key, value in before_transformation.items()}
        return dictionary

    names_dict = excel_to_dict('names_dict.xlsx') 
    sanction_list_dict = pd.read_csv("cleaned_indiv_sanction_list.csv").to_dict('records')
    
    client_name = client
    split_client_name = split_name_list(client_name)

    for i in range(len(split_client_name)):
        split_client_name[i] = preprocess_name(names_dict, split_client_name[i])        
    stitched_client_name = stitch_name(split_client_name)
    
    for record in sanction_list_dict:
        current_sanc_name = record['name']
        split_sanction_name = split_name_list(current_sanc_name)
        if len(split_client_name) != len(split_sanction_name):
            continue
        for i in range(len(split_sanction_name)):
            split_sanction_name[i] = preprocess_name(names_dict, split_sanction_name[i])
        
        stitched_sanc_name = stitch_name(split_sanction_name)
        
        if abs(len(stitched_client_name) - len(stitched_sanc_name))>3:
            # print(stitched_client_name, stitched_sanc_name)
            continue
        
        try:
            flag = ER_name_matching(client_name, current_sanc_name)
            # print("go")
        except:
            continue
        else:
            if flag is None:
                continue
            if flag > 0:
                return True
    return False    