# Sentiment Analysis Packages
import pandas as pd
import numpy as np
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import re
import string
import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
nltk.download('stopwords')


# Sentiment Analysis
def sentiment_model(test_query):
    
    if len(test_query) == 0:
        sys.exit("No Articles found.")
        
    else:

        # load json and create model
        from keras.models import model_from_json
        json_file = open('bi_lstm_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        reconstructed_model_bi_lstm = model_from_json(loaded_model_json)

        # load weights into new model
        reconstructed_model_bi_lstm.load_weights("bi_lstm_model.h5")

        # load json and create model
        from keras.models import model_from_json
        json_file = open('lstm_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        reconstructed_model_lstm = model_from_json(loaded_model_json)

        url = r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)
        (?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([
          ^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))'''

        tokenizer = RegexpTokenizer(r'\w+')

        def clean_data(temp):
            temp = temp.map(lambda x:str(x).lower()) 
            # removing emails
            temp = temp.map(lambda x:re.sub(r"\b[^\s]+@[^\s]+[.][^\s]+\b", "", x)) 
            # removing url
            temp = temp.map(lambda x:re.sub(url, "", x)) 
            # removing numbers
            temp = temp.map(lambda x:re.sub(r'[^a-zA-z.,!?/:;\"\'\s]', "", x)) 
            # removing white space
            temp = temp.map(lambda x:re.sub(r'^\s*|\s\s*', ' ', x).strip()) 
            # removing punctuations
            temp = temp.map(lambda x:''.join([c for c in x if c not in string.punctuation])) 
            # removing special characters
            temp = temp.map(lambda x:re.sub(r'[^a-zA-z0-9.,!?/:;\"\'\s]', '', x)) 
            # unicode
            temp = temp.map(lambda x:unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8', 'ignore')) 
            # tokenising text for cleaning
            temp = temp.map(lambda x:tokenizer.tokenize(x)) 
            # removing stop words
            temp = temp.map(lambda x:[i for i in x if i not in stopwords.words('english')]) 
            temp = temp.map(lambda x:' '.join(x))
            return temp

        test_query['body'] = test_query['text']
        test_query.text = clean_data(test_query.text)

        # Data Preprocessing for model ingestion
        maxlen = 50
        embedding_dim = 100

        X = test_query.text.values
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(test_query.text.values)
        X = tokenizer.texts_to_sequences(X)
        vocab_size = len(tokenizer.word_index) + 1
        test_input = pad_sequences(X, padding='pre', maxlen=maxlen)

        # Predicting output

        # LSTM
        test_lstm = reconstructed_model_lstm.predict(test_input)
        test_classes_lstm = np.argmax(test_lstm,axis=1)
        test_query['prediction_lstm'] = test_classes_lstm

        # BI-LSTM
        test_bi_lstm = reconstructed_model_bi_lstm.predict(test_input)
        test_classes_bi_lstm = np.argmax(test_bi_lstm,axis=1)
        test_query['prediction_bi_lstm'] = test_classes_bi_lstm

        test_query.loc[test_query['prediction_lstm'] == 0, 'sentiment_lstm'] = 'Financial Crime'
        test_query.loc[test_query['prediction_lstm'] == 1, 'sentiment_lstm'] = 'Serious Crime'
        test_query.loc[test_query['prediction_lstm'] == 2, 'sentiment_lstm'] = 'General News (Positive)'
        test_query.loc[test_query['prediction_lstm'] == 3, 'sentiment_lstm'] = 'General News (Neutral)'

        test_query.loc[test_query['prediction_bi_lstm'] == 0, 'sentiment_bi_lstm'] = 'Financial Crime'
        test_query.loc[test_query['prediction_bi_lstm'] == 1, 'sentiment_bi_lstm'] = 'Serious Crime'
        test_query.loc[test_query['prediction_bi_lstm'] == 2, 'sentiment_bi_lstm'] = 'General News (Positive)'
        test_query.loc[test_query['prediction_bi_lstm'] == 3, 'sentiment_bi_lstm'] = 'General News (Neutral)'
        
        test_query = test_query[['title', 'time', 'year_of_birth', 'description', 'link', 'body',
                                       'names_list', 'confidence_score', 'sentiment_lstm', 'sentiment_bi_lstm']]
    
        return pd.DataFrame(test_query)

