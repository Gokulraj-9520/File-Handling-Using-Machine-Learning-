import streamlit as st
import pandas as pd
import pickle
import nltk
import os
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import ast
import numpy as np
from cleantext import clean
from pprint import pprint
from sklearn.metrics import accuracy_score, classification_report
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
lemmatizer = WordNetLemmatizer()
nltk.download('wordnet')
nltk.download('stopwords')

st.title("File Identifying Using ML Models")

file=st.file_uploader("Upload Your file", type=["html"])

submit=st.button("Submit")


def preprocess_table(table):
    table_str = table.to_string()
    table_str = re.sub(r'[^a-zA-Z0-9\s]', '', table_str)
    table_str = re.sub(r'\s+', ' ', table_str)
    return table_str

preprocessed_tables = []

if submit:
    files=pd.read_html(file)
    for table in files:
        preprocessed_table=preprocess_table(table)
        preprocessed_tables.append(preprocessed_table)

    #pprint(preprocessed_tables)
    preprocessed_texts=[]
    for values in preprocessed_tables:
        preprocessed_texts.append(values.split())

    #pprint(preprocessed_texts)

    for values in preprocessed_texts:
        for value in values:
            if value.isalpha():
                pass
            else:
                values.remove(value)
    removable_value=['1','2','3','4','5','6','7','8','9','0','NaN']

    def remove_unwanted_values(nested_list, removable_value):
        cleaned_list=[]
        for sublist in nested_list:
            new_sublist=[value for value in sublist if value not in removable_value and not value.isnumeric()]
            cleaned_list.append(new_sublist)
        return cleaned_list
    cleaned_texts=remove_unwanted_values(preprocessed_texts,removable_value)
    #print(cleaned_texts)

    df=pd.DataFrame()
    df['preprocessed_file']=cleaned_texts
    stop_words=set(stopwords.words('english'))
    df['extracted_text'] = df['preprocessed_file'].apply(lambda tokens: [word for word in tokens if word.lower() not in stop_words])

    def remove_duplicates(words_list):
        
        seen = []
        for word in words_list:
            word=word.lower()
            if word not in seen:
                seen.append(word)
        return seen
    
    df['distinct_extracted_text'] = df['extracted_text'].apply(remove_duplicates)

    def get_wordnet_pos(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)
    
    def lemmatize_words(words_list):
        return sorted([lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words_list if len(word) > 2])


    df['extracted_text_lemmatized'] = df['distinct_extracted_text'].apply(lemmatize_words)

    df['extracted_text_lemmatized'] = df['extracted_text_lemmatized'].apply(lambda tokens: ' '.join(tokens))


    df['extracted_text_lemmatized']=df['extracted_text_lemmatized'].str.lower()

    # Function to keep only alphabetic words
    def keep_only_alphabetic(text):
        # Use regex to find all alphabetic words
        return ' '.join(re.findall(r'\b[a-zA-Z]+\b', text))

    # Apply the function to the DataFrame
    df['extracted_text_lemmatized'] = df['extracted_text_lemmatized'].apply(keep_only_alphabetic)

    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    tfidf_matrix = vectorizer.transform(df['extracted_text_lemmatized'])
    df_normalized = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    with open('random_forest.pkl','rb') as f:
        model=pickle.load(f)
    pred=model.predict(df_normalized)
    #print(pred[0])
    if pred[0]==0:
        result="Balance Sheets"
    elif pred[0]==1:
        result="Cash Flow"
    elif pred[0]==2:
        result="Income Statement"
    elif pred[0]==3:
        result='Notes'
    elif pred[0]==4:
        result='Others'
    st.success(f"Your File is {result}")