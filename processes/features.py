import pandas as pd
import pprint as pp
import textstat as textstat
from lexicalrichness import LexicalRichness
from nltk import tokenize
import csv
import numpy as np
import math
import re
import sys
import random

def unique_words(row):
    text = row['reviewText']
    lex = LexicalRichness(text)
    return lex.terms

def flesch_kincaid(row):
    text = row['reviewText']
    words = max(1, textstat.lexicon_count(text))
    sentences = max(1, sentence_count(row))
    syllables = textstat.syllable_count(text, lang='en_US')
    score = 206.835 - 1.015 * (float(words)/sentences) - 84.6 * (float(syllables)/words)
    return score

def dale_chall(row):
    text = row['reviewText']
    easywords =  open("easy_words.txt").read().splitlines()
    words = tokenize.word_tokenize(text)
    words = list(map(lambda x:x.lower(),words))
    sentences = max(1, sentence_count(row))
    wordcount = max(1, textstat.lexicon_count(text))
    easywordcount = 0
    for easyword in easywords:
        easywordcount += words.count(easyword)
    diffwordsratio = (float(wordcount-easywordcount)/wordcount)
    score = 0.1579*(diffwordsratio*100)+0.0496*(float(wordcount)/sentences)
    if diffwordsratio > 0.05:
        score += 3.635
    return score

def word_count(row):
    text = row['reviewText']
    words = text.split()
    return len(words)

def avg_word_length(row):
    text = row['reviewText']
    words = text.split()
    if len(words) == 0:
        return 0
    average = sum(len(word) for word in words) / len(words)
    return average

def word_length_diversity(row):
    text = row['reviewText']
    words = text.split()
    word_lengths = list(map(lambda x: len(x), words))
    return np.std(word_lengths)

def sentence_count(row):
    text = row['reviewText']
    sentences = re.findall(r'[!?]|(\. )|(\.$)', text)
    return max(1, len(sentences))
    
def avg_sentence_length(row):
    text = row['reviewText']
    sentences = tokenize.sent_tokenize(text)
    if len(sentences) == 0:
        return 0
    average = sum(len(sentence) for sentence in sentences) / len(sentences)
    return average

def sentence_length_diversity(row):
    text = row['reviewText']
    sentences = tokenize.sent_tokenize(text)
    sentence_lengths = list(map(lambda x: len(x), sentences))
    return np.std(sentence_lengths)

def type_token_ratio(row):
    text = row['reviewText']
    lex = LexicalRichness(text)
    try:
        return lex.ttr
    except:
        return 0

def get_features(df):
    new_columns = ['unique_words', 'word_count', 'avg_word_length', 'word_length_diversity', 'sentence_count', 'avg_sentence_length', 'sentence_length_diversity', 'type_token_ratio', 'flesch_kincaid', 'dale_chall']
    all_columns = list(df.columns).extend(new_columns)
    df = df.astype({"reviewText": str})
    df = df[df.reviewText != '']

    for name in new_columns:
        print(name)
        string_to_eval = name + '(row)'
        df[name] = df.apply(lambda row: eval(string_to_eval), axis=1)
        df[name]=((df[name]-df[name].min())/(df[name].max()-df[name].min()))
        
    df['price_log'] = df.apply(lambda row: math.log10(row['price']), axis=1)
    df['overall_std']=((df['overall']-df['overall'].min())/(df['overall'].max()-df['overall'].min()))
    
    df = df.dropna()
    df.to_json("features.json")
    return df

