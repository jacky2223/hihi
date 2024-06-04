# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 17:32:06 2022

@author: FJUSER211027A
"""

import os
import pandas as pd
import jieba
print(jieba.__version__)
os.chdir("C:\\Users\\jacky\\Downloads")
YahooNews=pd.read_csv('航運報導.csv')

###############################
###Part I: Identify the Nose###
###############################
import re

RE_SUSPICIOUS = re.compile(r'[&#<>{}\[\]\\]')

def impurity(text, min_len=10):
    """returns the share of suspicious characters in a text"""
    if text == None or len(text) < min_len:
        return 0
    else:
        return len(RE_SUSPICIOUS.findall(text))/len(text)

YahooNews['Impurity']=YahooNews['Context'].apply(impurity, min_len=10)
YahooNews.columns
YahooNews[['Context', 'Impurity']].sort_values(by='Impurity', ascending=False).head(3)

#####################################################
###Part II: Removing Nose with Regular Expressions###
#####################################################

#remark: html.unescape
import html
p = '&lt;abc&gt;' #&lt; and &gt; are special simbles in html
#not showing in text example
txt= html.unescape(p)
print (txt)

import html

def clean(text):
    # convert html escapes like &amp; to characters.
    text = html.unescape(text) #in this example, this part does nothing
    # tags like <tab>
    text = re.sub(r'<[^<>]*>', ' ', text)
    # markdown URLs like [Some text](https://....)
    text = re.sub(r'\[([^\[\]]*)\]\([^\(\)]*\)', ' ', text)
    # text or code in brackets like [0]
    text = re.sub(r'\[[^\[\]]*\]', ' ', text)
    # standalone sequences of specials, matches &# but not #cool
    text = re.sub(r'(?:^|\s)[&#<>{}\[\]+|\\:-]{1,}(?:\s|$)', ' ', text)
    # standalone sequences of hyphens like --- or ==
    text = re.sub(r'(?:^|\s)[\-=\+]{2,}(?:\s|$)', ' ', text)
    # sequences of white spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

YahooNews['Clean_text'] = YahooNews['Context'].apply(clean)
YahooNews['Impurity']   = YahooNews['Clean_text'].apply(impurity, min_len=20)

YahooNews[['Clean_text', 'Impurity']].sort_values(by='Impurity', ascending=False) \
                              .head(3)     


####################################################
###Part III: Character Normalization with textacy###
####################################################  
#No need for Chinese


#############################################
###Part IV: Character Masking with textacy###
#############################################
from textacy.preprocessing import replace

YahooNews['Clean_text']=YahooNews['Clean_text'].apply(replace.urls)

YahooNews.rename(columns={'Context': 'Raw_text', 'Clean_text': 'Context'}, inplace=True)
YahooNews.drop(columns=['Impurity'], inplace=True)




##########################
###Liguistic Processing###
##########################
#1加入繁體詞典
import jieba

jieba.set_dictionary('dict.txt.big.txt')
stopwords1 = [line.strip() for line in open('stopWords.txt', 'r', encoding='utf-8').readlines()]

def remove_stop(text):
    c1=[]
    for w in text:
        if w not in stopwords1:
            c1.append(w)
    c2=[i for i in c1 if i.strip() != '']
    return c2



YahooNews['tokens']=YahooNews['Context'].apply(jieba.cut)
YahooNews['tokens_new']=YahooNews['tokens'].apply(remove_stop)
YahooNews.iloc[0,:]


#Freq charts
from collections import Counter
counter = Counter()#use a empty string first
YahooNews['tokens_new'].apply(counter.update)
print(counter.most_common(15))

import seaborn as sns
sns.set(font="SimSun")
min_freq=2
#transform dict into dataframe
freq_df = pd.DataFrame.from_dict(counter, orient='index', columns=['freq'])
freq_df = freq_df.query('freq >= @min_freq')
freq_df.index.name = 'token'
freq_df = freq_df.sort_values('freq', ascending=False)
freq_df.head(15)

ax = freq_df.head(15).plot(kind='barh', width=0.95, figsize=(8,3))
ax.invert_yaxis()
ax.set(xlabel='Frequency', ylabel='Token', title='Top Words')

###Creating Word Clouds
from matplotlib import pyplot as plt
from wordcloud import WordCloud ###
from collections import Counter ###

wordcloud = WordCloud(font_path="SimHei.ttf", background_color="white")
wordcloud.generate_from_frequencies(freq_df['freq'])
#plt.figure(figsize=(20,10)) 
plt.imshow(wordcloud)




