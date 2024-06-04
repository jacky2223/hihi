# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 16:05:24 2021

@author: Shawn
"""


def list_to_string(org_list, seperator=' '):
    return seperator.join(org_list)

YahooNews['News_seg']=YahooNews['tokens_new'].apply(list_to_string)
YahooNews['News_seg'][1]


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(decode_error='ignore', min_df=2) 

dt01 = cv.fit_transform(YahooNews['News_seg'])
print(cv.get_feature_names_out())
fn=cv.get_feature_names_out()


dtmatrix=pd.DataFrame(dt01.toarray(), columns=cv.get_feature_names_out())


from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(dt01[20], dt01[24])

sm = pd.DataFrame(cosine_similarity(dt01, dt01))


YahooNews['Context'][20]
YahooNews['Context'][24]



from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()


tfidf_dt = tfidf.fit_transform(dt01)
tfidfmatrix = pd.DataFrame(tfidf_dt.toarray(), columns=cv.get_feature_names_out())
cosine_similarity(tfidf_dt[20], tfidf_dt[22])


sm1 =pd.DataFrame(cosine_similarity(tfidf_dt, tfidf_dt))


sm2 = pd.DataFrame(cosine_similarity(tfidf_dt.transpose(), tfidf_dt.transpose()))

YahooNews['Context'][20]
YahooNews['Context'][22]


from matplotlib import pyplot as plt
from wordcloud import WordCloud ###
from collections import Counter ###

tfidfsum=tfidfmatrix.T.sum(axis=1)

wordcloud = WordCloud(font_path="SimHei.ttf", background_color="white")
wordcloud.generate_from_frequencies(tfidfsum)
#plt.figure(figsize=(20,10)) 
plt.imshow(wordcloud)

from sklearn.cluster import KMeans

from sklearn import preprocessing 
distortions = []
for i in range(1, 31):
    km = KMeans(
        n_clusters=i, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )
    km.fit(preprocessing.normalize(tfidf_dt))
    distortions.append(km.inertia_)

# plot
from matplotlib import pyplot as plt
plt.plot(range(1, 31), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()


km = KMeans(
    n_clusters=5, init='random',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=0
)
y_km = km.fit_predict(preprocessing.normalize(tfidf_dt))

g0 = YahooNews['Context'][y_km==0]
g0.head()
g1 = YahooNews['Context'][y_km==1]
g1.head()
g2 = YahooNews['Context'][y_km==2]
g2.head()
g3 = YahooNews['Context'][y_km==3]
g3.head()
g4 = YahooNews['Context'][y_km==4]
g4.head()




YahooNews['length']= YahooNews['Context'].str.len()

YahooNews.info()
YNews = YahooNews.copy()  

YNews.dtypes
YNews['DateTime'] = YNews['Date'].astype(str) + " " + YNews['Time']
type(YNews['DateTime'])
YNews['DateTime'].dtypes
YNews['DateTime'] = pd.to_datetime(YNews['DateTime'], format='%Y/%m/%d %H:%M:%S')
YNews['DateTime'][1]-YNews['DateTime'][0]

#轉成日期格式
YNews['DateOnly'] = YNews['DateTime'].dt.date
YNews['week_of_year'] = YNews['DateTime'].dt.isocalendar().week
YNews['day_of_week'] = YNews['DateTime'].dt.dayofweek
YNews['day'] = YNews['DateTime'].dt.day
YNews['month'] = YNews['DateTime'].dt.month
YNews['year'] = YNews['DateTime'].dt.year

YNews.describe()
#agg可計算新聞之長度，plot可將其畫成圖形
YNews.groupby('DateOnly').agg({'length':'mean'}).plot(rot=45)
#size可計算新聞之篇幅數量
YNews.groupby('day_of_week').size().plot(rot=45)
#畫出長條圖
YNews.groupby('day_of_week').size().plot.bar()


#計算文章來自各網站的次數
YahooNews['From'].value_counts()

#尋找各文章和標題包含關鍵字的字數
YahooNews[YahooNews['Context'].str.contains('貨櫃')].From.count()
YahooNews[YahooNews['Context'].str.contains('貨櫃')].From.value_counts()
YahooNews[YahooNews['Title'].str.contains('貨櫃')].From.value_counts()

YahooNews[YahooNews['Context'].str.contains('運價')].From.value_counts()
YahooNews[YahooNews['Title'].str.contains('運價')].From.value_counts()

YahooNews[YahooNews['Context'].str.contains('TW')].From.value_counts()
YahooNews[YahooNews['Title'].str.contains('TW')].From.value_counts()

YahooNews[YahooNews['Context'].str.contains('紅海')].From.value_counts()
YahooNews[YahooNews['Title'].str.contains('紅海')].From.value_counts()

YahooNews[YahooNews['Context'].str.contains('航線')].From.value_counts()
YahooNews[YahooNews['Title'].str.contains('航線')].From.value_counts()

#YahooNews 的敘述性統計
YahooNews.describe()
YahooNews['length'].var()
YahooNews['length'].plot(kind='hist').mean()

#畫各網站的平均文章長度的箱型圖
YahooNews['length'].plot(kind='box')
YNews =YahooNews[YahooNews['From'].isin(['經濟日報','yahoo新聞','中時新聞網','鉅亨網'])]
YNews.boxplot(column='length',by='From', vert=False)












