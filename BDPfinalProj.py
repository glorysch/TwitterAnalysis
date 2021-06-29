#!/usr/bin/env python
# coding: utf-8

# # Big Data Platforms Final Project 
# ## Glory Scheel
# ### December 11, 2020
# 

# ## Import Data

# In[135]:


import os
os.environ['PROJ_LIB'] = 'C:/Users/glory/Anaconda3/Lib/site-packages/mpl_toolkits/basemap'
from mpl_toolkits.basemap import Basemap
import nltk
from simhash import Simhash, SimhashIndex
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
import seaborn as sns


# In[2]:


yDF=pd.read_csv('notebooks_scheelg_yaleDF2.csv')
mDF=pd.read_csv('notebooks_scheelg_fsuDF.csv')
ufDF=pd.read_csv('notebooks_scheelg_ufDF2.csv')
cDF=pd.read_csv('notebooks_scheelg_chiDF2.csv')


# In[3]:


print('Chicago DF length:', len(cDF))
print('Yale DF length:',len(yDF))
print('FSU DF length:',len(mDF))
print('Florida DF length:',len(ufDF))


# ## Cleaning

# create function that:
# - drop rows with null values in user column
# - splits up elements in each user cell so I can extract individual variables

# In[3]:


def cleanSet(DF):
    DF = DF[DF['user'].notna()]
    meep=[i[4:-1].split(', ') for i in DF['user']]
    moop=[]
    for j in meep:
        merp=[]
        for i in j:
            if '=' in i:
                merp.append(i)
            else:
                pass
        moop.append(merp)
    DF['followers']=[i[7] for i in moop]
    DF['location']=[i[16] for i in moop]
    DF['following']=[i[9] for i in moop]
    DF['total_tweets']=[i[33] for i in moop]
    DF=DF.drop(columns='user')
    DF['followers']=DF['followers'].str.extract('(\d+)', expand=False)
    DF['location']=DF['location'].str.lstrip('location=')
    DF['location']=DF['location'].str.lstrip("''").str.rstrip("''")
    DF['following']=DF['following'].str.extract('(\d+)', expand=False)
    DF['total_tweets']=DF['total_tweets'].str.extract('(\d+)', expand=False)
    return(DF)


# In[4]:


uChiDF=cleanSet(cDF)
yaleDF=cleanSet(yDF)
UFDF=cleanSet(ufDF)
fsuDF=cleanSet(mDF)


# ## Identify most influential twitter users
# - By message volume
# - By message retweet
# - How much are they tweeting about the Universities vs. other topics?
# 
# I aggregated the data by user and either grouped the users data by a sum or max value depending on what was appropriate. I then created a column called influence rating that used a weighted linear equation of some of the columns to create a score for the user's influence. I grabbed the top 5 influential users.

# In[121]:


def AggUser(DF):
    DF['uni_tweet_count']=1
    
    DF['followers']=pd.to_numeric(DF['followers'])
    DF['total_tweets']=pd.to_numeric(DF['total_tweets'])
    DF['favorite_count']=pd.to_numeric(DF['favorite_count'])
    DF['quote_count']=pd.to_numeric(DF['quote_count'])
    DF['reply_count']=pd.to_numeric(DF['reply_count'])
    DF['retweet_count']=pd.to_numeric(DF['retweet_count'])
    
    Maxcols=['id','followers','total_tweets']
    Sumcols=['favorite_count','id','quote_count','reply_count','retweet_count','uni_tweet_count']
    
    maxDF=DF[Maxcols].groupby('id').max()
    sumDF=DF[Sumcols].groupby('id').sum()
    
    maxDF=maxDF.sort_values(by='id')
    sumDF=sumDF.sort_values(by='id')
    
    newDF=pd.concat([maxDF,sumDF],axis=1)
    newDF['percent_uni_tweet']=newDF['uni_tweet_count']/newDF['total_tweets']
    newDF['influence_rating']=(newDF['followers']*.35)+(newDF['percent_uni_tweet']*.35)+(newDF['total_tweets']*.10)+(newDF['retweet_count']*.80)
    newDF=newDF[newDF['total_tweets']!=0]
    newDF=newDF[newDF['followers'].notnull()]
    return(newDF)


# ### University of Chicago

# In[122]:


chiAgg=AggUser(uChiDF)


# In[123]:


chiAgg.head()


# In[124]:


chiAgg.sort_values(by='influence_rating',ascending=False).iloc[0:5]


# ### Yale University

# In[125]:


yaleAgg=AggUser(yaleDF)


# In[126]:


yaleAgg.head()


# In[127]:


yaleAgg.sort_values(by='influence_rating',ascending=False).iloc[1:6]


# ### Florida State University

# In[128]:


fsuAgg=AggUser(fsuDF)


# In[129]:


fsuAgg.head()


# In[130]:


fsuAgg.sort_values(by='influence_rating',ascending=False).iloc[0:5]


# ### University of Florida

# In[131]:


ufAgg=AggUser(UFDF)


# In[132]:


ufAgg.head()


# In[133]:


ufAgg.sort_values(by='influence_rating',ascending=False).iloc[0:6]


# ## Where are the twitter users located?
# - create function that grabs the lat and lons for every location and plots a density plot on a map

# In[76]:


def plotLatLon(DF):
    mer=DF['location'].value_counts()[1:100]
    mee=pd.DataFrame(mer)
    mee.reset_index(inplace=True)

    geolocator = Nominatim(user_agent="gloryapp")
    lat=[]
    lon=[]
    for i in mee['index']:
        try:
            location = geolocator.geocode(i)#looks like hyde park gives the wrong address so since this is a top place in chicago we will just change this inthe location column to chicago
            latlon=[i for i in location.raw.values()][5:7]
            lat.append(latlon[0])
            lon.append(latlon[1])
        except:
            lat.append(np.nan)
            lon.append(np.nan)
    mee['lat']=lat
    mee['lon']=lon
    mee = mee[mee['lat'].notna()]
    mee["lat"] = pd.to_numeric(mee["lat"])
    mee["lon"] = pd.to_numeric(mee["lon"])
    mee["location"] = pd.to_numeric(mee["location"])
    
    fig = plt.figure(figsize=(20, 15), edgecolor='w')
    m = Basemap(projection='cyl', resolution=None,
            llcrnrlat=-50, urcrnrlat=80,
            llcrnrlon=-180, urcrnrlon=180, )
    m.etopo(scale=0.5, alpha=0.5)
    x, y, s = mee['lon'],mee['lat'], mee['location']
    
    plt.scatter(x, y, s=s/100)


# ### University of Chicago
# -  first do a little cleaning of the top locations

# In[118]:


uChiDF["location"]=uChiDF["location"].str.lower()
uChiDF["location"].replace({"hyde park": "university of chicago"}, inplace=True)#hyde park was a top location
uChiDF=uChiDF[uChiDF['location']!='none']


# In[120]:


plotLatLon(uChiDF)


# ### Yale University

# In[127]:


yaleDF["location"]=yaleDF["location"].str.lower()
yaleDF=yaleDF[yaleDF['location']!='none']
plotLatLon(yaleDF)


# ### Florida State University

# In[79]:


fsuDF["location"]=fsuDF["location"].str.lower()
fsuDF=fsuDF[fsuDF['location']!='none']
plotLatLon(fsuDF)


# ### University of Florida

# In[130]:


UFDF["location"]=UFDF["location"].str.lower()
UFDF=UFDF[UFDF['location']!='none']
plotLatLon(UFDF)


# 

# ## What distinguishes University of Chicago Twitterers vs Twitterers who tweet about other universities
# - make visualization

# In[163]:


uChiDF.columns


# In[191]:


uChiDF['uni']='UChicago'
yaleDF['uni']='Yale'
UFDF['uni']='UF'
fsuDF['uni']='FSU'
newDF1=pd.concat([uChiDF,yaleDF])
newDf2=pd.concat([newDF1,UFDF])
newDf3=pd.concat([newDf2,fsuDF])


# In[196]:


uniDF=newDf3.groupby('uni').median()


# In[160]:


uniDF.columns


# In[197]:


uniDF=uniDF[['followers', 'total_tweets', 'uni_tweet_count']].reset_index()
uniDF


# In[173]:


uniDF.columns


# In[198]:


uniDF.plot('uni',[1,2,3],kind = 'bar')


# It looks like Uchicago has the highest amount of median followers and the lowest amount of total_tweets. This would suggest to me that they have a very good tweet to follower ratio but could possibly increase the amount of tweets per user to get a better influence ranking.

# ## What are the timelines of these tweets? Do you see significant peaks and valleys?
# - Do you see data collection gaps?

# ### University of Chicago

# In[15]:


chiDT=pd.to_datetime(uChiDF['created_at'])
chiDT.hist()


# ### Yale University

# In[16]:


yaleDT=pd.to_datetime(yaleDF['created_at'])
yaleDT.hist()


# ### Florida State University

# In[134]:


fsuDT=pd.to_datetime(fsuDF['created_at'])
fsuDT.hist()


# ### University of Florida

# In[17]:


ufDT=pd.to_datetime(UFDF['created_at'])
ufDT.hist()


# ## How unique are the messages for each of these universities?
# - Are they mostly unique? Or mostly people are just copy-pasting the same text?
# - You can use something like Jaccard similarity / Cosine Similarity / Simhash / Minhash to measure uniqueness / similarity
# - Visualize message duplication (for each university – not between the universities)
# - Please note: this is not a topic modeling (LDA / LSA) – but text similarity analysis.

# In[63]:


def get_features(s):
    width = 5
    try:
        s = s.lower()#makes string lowercase
    except:
        print(s)
    s = re.sub(r'[^\w]+', '', s)#removes spaces and non letter characters
    return [s[i:i + width] for i in range(max(len(s) - width + 1, 1))]#splits the letters up in intervals of 3 as shown below


# In[52]:


def simHashBarPlot(df):
    dat=df['text2'].to_dict()
    data_first = {k: dat[k] for k in list(dat)[:5]}
    objs = [(str(k), Simhash(get_features(v))) for k, v in dat.items()]
    index = SimhashIndex(objs, k=3)
    
    dups=[]
    for i,j in dat.items():
        s=Simhash(get_features(j))
        ind=index.get_near_dups(s)
        dups.append([i,j,len(ind)-1,ind])#get key, value amount of duplicates and index of duplicates
    dupDF=pd.DataFrame(dups,columns=['index','title','amountDuplicates','index of duplicates'])
    isDuplicated=[]
    for i in dupDF['amountDuplicates']:
        if i==0:
            isDuplicated.append(0)
        else:
            isDuplicated.append(1)
    dupDF['isDuplicated']=isDuplicated
    counts=dupDF['isDuplicated'].value_counts()
    counts.plot(kind='bar')
    plt.xlabel("Count of Near Duplicated(1) vs. Count of Unique(0)", labelpad=14)


# In[ ]:


def simHashDensPlot(df)
isDupDF=dupDF.where(dupDF['isDuplicated']==1).dropna()
isDupDF=isDupDF.reset_index()
isDupDF['index']=isDupDF.index
isDupDF.head()

sns.distplot(isDupDF['amountDuplicates'])


# ### University of Chicago

# In[80]:


chiSubset=uChiDF.sample(n=20000)
simHashBarPlot(chiSubset)


# ### Yale University

# In[55]:


yaleSubset=yaleDF.sample(n=20000)
simHashBarPlot(yaleSubset)


# ### Florida State University

# In[56]:


fsuSubset=fsuDF.sample(n=20000)
simHashBarPlot(fsuSubset)


# ### University of Florida

# In[73]:


ufDF=ufDF[ufDF['text2'].notnull()]


# In[74]:


#ufDF=ufDF[ufDF['text2']!='nan']
ufSubset=UFDF.sample(n=20000)
simHashBarPlot(ufSubset)

