#!/usr/bin/env python
# coding: utf-8

# In[133]:


import numpy as np
import pandas as pd
import ast
get_ipython().system('pip install nltk')
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[134]:


movies= pd.read_csv('tmdb_5000_movies.csv')
credits= pd.read_csv('tmdb_5000_credits.csv')


# In[135]:


movies.head()


# In[136]:


credits.head()


# In[137]:


movies.merge(credits,on='title').shape


# In[138]:


movies =movies.merge(credits,on='title')


# In[139]:


movies.head()


# In[140]:


#genres
#id
#keywords
# title
#overview
#cast
#crew 
movies= movies[['movie_id','vote_average','title','overview','genres','keywords','cast','crew']]


# In[141]:


movies.info()


# In[142]:


movies.isnull().sum()


# In[143]:


movies.duplicated().sum()


# In[144]:


movies.iloc[0].genres


# In[145]:





# In[146]:


def my_convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[147]:


movies['genres']=movies['genres'].apply(my_convert)


# In[148]:


movies.head(2)


# In[149]:


movies['keywords']=movies['keywords'].apply(my_convert)


# In[150]:


def my_convert2(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter !=10:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L


# In[151]:


movies['cast']=movies['cast'].apply(my_convert2)


# In[152]:


movies.head(2)


# In[153]:


movies['cast'][0]


# In[154]:


movies['crew'][0]


# In[155]:


def my_fetch_crew(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director' or i['job']=='Writer' or i['job']=='Producer':
            L.append(i['name'])
    return L


# In[156]:


movies['crew']=movies['crew'].apply(my_fetch_crew)


# In[157]:


movies.head(2)


# In[158]:


movies['crew'][0]


# In[159]:


movies['overview'][0]


# In[160]:


movies['overview']=movies['overview'].apply(lambda x:str(x).split())


# In[161]:


movies.dtypes


# In[162]:


movies['genres']=movies['genres'].apply(lambda x:[i.replace(' ','') for i in x])


# In[163]:


movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(' ','') for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(' ','') for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(' ','') for i in x])


# In[164]:


movies.head()


# In[165]:


movies['tags']= movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']


# In[166]:


movies.head()


# In[167]:


movies['tags'][0]


# In[168]:


new_df=movies[['movie_id','vote_average','title','tags']]


# In[169]:


new_df.head()


# In[170]:


new_df['tags']=new_df['tags'].apply(lambda x:' '.join(x))


# In[171]:


new_df.head()


# In[172]:


new_df['tags'][0]


# In[173]:


new_df.head()


# In[174]:


new_df['tags'][0]


# In[175]:


new_df['tags']=new_df['tags'].apply(lambda x:x.lower())


# In[176]:


new_df.head()


# In[177]:





# In[178]:





# In[179]:




ps=PorterStemmer()


# In[180]:


def stem(text):
    y=[]
    
    for i in str(text).split():
        y.append(ps.stem(i))
    return ' '.join(y)


# In[181]:


new_df['tags']=new_df['tags'].apply(stem)


# In[182]:


new_df['tags'][0]


# In[183]:



cv= CountVectorizer(max_features=5000,stop_words='english')


# In[184]:


vectors=cv.fit_transform(new_df['tags']).toarray()


# In[185]:


vectors


# In[186]:


cv.get_feature_names()


# In[187]:


vectors.shape


# In[188]:





# In[189]:


similarity=cosine_similarity(vectors)


# In[271]:


def recommend_content1(movie):
    movie_index= new_df[new_df['title']==movie].index[0]
    distances=similarity[movie_index]
    movies_list= sorted(list(enumerate(distances)), reverse=True,key=lambda x:x[1])[1:11]
    title=[]
    rating=[]
    for i in movies_list:
        title.append(new_df.iloc[i[0]].title)
        rating.append(new_df.iloc[i[0]].vote_average)
    index=['1','2','3','4','5','6','7','8','9','10']
    df = pd.DataFrame(list(zip(index,title, rating)),
               columns =['index','title', 'rating'])
    return df


# In[272]:


recommended_movies=recommend_content1('Batman')


# In[273]:


recommended_movies


# In[278]:


user_ratings = x.pivot_table(index=['index'],columns=['title'],values='rating')
user_ratings


# In[277]:


item_similarity_df = user_ratings.corr(method='pearson')
item_similarity_df.head()


# In[ ]:




