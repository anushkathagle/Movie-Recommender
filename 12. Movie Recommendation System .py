#!/usr/bin/env python
# coding: utf-8

# ## <center>------- Movie Recommendation System - Project ---------</center>

# ### Filtering Based System

# In[1]:


import pandas as pd

import warnings
warnings.filterwarnings("ignore")


# In[2]:


Movies = pd.read_csv("movies.csv")
Ratings = pd.read_csv("ratings.csv")


# In[3]:


Movies.shape


# In[4]:


Movies.keys()


# In[5]:


Ratings.shape


# In[6]:


Ratings.keys()


# In[7]:


Ratings.head(10)


# In[8]:


Movies.head()


# In[9]:


df = pd.merge(Movies, Ratings, on="movieId")


# In[10]:


df.head()


# In[11]:


def Convert(title):
    return title.lower()


# In[12]:


df.title = df.title.apply(Convert)


# In[13]:


df


# In[14]:


df.title.value_counts()


# In[15]:


### Calculate mean Reating of All Movies
group = df.groupby('title')['rating'].mean().sort_values(ascending = False).head()
group


# In[16]:


dfNew = pd.DataFrame(df.groupby('title')['rating'].mean())
dfNew["NumOfRatings"] = df.groupby('title')["rating"].count()


# In[17]:


dfNew.head()


# In[18]:


df.head(10)


# In[19]:


movieTitle = df.pivot_table(index="userId", columns='title', values="rating")


# In[20]:


movieTitle.head(5)


# In[ ]:





# In[32]:


def Rec():
    movieName = input("Enter Movie Name").lower()
    userRatings = movieTitle[movieName]
    similar_movies = movieTitle.corrwith(userRatings)
    corr_movie = pd.DataFrame(similar_movies, columns=["Correlation"])
    corr_movie.dropna(inplace=True)
    corr_movie = corr_movie.join(dfNew.NumOfRatings)
    Sugg = corr_movie[corr_movie.NumOfRatings > 100].sort_values("Correlation", ascending = False)
    Sugg[:5]


# In[34]:


Rec()


# ### Content Based

# In[47]:


import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


# In[49]:


df = pd.read_csv("data.csv")


# In[50]:


df.head()


# In[51]:


df.keys()


# In[52]:


features = ["keywords", "cast","genres", "director", ]


# In[54]:


for i in features:
    df[i] = df[i].fillna("")


# In[55]:


def MovieCount(row):
    
    Out = ""
    for i in features:
        Out += row[i]
        Out += " "

    return Out


# In[56]:


MovieCount(df[0:1])[0]


# In[57]:


df["movie_content"] = df.apply(MovieCount, axis=1)


# In[58]:


df.movie_content


# In[59]:


def LowerCaseTitle(title):
    return title.lower()


# In[60]:


df.title = df.title.apply(LowerCaseTitle)


# In[62]:


cv = CountVectorizer()


# In[63]:


metrix = cv.fit_transform(df.movie_content)


# In[64]:


len(cv.get_feature_names())


# In[65]:


word_sim = cosine_similarity(metrix)


# In[66]:


df.title.head()


# In[67]:


movie_name = input("Enter Your Movie")
movie_name = movie_name.lower()
m_index = df[df.title == movie_name]['index'].values[0]
similar_word_value = word_sim[m_index]
similar_movies = list(enumerate(similar_word_value))
sorted_similar_movies = sorted(similar_movies, key = lambda x:x[1], reverse=True)
movies_num = 5
print("\n Top Similar Movies:")
for i in range(movies_num):
    index = sorted_similar_movies[i][0]
    movie = list(df[index:index+1].title)[0]
    print(i+1, movie)


# In[ ]:





# In[ ]:




