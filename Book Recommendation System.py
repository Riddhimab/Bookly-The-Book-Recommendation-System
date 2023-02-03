#!/usr/bin/env python
# coding: utf-8

# In[57]:


import numpy as np
import pandas as pd


# In[58]:


books= pd.read_csv("books.csv")
users= pd.read_csv("users.csv")
ratings= pd.read_csv("ratings.csv")


# In[59]:


books.head()


# In[60]:


users.head()


# In[61]:


ratings.head()


# In[62]:


print(books.shape)
print(users.shape)
print(ratings.shape)


# In[63]:


books.isnull().sum()


# In[64]:


users.isnull().sum()


# In[65]:


ratings.isnull().sum()


# In[66]:


books.duplicated().sum()


# In[67]:


ratings.duplicated().sum()


# In[68]:


users.duplicated().sum()


# Popularity Based Recommendation System

# In[69]:


ratings.merge(books, on= 'ISBN').shape


# In[70]:


books_rating=ratings.merge(books, on= 'ISBN')


# In[71]:


books_rating.groupby("Book-Title").count()


# In[72]:


count_rating=books_rating.groupby("Book-Title").count()["Book-Rating"].reset_index()
count_rating.rename(columns={'Book-Rating':"Count_ratings"}, inplace=True)


# In[73]:


count_rating


# In[181]:


avg_rating=books_rating.groupby("Book-Title").mean()["Book-Rating"].reset_index()
avg_rating.rename(columns={'Book-Rating':"Average_ratings"}, inplace=True)
avg_rating=avg_rating.round(2)


# In[182]:


avg_rating


# In[76]:


popularity_df=count_rating.merge(avg_rating, on = 'Book-Title')


# In[78]:


popularity_data=popularity_df[popularity_df['Count_ratings']>=250].sort_values('Average_ratings',ascending=False).head(50)


# In[79]:


popularity_data=popularity_data.merge(books, on="Book-Title").drop_duplicates("Book-Title")[['Book-Title','Book-Author','Image-URL-M','Count_ratings','Average_ratings']]


# In[80]:


popularity_data['Image-URL-M'][0]


# In[140]:


popularity_data


# Collaborative Filtering Based Recommender System

# In[81]:


x = books_rating.groupby('User-ID').count()['Book-Rating'] > 200
seasoned_readers = x[x].index


# In[82]:


filtered_rating = books_rating[books_rating['User-ID'].isin(seasoned_readers)]


# In[83]:


y=filtered_rating.groupby("Book-Title").count()['Book-Rating']>=50
famous_books=y[y].index


# In[84]:


final_rating=filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]


# In[85]:


final_rating.drop_duplicates()


# In[86]:


pivot_table=final_rating.pivot_table(index='Book-Title',columns='User-ID',values='Book-Rating')


# In[87]:


pivot_table.fillna(0,inplace=True)


# In[88]:


pivot_table


# In[89]:


from sklearn.metrics.pairwise import cosine_similarity


# In[90]:


similarity_score=cosine_similarity(pivot_table)


# In[91]:


similarity_score.shape


# In[175]:



def recommend(book_name):
        # index fetch
    data = []
    try:
        index = np.where(pivot_table.index==book_name)[0][0]
        similar_items = sorted(list(enumerate(similarity_score[index])),key=lambda x:x[1],reverse=True)[1:5]
    
        for i in similar_items:
            item = []
            temp_df = books[books['Book-Title'] == pivot_table.index[i[0]]]
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
        
            data.append(item)
    except IndexError:
            data=("This title could not be found. Popular searches: 1984, Zoya, 2nd Chance,4 Blondes")
    return data
    


# In[149]:


pivot_table.index


# In[185]:


import pickle
pickle.dump(popularity_data,open('popularity1.pkl','wb'))


# In[131]:


books.drop_duplicates('Book-Title').sort_values(by =["Book-Title"])


# In[96]:


pickle.dump(pivot_table,open('pivot_table.pkl','wb'))
pickle.dump(books,open('books.pkl','wb'))
pickle.dump(similarity_score,open('similarity_scores.pkl','wb'))


# In[110]:


books


# In[177]:


recommend("Zen and the Art of Motorcycle Maintenance: An Inquiry into Values")


# In[176]:


recommend("Harry")


# In[ ]:




