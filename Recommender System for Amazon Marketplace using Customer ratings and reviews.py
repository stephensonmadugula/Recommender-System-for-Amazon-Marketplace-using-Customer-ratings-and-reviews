#!/usr/bin/env python
# coding: utf-8

# # Collaborative filtering - memory based using cosine distance and kNN

# Recommender systems are an integral part of many online systems. From e-commerce to online streaming platforms.
# Recommender systems employ the past purchase patters on it's user to predict which other products they may in interested in and likey to purchase. Recommending the right products gives a significat advantage to the business. A mojor portion of the revenue is generated through recommendations.
# 

# The Collaborative Filtering algorithm is very popular in online streaming platforms and e-commerse sites where the customer interacts with each product (which can be a movie/ song or consumer products) by either liking/ disliking or giving a rating of sorts.
# One of the requirements to be able to apply collaborative filtering is that sufficient number of products need ratings associated with not them. User interaction is required.
# 
# 
# 

# This notebook walks through the implementation of collaborative filtering using memory based technique of distnce proximity using cosine distances and nearest neighbours.

# ## Importing libraries and initial data checks

# In[36]:


# import required libraries
import pandas as pd
import numpy as np


# ### About the data
# 
# This is a dataset related to over 2 Million customer reviews and ratings of Beauty related products sold on Amazon's website.
# 
# It contains:
# - the unique UserId (Customer Identification),
# - the product ASIN (Amazon's unique product identification code for each product),
# - Ratings (ranging from 1-5 based on customer satisfaction) and
# - the Timestamp of the rating (in UNIX time)

# In[37]:


# read the dataset
df = pd.read_csv('ratings_Beauty.csv')
df.shape


# In[38]:


# check the first 5 rows
df.head()


# Check if there are any duplicate values present

# In[39]:


duplicates = df.duplicated(["UserId","ProductId", "Rating", "Timestamp"]).sum()
print(' Duplicate records: ',duplicates)


# See the number of unique values present

# In[40]:


print('unique users:',len(df.UserId.unique()))
print('unique products:',len(df.ProductId.unique()))
print("total ratings: ",df.shape[0])


# Check for null values

# In[41]:


df.isnull().any()


# Number of rated products per user

# In[42]:


products_user= df.groupby(by = "UserId")["Rating"].count().sort_values(ascending =False)
products_user.head()


# Number of ratings per product

# In[43]:


product_rated = df.groupby(by = "ProductId")["Rating"].count().sort_values(ascending = False)
product_rated.head()


# Number of products rated by each user

# In[44]:


rated_users=df.groupby("UserId")["ProductId"].count().sort_values(ascending=False)
print(rated_users)


# In[45]:


rated_products=df.groupby("ProductId")["UserId"].count().sort_values(ascending=False)
print(rated_products)


# Number of products with some minimum ratings

# In[46]:


print('Number of products with minimum of 5 reviews/ratings:',rated_products[rated_products>5].count())
print('Number of products with minimum of 4 reviews/ratings:',rated_products[rated_products>4].count())
print('Number of products with minimum of 3 reviews/ratings:',rated_products[rated_products>3].count())
print('Number of products with minimum of 2 reviews/ratings:',rated_products[rated_products>2].count())
print('Number of products with minimum of 1 reviews/ratings:',rated_products[rated_products>1].count())


# ## Visualizing the data

# In[47]:


# plot the data
import plotly.graph_objects as go
index = ['Total size of records', "Number of unique users","Number of unique products"]
values =[len(df),len(df['UserId'].unique()),len(df['ProductId'].unique())]

plot = go.Figure([go.Bar(x=index, y=values,textposition='auto')])
plot.update_layout(title_text='Number of Users and Products w.r.to Total size of Data',
                    xaxis_title="Records",
                    yaxis_title="Total number of Records")

plot.show()


# ### The ratings given by users

# In[48]:


print("Range of Ratings: ", df['Rating'].value_counts())
print(list(df['Rating'].value_counts()))

values = list(df['Rating'].value_counts())

plot = go.Figure([go.Bar(x = df['Rating'].value_counts().index, y = values,textposition='auto')])

plot.update_layout(title_text='Ratings given by user',
                    xaxis_title="Rating",
                    yaxis_title="Total number of Ratings")

plot.show()


# ### Products which are most popular

# In[49]:


print("Products with occurred the most: \n",df['ProductId'].value_counts().nlargest(5))

values = list(df['ProductId'].value_counts())


plot = go.Figure([go.Bar(x = df['ProductId'].value_counts().nlargest(5).index, y = values,textposition='auto')])

plot.update_layout(title_text='Most rated products',
                    xaxis_title="ProductID",
                    yaxis_title="Number of times occurred in the data")

plot.show()


# ### Average rating given by each user
# 

# In[50]:


ratings_per_user = df.groupby('UserId')['Rating'].count().sort_values(ascending=False)
print("Average rating given by each user: ",ratings_per_user.head())

plot = go.Figure(data=[go.Histogram(x=ratings_per_user)])
plot.show()


# In[51]:


ratings_per_product = df.groupby('ProductId')['Rating'].count().sort_values(ascending=False)
# print("Average rating given by each user: ",ratings_per_user.head())

plot = go.Figure(data=[go.Histogram(x=ratings_per_product)])
plot.show(title_text='Number of ratings per product',
                    xaxis_title="Product",
                    yaxis_title="Number of ratings")


# In[52]:


ratings_per_product = df.groupby('ProductId')['Rating'].count().sort_values(ascending=False)
# print("Average rating given by each user: ",ratings_per_user.head())

plot = go.Figure(data=[go.Histogram(x=ratings_per_product.nlargest(2000))])
plot.show(title_text='Number of ratings per product',
                    xaxis_title="Product",
                    yaxis_title="Number of ratings")


# ### Products with very less ratings
# 

# In[53]:



rating_of_products = df.groupby('ProductId')['Rating'].count()
# convert to make dataframe to analyse data
number_of_ratings_given = pd.DataFrame(rating_of_products)
print("Products with ratings given by users: \n",number_of_ratings_given.head())

less_than_ten = []
less_than_fifty_greater_than_ten = []
greater_than_fifty_less_than_hundred = []
greater_than_hundred = []
average_rating = []

for rating in number_of_ratings_given['Rating']:
    if rating <=10:
        less_than_ten.append(rating)
    if rating > 10 and rating <= 50:
        less_than_fifty_greater_than_ten.append(rating)
    if rating > 50 and rating <= 100:
        greater_than_fifty_less_than_hundred.append(rating)
    if rating > 100:
        greater_than_hundred.append(rating)

    average_rating.append(rating)
    
print("Ratings_count_less_than_ten: ", len(less_than_ten))
print("Ratings_count_greater_than_ten_less_than_fifty: ", len(less_than_fifty_greater_than_ten))
print("Ratings_count_greater_than_fifty_less_than_hundred: ", len(greater_than_fifty_less_than_hundred))
print("Ratings_count_greater_than_hundred: ", len(greater_than_hundred))
print("Average number of products rated by users: ", np.mean(average_rating))


# In[54]:


x_values = ["Ratings_count_less_than_ten","Ratings_count_greater_than_ten_less_than_fifty",
           "Ratings_count_greater_than_fifty_less_than_hundred","Ratings_count_greater_than_hundred"]
y_values = [len(less_than_ten),len(less_than_fifty_greater_than_ten),len(greater_than_fifty_less_than_hundred),
            len(greater_than_hundred)]


plot = go.Figure([go.Bar(x = x_values, y = y_values, textposition='auto')])

plot.add_annotation(
        x=1,
        y=100000,
        xref="x",
        yref="y")

plot.update_layout(title_text='Ratings Count on Products',
                    xaxis_title="Ratings Range",
                    yaxis_title="Count of Rating")
plot.show()


# In[55]:


from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()


# ### To convert alphanumeric data to numeric

# In[56]:


dataset = df
dataset['user'] = label_encoder.fit_transform(df['UserId'])
dataset['product'] = label_encoder.fit_transform(df['ProductId'])
dataset.head()


# In[57]:



# average rating given by each user
average_rating = dataset.groupby(by="user", as_index=False)['Rating'].mean()
print("Average rating given by users: \n",average_rating.head())
print("----------------------------------------------------------\n")


# let's merge it with the dataset as we will be using that later
dataset = pd.merge(dataset, average_rating, on="user")
print("Modified dataset: \n", dataset.head())
print("----------------------------------------------------------\n")

# renaming columns
dataset = dataset.rename(columns={"Rating_x": "real_rating", "Rating_y": "average_rating"})
print("Dataset: \n", dataset.head())
print("----------------------------------------------------------\n")


# Certain users tend to give higher ratings while others tend to gibve lower ratings. To negate this bias, we normalise the ratings given by the users.

# In[58]:


dataset['normalized_rating'] = dataset['real_rating'] - dataset['average_rating']
print("Data with adjusted rating: \n", dataset.head())


# # Cosine Similarity

# We use a distance based metric - cosine similarity to identify similar users. It is important first, to remove products that have very low number of ratings.

# ## Filter based on number of ratings available

# In[59]:


rating_of_product = dataset.groupby('product')['real_rating'].count() # apply groupby 
ratings_of_products_df = pd.DataFrame(rating_of_product)
print("Real ratings:\n",ratings_of_products_df.head()) # check for real rating for products


# In[60]:


filtered_ratings_per_product = ratings_of_products_df[ratings_of_products_df.real_rating >= 200]
print(filtered_ratings_per_product.head())
print(filtered_ratings_per_product.shape)


# In[61]:


# build a list of products to keep
popular_products = filtered_ratings_per_product.index.tolist()
print("Popular product count which have ratings over average rating count: ",len(popular_products))
print("--------------------------------------------------------------------------------")

filtered_ratings_data = dataset[dataset["product"].isin(popular_products)]
print("Filtered rated product in the dataset: \n",filtered_ratings_data.head())
print("---------------------------------------------------------------------------------")

print("The size of dataset has changed from ", len(dataset), " to ", len(filtered_ratings_data))
print("---------------------------------------------------------------------------------")


# ## Creating the User-item matrix

# In[62]:


similarity = pd.pivot_table(filtered_ratings_data,values='normalized_rating',index='UserId',columns='product')
similarity = similarity.fillna(0)
print("Updated Dataset: \n",similarity.head())


# As you can see, this is a very sparse matrix

# In[63]:


from sklearn.metrics.pairwise import cosine_similarity
import operator


# In[64]:


selecting_users = list(similarity.index)
selecting_users = selecting_users[:100]
print("You can select users from the below list:\n",selecting_users)


# In[65]:


def getting_top_5_similar_users(user_id, similarity_table, k=5):
    '''

    :param user_id: the user we want to recommend
    :param similarity_table: the user-item matrix
    :return: Similar users to the user_id.
    '''

    # create a dataframe of just the current user
    user = similarity_table[similarity_table.index == user_id]
    # and a dataframe of all other users
    other_users = similarity_table[similarity_table.index != user_id]
    # calculate cosine similarity between user and each other user
    similarities = cosine_similarity(user, other_users)[0].tolist()

    indices = other_users.index.tolist()
    index_similarity = dict(zip(indices, similarities))

    # sort by similarity
    index_similarity_sorted = sorted(index_similarity.items(), key=operator.itemgetter(1))
    index_similarity_sorted.reverse()

    # take users
    top_users_similarities = index_similarity_sorted[:k]
    users = []
    for user in top_users_similarities:
        users.append(user[0])

    return users


# In[66]:


user_id = "A0010876CNE3ILIM9HV0"
similar_users = getting_top_5_similar_users(user_id, similarity)


# In[67]:


print("Top 5 similar users for user_id:",user_id," are: ",similar_users)


# ## Recommend products based on these top similar users

# In[68]:


def getting_top_5_recommendations_based_on_users(user_id, similar_users, similarity_table, top_recommendations=5):
    '''

    :param user_id: user for whom we want to recommend
    :param similar_users: top 5 similar users
    :param similarity_table: the user-item matrix
    :param top_recommendations: no. of recommendations
    :return: top_5_recommendations
    '''

    # taking the data for similar users
    similar_user_products = dataset[dataset.UserId.isin(similar_users)]
#     print("Products used by other users: \n", similar_user_products.head())
#     print("---------------------------------------------------------------------------------")

    # getting all similar users
    similar_users = similarity_table[similarity_table.index.isin(similar_users)]

    #getting mean ratings given by users
    similar_users = similar_users.mean(axis=0)


    similar_users_df = pd.DataFrame(similar_users, columns=['mean'])

    # for the current user data
    user_df = similarity_table[similarity_table.index == user_id]


    # transpose it so its easier to filter
    user_df_transposed = user_df.transpose()


    # rename the column as 'rating'
    user_df_transposed.columns = ['rating']

    # rows with a 0 value.
    user_df_transposed = user_df_transposed[user_df_transposed['rating'] == 0]


    # generate a list of products the user has not used
    products_not_rated = user_df_transposed.index.tolist()
#     print("Products not used by target user: ", products_not_rated)
#     print("-------------------------------------------------------------------")

    # filter avg ratings of similar users for only products the current user has not rated
    similar_users_df_filtered = similar_users_df[similar_users_df.index.isin(products_not_rated)]

    # order the dataframe
    similar_users_df_ordered = similar_users_df_filtered.sort_values(by=['mean'], ascending=False)



    # take the top products
    top_products = similar_users_df_ordered.head(top_recommendations)
    top_products_indices = top_products.index.tolist()


    return top_products_indices


# In[69]:


print("Top 5 productID recommended are: ",
      getting_top_5_recommendations_based_on_users(user_id, similar_users, similarity))


# In[70]:


filtered_ratings_data.shape


# In[71]:


filtered_ratings_data.head()


# In[72]:


filtered_ratings_data[filtered_ratings_data['UserId']=="A0010876CNE3ILIM9HV0"]


# In[73]:


from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(filtered_ratings_data,test_size=0.2)

train_data = pd.DataFrame(train_data)
test_data = pd.DataFrame(test_data)


# In[74]:


similarity = pd.pivot_table(train_data,values='normalized_rating',index='UserId',columns='product')
similarity = similarity.fillna(0)
print("Updated Dataset: \n",similarity.head())


# In[75]:


similarity.shape


# In[76]:


selecting_users = list(similarity.index)
selecting_users = selecting_users[:100]
print("You can select users from the below list:\n",selecting_users)


# In[77]:


user_id = "A02720223TDVZSWVZYFN7"
similar_users = getting_top_5_similar_users(user_id, similarity)


# In[78]:


print("Top 5 similar users for user_id:",user_id," are: ",similar_users)


# In[79]:


print("Top 5 productID recommended are: ",
      getting_top_5_recommendations_based_on_users(user_id, similar_users, similarity))


# In[80]:


test_data.shape


# In[81]:


len(test_data.user.unique())


# In[82]:


test_data.UserId


# In[83]:


test_data.head()


# In[84]:


def recommend_products_for_user(userId, similarity_matrix):
    similar_users = getting_top_5_similar_users(user_id, similarity_matrix)
#     print("Top 5 similar users for user_id:",user_id," are: ",similar_users)
    product_list = getting_top_5_recommendations_based_on_users(user_id, similar_users, similarity)
#     print("Top 5 productID recommended are: ", product_list)
    return product_list


# In[85]:


recommend_products_for_user("A2XVNI270N97GL", similarity)


# ### Conclusion
# 
# Recommender systems are a powerful technology that adds to a businesses value. Some business thrive on their recommender systems. It helps the business by creating more sales and it helps the end user buy enabling them to find items they like.
