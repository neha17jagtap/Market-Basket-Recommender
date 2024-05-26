#!/usr/bin/env python
# coding: utf-8

# In[4]:


#Import all relevant libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from mlxtend.preprocessing import TransactionEncoder
import warnings


# In[5]:


# Load your retail dataset from Excel
retail_df = pd.read_excel("C:/Users/vysha/Downloads/cleaned_dataset.xlsx")

# Display the first few rows of the DataFrame
retail_df.head(10)


# In[6]:


#Taking only UK into consideration 
# Group by 'InvoiceNo' and 'Description', sum 'Quantity', unstack, reset index, fill NaN with 0, set 'InvoiceNo' as index
basket = retail_df[retail_df['Country'] =="United Kingdom"]        .groupby(['InvoiceNo', 'Description'])["Quantity"]        .sum().unstack()        .reset_index().fillna(0)        .set_index("InvoiceNo")

basket.head()


# In[7]:


#Create function to hot encode the values
def encode_values(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

#Apply function to data
basket_encoded = basket.applymap(encode_values)

basket_encoded


# In[8]:


#filter for only invoices with 2 or more items
basket_filtered = basket_encoded[(basket_encoded > 0).sum(axis=1) >= 2]

basket_filtered


# In[9]:



# Suppress DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Generate the frequent itemsets
frequent_itemsets = apriori(basket_filtered, min_support=0.02, use_colnames=True).sort_values("support",ascending=False)
frequent_itemsets.head(10)


# In[10]:


#Apply association rules
# 1- Implement a technique that uses Association Rule
assoc_rules = association_rules(frequent_itemsets, metric="lift",min_threshold=1).sort_values("lift",ascending=False).reset_index(drop=True)
assoc_rules


# In[11]:


def RecommendItems(CurItemOfInterest, CustomerID, Country, Time, Date, top_n=10):
   
    target_item = CurItemOfInterest

    # Filter association rules for those containing the target item in the antecedents
    target_rules = assoc_rules[assoc_rules['antecedents'].apply(lambda x: target_item in x)]

    # Print the target rules for debugging
    print("Target Rules:", target_rules)

    # Sort the rules by confidence or lift, depending on your preference
    sorted_target_rules = target_rules.sort_values(by='confidence', ascending=False)

    # Print the sorted target rules for debugging
    print("Sorted Target Rules:", sorted_target_rules)

    # Extract the top N recommended items from the antecedents of the sorted rules
    recommended_items = []

    for _, rule in sorted_target_rules.head(top_n).iterrows():
        recommended_items.extend(rule['consequents'])

    # Display or use the recommendations as needed
    print(f"Top {top_n} Recommendations for {target_item}: {recommended_items}")

    # You can return the recommendations or use them in any other way based on your requirements
    return recommended_items[:top_n]

CurItemOfInterest = "WHITE HANGING HEART T-LIGHT HOLDER"
customer_id = "17850"
country = "United_Kingdom"

top_10_recommendations = RecommendItems(CurItemOfInterest=CurItemOfInterest, CustomerID=customer_id, Country=country, Time=None, Date=None, top_n=10)


# In[12]:


##popular items
import pandas as pd


def RecommendItems(CurItemOfInterest, CustomerID, Time, Date):
    # Get popular items across all countries
    popular_items = retail_df['Description'].head(10)
    
    # Check if the current item of interest is popular overall
    if CurItemOfInterest in popular_items:
        # Remove the current item of interest from the list of popular items
        popular_items.drop(CurItemOfInterest, inplace=True)
    
    # Format items and counts as strings
    recommended_items = ["{} - {}".format(item, count) for item, count in popular_items.items()]
    
    return recommended_items

recommended_items = RecommendItems('CurItemOfInterest', 'CustomerID', 'Time', 'Date')
print('\n'.join(recommended_items))


# In[13]:


########3.Implementing collaberative filtering technique.
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
def RecommendItems(CurItemOfInterest, CustomerID, Country, Time, Date, top_n=10):
    # Filter transactions for the given country
    country_transactions = retail_df[retail_df['Country'] == Country]

    # Drop rows with missing CustomerID values
    country_transactions = country_transactions.dropna(subset=['CustomerID'])

    # Create a user-item matrix
    user_item_matrix = country_transactions.pivot_table(index='CustomerID', columns='StockCode', values='Quantity', fill_value=0)

    # Calculate user similarity using cosine similarity
    user_similarity = cosine_similarity(user_item_matrix)

    # Convert the similarity matrix into a DataFrame for easier manipulation
    user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

    # Check if the user exists in the user-item matrix
    if CustomerID not in user_item_matrix.index:
        return [f"User {CustomerID} not found in the dataset for country {Country}."]

    # Get similarity scores for the specified user
    similar_users = user_similarity_df[CustomerID]

    # Find top similar users (excluding the user itself)
    top_similar_users = similar_users.drop(CustomerID).nlargest(top_n)

    # Get items purchased by top similar users but not by the specified user
    recommended_items = user_item_matrix.loc[top_similar_users.index, user_item_matrix.loc[CustomerID] == 0]

    # Sum up the quantity of each item across similar users and sort by total quantity
    recommended_items = recommended_items.sum().sort_values(ascending=False)

    # Get top-N recommended items as a DataFrame with stock ID, description, and quantity
    recommended_items_df = pd.DataFrame({'StockCode': recommended_items.index, 'Quantity': recommended_items.values})
    recommended_items_df = recommended_items_df.merge(retail_df[['StockCode', 'Description']], on='StockCode', how='left').drop_duplicates(subset=['StockCode'])

    return recommended_items_df.head(top_n)

customer_id = 13047
country = 'United Kingdom'
recommendations = RecommendItems(CurItemOfInterest=None, CustomerID=customer_id, Country=country, Time=None, Date=None, top_n=10)

print(f"Top-10 recommendations for user {customer_id} in {country}:")
print(recommendations)


# In[ ]:





# In[ ]:





# In[ ]:




