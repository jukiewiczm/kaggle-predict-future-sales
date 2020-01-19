
# coding: utf-8

# # Basic overview of the datasets

# ## Loading datasets

# In[1]:


import numpy as np
import pandas as pd
from functools import reduce
from bokeh.plotting import figure, output_notebook, show
from bokeh.models import ColumnDataSource
from bokeh.layouts import row


# In[2]:


train_path = "../data/sales_train.csv.gz"
test_path = "../data/test.csv.gz"
items_path = "../data/items.csv"
items_categories_path = "../data/item_categories.csv"
shops_path = "../data/shops.csv"


# In[3]:


train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
items = pd.read_csv(items_path)
item_categories = pd.read_csv(items_categories_path)
shops = pd.read_csv(shops_path)


# ## Taking a look at the datsets

# In[4]:


train.head(3)


# In[5]:


test.head(3)


# In[6]:


items.head(3)


# In[7]:


item_categories.head(3)


# In[8]:


shops.head(3)


# ## Count/missing data statistics

# In[9]:


train[['date_block_num', 'item_price', 'item_cnt_day']].describe()


# Interesting insights:
# * Item price seems to have negative value, perhaps it is a missing value
# * Item count per day also have negative values, there could be some missing values (which would be a bit strange tbh, because one could simply not insert the row). The other option is that these are returns of the product. Not sure if these should be counted in.

# In[10]:


train.isna().apply(lambda col: col.sum())


# No NaN values, which means some of these negative values may actually be missing ones.

# Let's check if there are additional information for all items/categories/shops

# In[11]:


# Utility function for counts checking.
def calculate_counts(column_name, ref_sets, extension_set):
    ref_sets_ids = [set(ref[column_name]) for ref in ref_sets]
    ext_ids = set(extension_set[column_name])
    
    refs_union = reduce(lambda s1, s2: s1 | s2, ref_sets_ids)
    
    ref_counts = [len(ref) for ref in ref_sets_ids]
    ext_count = len(ext_ids)
    union_count = len(refs_union)
    intersection_count = len(ext_ids & refs_union)
    
    all_counts = ref_counts + [union_count, ext_count, intersection_count]
    res_index = ["Ref {}".format(i) for i in range(1, len(ref_sets) + 1)] +        ['Refs Union', 'Extension', 'Union x Extension']
    
    return pd.DataFrame({'Count': all_counts},
                        index=res_index)


# In[12]:


calculate_counts('item_id', [train, test], items)


# In[13]:


calculate_counts('shop_id', [train, test], shops)


# In[14]:


calculate_counts('item_category_id', [items], item_categories)


# Looks good, there are no shops/categories/items that are in the datasets but do not exist in suplementary datasets.

# In[15]:


calculate_counts('item_id', [train], test)


# There are 363 items that appear in the test set but not in the train set

# In[16]:


calculate_counts('shop_id', [train], test)


# No shops missing though.

# Number of categories per item

# In[17]:


items.groupby('item_id')['item_category_id'].count().max()


# None of the items has more than one category, which is good.

# ## Training/test sets item/shops combinations comparison

# One of the very important insights is that the test set seems to be pretty big comparing to item/shop combinations per month from training data.

# In[18]:


test.shape


# In[19]:


train_block_shop_item = pd.DataFrame(
    {'shop_item' :train['shop_id'].astype(str) + "_" + train['item_id'].astype(str)}
)
train_block_shop_item['date_block_num'] = train['date_block_num']
train_block_shop_item.groupby('date_block_num')[['shop_item']].nunique().transpose()


# This obviously comes from the fact that the test set was prepared in such a way that it contains all combinations of shops and items from the month. Probably to avoid data leaks or provide additional challenge. Nevertheless, this means that there will be a need to extend the training set the same way (add all item/shop combinations for a month) in order to obtain proper validation setup using training data (at least for the validation set).

# One of the approaches would be to set unknown combinations to 0 instead of predicting them, but that would only be reasonable if the item/shop combinations are pretty stable across months. Let's check out.

# In[20]:


item_shop_sets = train_block_shop_item.groupby('date_block_num')['shop_item'].    apply(lambda x: set(x.tolist())).tolist()

pd.DataFrame([len(old & new) / len(new) for old, new in zip(item_shop_sets, item_shop_sets[1:])]).transpose()


# Well, can't say this looks reasonable, differences are pretty huge, so this would be pretty much guessing. If the differences were around, say, 10%, I could give it a try.

# But let's not give up that quickly, perhaps the shops/items characteristics are seasonal. Let's check this on monthly basis.

# In[21]:


pd.DataFrame([len(old & new) / len(new) for old, new in zip(item_shop_sets, item_shop_sets[12:])]).transpose()


# Even worse, I guess it will be best to leave it for a model to learn.

# To complete this part:
# 
# Total number of product-shop combinations:

# In[22]:


items.shape[0]*shops.shape[0]


# Product-shop combinations in train set

# In[23]:


train_combinations = train[['shop_id', 'item_id']].drop_duplicates()
train_combinations.shape[0]


# ## Basic visualizations

# In[24]:


# some additional preprocessing of the data for easier analysis
unique_dates = pd.DataFrame({'date': train['date'].drop_duplicates()})
unique_dates['date_parsed'] = pd.to_datetime(unique_dates.date, format="%d.%m.%Y")
unique_dates['day'] = unique_dates['date_parsed'].apply(lambda d: d.day)
unique_dates['month'] = unique_dates['date_parsed'].apply(lambda d: d.month)
unique_dates['year'] = unique_dates['date_parsed'].apply(lambda d: d.year)

train_postproc = train.merge(unique_dates, on='date').sort_values('date_parsed')


# In[25]:


# Embedding Bokeh into jupyter notebook
output_notebook()


# In[26]:


def my_figure(**options): 
    return figure(tools="hover,pan,wheel_zoom,box_zoom,reset", **options)


# ### Total sales throughout the years, month by month

# In[27]:


data = train_postproc.groupby(['year', 'month']).agg({'item_cnt_day': np.sum}).reset_index().pivot(index='month', columns='year', values='item_cnt_day')
data.plot(figsize=(12, 8))


# Generally negative trend from year by year, seasonal trends visible as well.

# ### Number of items per category

# In[28]:


data = items.groupby('item_category_id')['item_id'].count().reset_index(name='count')
data['item_category_id'] = data['item_category_id'].astype(str)
src = ColumnDataSource(data)
sorted_range = data.sort_values('count')['item_category_id'].astype(str).tolist()
TOOLTIPS = [
    ("index", "$index"),
    ("value", "@count")
]
p = my_figure(x_range = sorted_range, tooltips=TOOLTIPS, width=800, title="Number of items per category")
p.title.text_font_size = '16pt'
p.xaxis.major_tick_line_color=None
p.xaxis.major_label_text_font_size='0pt'
p.vbar('item_category_id', top='count', width=0.9, source=src)
show(p)


# In[29]:


data[['count']].describe(percentiles=np.arange(0.1, 1, 0.1)).transpose()


# In[30]:


item_category = data.nlargest(1, 'count')['item_category_id'].to_numpy()[0]
item_categories[item_categories['item_category_id'] == int(item_category)]


# ---
# **The most popular category is "Cinema - DVD"**
# 
# ---

# ### Item prices

# In[31]:


data = train[['item_price']].drop_duplicates().sort_values('item_price')
data['index'] = range(data.shape[0])
TOOLTIPS = [
    ("index", "$index"),
    ("value", "@item_price")
]
p = my_figure(tooltips=TOOLTIPS, title="Item prices", width=800)
p.title.text_font_size = '16pt'
src = ColumnDataSource(data)
src_top = ColumnDataSource(data[-50:])
p.line(x='index', y='item_price', source=src)
p.circle(x='index', y='item_price', size=3, source=src_top)
show(p)


# In[32]:


data[['item_price']].describe(percentiles=np.arange(0.1, 1, 0.1)).transpose()


# In[33]:


item_id = train_postproc[train_postproc['item_price'] == int(data['item_price'].nlargest(1).to_numpy()[0])]['item_id'].to_numpy()[0]
items[items['item_id'] == item_id]
train_postproc[train_postproc['item_id'] == item_id]


# In[34]:


items[items['item_id'] == item_id]


# ---
# **The most expensive item in the dataset is a software license to Radmin 3. I believe they sold a pack of 522 licenses at once, that's why the price is so high.**
# 
# ---

# ### Frequency of item sale

# In[35]:


data = train['item_id'].value_counts().sort_values().reset_index()
data.columns = ['item_id', 'count']
data['item_id'] = data['item_id'].astype(str)
TOOLTIPS = [
    ("index", "@x"),
    ("value", "@y")
]
p = my_figure(tooltips=TOOLTIPS, width=800, title="Frequency of items sale")
p.title.text_font_size = '16pt'
p.line(x=data.index, y=data['count'])
p.circle(x=data.index[-10:], y=data['count'][-10:], size=3)
show(p)


# In[36]:


data[['count']].describe(percentiles=np.arange(0.1, 1, 0.1)).transpose()


# In[37]:


item_id = int(data.nlargest(1, 'count')['item_id'].to_numpy()[0])
items[items['item_id'] == item_id]


# ---
# **The most frequently sold item is a plastic bag! Not really surprising**
# 
# ---

# ### Most frequently price changing items

# In[38]:


most_price_changing = train.groupby('item_id')['item_price'].nunique().nlargest(3).reset_index()


# In[39]:


data = train_postproc.    merge(most_price_changing[['item_id']], on='item_id').    sort_values('date_parsed').    groupby('item_id')['item_price'].    apply(list).    tolist()


# In[40]:


plots=[]
for series in data:
    p=my_figure()
    x = list(range(len(series)))
    p.line(x, series)
    plots.append(p)
show(row(*plots))


# Well, that's a huge variability for a product

# In[41]:


item_id = most_price_changing.loc[2, 'item_id']
items[items['item_id'] == item_id]


# ---
# **These most frequently changing products are card payments, so such variability is rather expected**
# 
# ---

# ### Total item sale given its price

# In[42]:


data = train_postproc.groupby(['item_price'])['item_cnt_day'].sum().reset_index()
data.plot(x='item_price', y='item_cnt_day', kind='scatter', figsize=(12, 8))


# ### Total sales day by day

# In[43]:


data = train_postproc.groupby(['year', 'month', 'day']).agg({'item_cnt_day': np.sum}).unstack('year')
data.plot(figsize=(12, 8))


# I believe there are some weekday/weekend trends as well.

# ### Total sales by shop

# In[44]:


data = train_postproc.groupby('shop_id')['item_cnt_day'].sum().sort_values().reset_index()
data.plot(x='shop_id', y='item_cnt_day', kind='bar', figsize=(12, 8))


# ### Total sales by shop split by year

# In[45]:


data = train_postproc.groupby(['year', 'shop_id'])['item_cnt_day'].sum().sort_values().reset_index().    pivot(index='shop_id', columns='year', values='item_cnt_day')
data.plot(kind='bar', figsize = (12, 12))


# ### Frequency of item sale quantities

# In[46]:


data = train_postproc.groupby(['year', 'month', 'item_id'])['item_cnt_day'].count().reset_index()['item_cnt_day'].value_counts()
data.plot(figsize=(12, 8))


# In[47]:


data.describe()


# **E.g. single item was sold over 50k times, but a pack of 400 items at once is sold very rarely.**
