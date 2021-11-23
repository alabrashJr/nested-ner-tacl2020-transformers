#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# In[2]:


df = pd.read_excel("litlebank_BILOU_POS_LEMMA.xlsx", engine="openpyxl")

# In[3]:


df.head()

# In[4]:


import re
import random

# In[6]:


ids = df["id"].apply(lambda x: re.sub("_[0-9]+$", "", x)).unique()
train_ids = random.sample(ids.tolist(), k=int(len(ids) * .8))
test_ids = random.sample(list(set(ids.tolist()) - set(train_ids)), k=int(len(ids) * .10))
valid_ids = list(set(ids.tolist()) - set(train_ids) - set(test_ids))

train_df = df[df["id"].apply(lambda x: re.sub("_[0-9]+$", "", x)).isin(train_ids)]
test_df = df[df["id"].apply(lambda x: re.sub("_[0-9]+$", "", x)).isin(test_ids)]
valid_df = df[df["id"].apply(lambda x: re.sub("_[0-9]+$", "", x)).isin(valid_ids)]

# In[7]:


len(train_df) + len(test_df) + len(valid_df), len(df), len(train_df), len(test_df), len(valid_df)


# In[43]:


class Category():
    def __init__(self, cat, start, end=-1):
        self.start = start
        self.end = end
        self.cat = cat

    def isComplete(self):
        return self.end > -1

    def setEnd(self, end):
        self.end = end

    def toDict(self):
        return {'category': self.cat, 'start': self.start, 'end': self.end}

    def toPyramid(self):
        return "{0},{1} {2}".format(self.start, self.end, self.cat)


# In[53]:


available_cats = {q.split('-')[1] for x in df['BILOU'].unique() for q in x.split('|') if q != 'O'}
available_cats


# In[54]:


def pyramid_format(items):
    completed_cats = []
    open_cats = {}
    for t in available_cats:
        open_cats[t] = []
    for item in items:
        index = item[0]
        assignments = item[1].split('|')
        for assignment in assignments:
            stat, cat = assignment.split('-')
            if stat == 'U':
                completed_cats.append(Category(cat, index, index + 1))
            elif stat == 'B':
                open_cats[cat].append(Category(cat, index))
            elif stat == 'L':
                open_cat = open_cats[cat].pop(0)
                completed_cats.append(Category(open_cat.cat, open_cat.start, index + 1))
    return "|".join(c.toPyramid() for c in completed_cats)


# In[55]:


test_v = [(i, x) for i, x in enumerate(valid_df.loc[valid_df.id == st_id, 'BILOU'].tolist()) if x != 'O']
test_v

# In[56]:


pyramid_format(test_v)

# In[57]:


with open('nested-ner-tacl2020-transformers/data/litbank/test.data', 'w') as f:
    for st_id in test_df.id.unique():
        f.write(" ".join(test_df.loc[test_df.id == st_id, 'token'].tolist()) + '\n')
        mr_tg = [(i, x) for i, x in enumerate(test_df.loc[test_df.id == st_id, 'BILOU'].tolist()) if x != 'O']
        pyr_format = pyramid_format(mr_tg)
        f.write(pyr_format + '\n')
        f.write('\n')

# In[60]:


# get_ipython().system('head -n 20 nested-ner-tacl2020-transformers/data/litbank/test.data')

# In[65]:


with open('nested-ner-tacl2020-transformers/data/litbank/train.data', 'w') as f:
    for st_id in train_df.id.unique():
        f.write(" ".join(train_df.loc[train_df.id == st_id, 'token'].tolist()) + '\n')
        #         f.write(",".join(train_df.loc[train_df.id==st_id,'XPOS'].tolist())+'\n')
        mr_tg = [(i, x) for i, x in enumerate(train_df.loc[train_df.id == st_id, 'BILOU'].tolist()) if x != 'O']
        pyr_format = pyramid_format(mr_tg)
        f.write(pyr_format + '\n')
        f.write('\n')

# In[73]:


train_df.loc[train_df.id == train_df.id.unique()[5], 'token'].tolist()[:2]

# In[69]:


# get_ipython().system('head -n 21 nested-ner-tacl2020-transformers/data/litbank/train.data')

# In[66]:


with open('nested-ner-tacl2020-transformers/data/litbank/dev.data', 'w') as f:
    for st_id in valid_df.id.unique():
        f.write(" ".join(valid_df.loc[valid_df.id == st_id, 'token'].tolist()) + '\n')
        #         f.write(",".join(valid_df.loc[valid_df.id==st_id,'XPOS'].tolist())+'\n')
        mr_tg = [(i, x) for i, x in enumerate(valid_df.loc[valid_df.id == st_id, 'BILOU'].tolist()) if x != 'O']
        pyr_format = pyramid_format(mr_tg)
        f.write(pyr_format + '\n')
        f.write('\n')

# In[67]:


# get_ipython().system('head -n 20 nested-ner-tacl2020-transformers/data/litbank/dev.data')

