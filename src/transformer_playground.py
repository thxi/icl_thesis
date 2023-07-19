#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import sys

sys.path.append("..")


# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# In[6]:


from tqdm.auto import tqdm


# ## tf

# https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html

# In[7]:


sentence = 'Life is short, eat dessert first'

dc = {s:i for i,s in enumerate(sorted(sentence.replace(',', '').split()))}
print(dc)


# In[8]:


import torch

sentence_int = torch.tensor([dc[s] for s in sentence.replace(',', '').split()])
print(sentence_int)


# In[11]:


torch.manual_seed(123)
embed = torch.nn.Embedding(6, 16)
embedded_sentence = embed(sentence_int).detach()

print(embedded_sentence)
print(embedded_sentence.shape)


# In[12]:


torch.manual_seed(123)

d = embedded_sentence.shape[1]

d_q, d_k, d_v = 24, 24, 28

W_query = torch.nn.Parameter(torch.rand(d_q, d))
W_key = torch.nn.Parameter(torch.rand(d_k, d))
W_value = torch.nn.Parameter(torch.rand(d_v, d))


# In[16]:


W_query.shape, embedded_sentence.shape


# In[17]:


x_2 = embedded_sentence[1]
query_2 = W_query.matmul(x_2)
key_2 = W_key.matmul(x_2)
value_2 = W_value.matmul(x_2)

print(query_2.shape)
print(key_2.shape)
print(value_2.shape)


# In[32]:


queries = W_query.matmul(embedded_sentence.T).T
keys = W_key.matmul(embedded_sentence.T).T
values = W_value.matmul(embedded_sentence.T).T

print("queries.shape:", queries.shape)
print("keys.shape:", keys.shape)
print("values.shape:", values.shape)


# for a single observation

# In[19]:


x_2 = embedded_sentence[1]
query_2 = W_query.matmul(x_2)
key_2 = W_key.matmul(x_2)
value_2 = W_value.matmul(x_2)


# In[21]:


omega_2 = query_2.matmul(keys.T)
print(omega_2)


# In[23]:


import torch.nn.functional as F

attention_weights_2 = F.softmax(omega_2 / d_k**0.5, dim=0)
print(attention_weights_2)


# In[28]:


W_value.shape


# In[29]:


values.shape


# In[25]:


context_vector_2 = attention_weights_2.matmul(values)

print(values.shape)
print(context_vector_2.shape)
print(context_vector_2)


# ![image.png](attachment:4538a0d8-aaa1-4e7a-95e5-e61af44d048a.png)

# In[66]:


# attention_weights = keys.T.matmul(queries).T

unnorm_attention_weights = queries.matmul(keys.T)


# In[67]:


unnorm_attention_weights


# In[70]:


attention_weights = F.softmax(unnorm_attention_weights / d_k**0.5, dim=1)


# In[74]:


attention_weights


# In[78]:


for j in range(6):
    a = F.softmax(unnorm_attention_weights[j] / d_k**0.5, dim=0)
    print(a)


# In[83]:


z = attention_weights.matmul(values)


# In[84]:


print(z.shape)


# export weights to file

# In[86]:


W_query


# In[ ]:




