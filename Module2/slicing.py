# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 11:52:34 2016

@author: DMalygin
"""
#%%
import pandas as pd
df = pd.read_csv('./Datasets/direct_marketing.csv')
df

#%%
df.recency
df['recency']
df[['recency']]
df.recency < 7
df[df.recency<7]