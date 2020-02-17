#!/usr/bin/env python
# coding: utf-8

# In[1]:


from stream import get_table_stream, get_table_stream_bbox
from projection import get_table_project, get_table_project_bbox
from table_utils import tocamelot
import table_utils


import os
import pandas as pd


# In[ ]:





# In[2]:


# with bbox
def text_table(file, parse_method = 'stream', page_nums = [], bbox = None):
    
    if not page_nums:
        page_nums = None
        
    tables = {}
    if not file.lower().endswith('pdf'):
        print('This is not a pdf file')
        return tables
    
    if parse_method == 'stream':
        newboxes = tocamelot(bbox, file)
        tables = get_table_stream_bbox(file, newboxes, remove_non_table = True)
    elif parse_method == 'projection':
        tables = get_table_project_bbox(file, bbox, remove_non_table = True)
    else:
        print('Choose a valid parse method')
        return tables
    return tables


# In[ ]:





# In[3]:


def text_table_nobox(file, parse_method = 'stream', page_nums = [], bbox = None):
    
    tables = {}
    table_boxes = {}
    
    if not page_nums:
        page_nums = None
        
    if not file.lower().endswith('pdf'):
        print('This is not a pdf file')
        return tables
    
    if parse_method == 'stream':
        tables, table_boxes = get_table_stream(file, page_nums = page_nums, remove_non_table = True)
    elif parse_method == 'projection':
        tables, table_boxes = get_table_project(file, page_nums = page_nums, remove_non_table = True)
    else:
        print('Choose a valid parse method')
        return tables, table_boxes
    return tables, table_boxes


# In[ ]:





# In[4]:


def save_tables(tables, outpath = './results/'):
    
    if tables == {}:
        return None
    
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    
    savefile = os.path.join(outpath, 'result_tables.xlsx')
 
    with pd.ExcelWriter(savefile) as writer:
        for page_n in tables:    
            if tables[page_n] != []:
                for idx in range(len(tables[page_n])):
                    table = tables[page_n][idx]
                    table.to_excel(writer, sheet_name='page_' + str(page_n + 1) + '_id_' + str(idx + 1))


# In[ ]:





# In[5]:


## test


# In[13]:

if __name__ == '__main__':
    file = './samples/pdf5.pdf'
    tables1, table_boxes1 = text_table_nobox(file, 'stream')
    tables2, table_boxes2 = text_table_nobox(file, 'projection')


# In[ ]:





# In[ ]:





# In[ ]:




