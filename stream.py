#!/usr/bin/env python
# coding: utf-8

# In[1]:


import camelot
import pdfplumber
import os
import pandas as pd
# pip install camelot-py[cv]
# pip install opencv-python-headless
import sys
#sys.path.insert(0, './pdf_parser')
from table_utils import check_non_table
import numpy as np


# In[6]:





# In[3]:


def clean_stream_bbox(allboxes, pdffile):
    
    if allboxes == {}:
        return {}
    with pdfplumber.open(pdffile) as pdf:
        pages = pdf.pages

    allboxes_clean = {}
    for i in allboxes:
        allboxes_clean[i] = []
        for box in allboxes[i]:
            x1 = box[0]
            y1 = int(pages[i].height) - box[1] 
            x2 = box[2]
            y2 = int(pages[i].height) - box[3]
            allboxes_clean[i].append([x1, y1, x2, y2])
            
    return allboxes_clean


# In[25]:

def merge_row_stream(tables, tol_s = 2):
    tables_clean = []
    
    for table in tables:
        row_drop = []
        for i in range(1, len(table)):
            #pre_ele = len([k for k in table.iloc[i-1, :] if k.strip() != '']) 
            cur_ele = len([k for k in table.iloc[i, :] if k.strip() != '']) 

            #if pre_ele <= int(table1.shape[1]/tol_ratio) or cur_ele <= int(table1.shape[1]/tol_ratio):
            if cur_ele <= tol_s and table.shape[1] > tol_s +1:
                row_drop.append(i)
                for j in range(table.shape[1]):
                    table.iloc[i-1, j] = str(table.iloc[i-1, j]) + ' ' + str(table.iloc[i, j])

        table.drop(row_drop, axis = 0, inplace = True)
        table.index = np.arange(len(table))
        
        tables_clean.append(table)
    
    return tables_clean


def clean_table_stream(tables):
    tables_clean = []
    for table in tables:
        row_drop = []
        
        upper_b = min(5, table.shape[0])
        for i in range(upper_b):
            cur_len = max([len(k.strip()) for k in table.iloc[i, :]])
            if cur_len > 50:
                row_drop.append(i)
                
        lower_b = max(0, len(table)-5)
        for i in range(lower_b, len(table)):
            cur_len = max([len(k.strip()) for k in table.iloc[i, :]])
            if cur_len > 50:
                row_drop.append(i)
                
        table.drop(row_drop, axis = 0, inplace = True)
        table.index = np.arange(len(table))
        tables_clean.append(table)
    
    return tables_clean


def get_table_stream(pdffile, page_nums = None, remove_non_table = False):

    tables = {}
    #alltables = {}
    bboxes = {}
    
    if pdffile.endswith('.pdf'):
        
        if page_nums is None:
            with pdfplumber.open(pdffile) as pdf:
                page_nums = range(len(pdf.pages))

        for i in page_nums:
            try:
                table = camelot.read_pdf(pdffile, flavor='stream', row_tol = 9, pages=str(i+1))
                tables[i] = []
                bboxes[i] = []
                for ele in table:
                    tables[i].append(ele.df)
                    bboxes[i].append([int(k) for k in ele._bbox])
                if remove_non_table == True:
                    tables[i], bboxes[i] = check_non_table(tables[i], bboxes[i])
                    tables[i] = clean_table_stream(tables[i])
                    tables[i] = merge_row_stream(tables[i])
            except:
                continue
                    
        bboxes = clean_stream_bbox(bboxes, pdffile)

        tables_clean = {}
        bboxes_clean = {}
        for i in tables:
            if tables[i] != []:
                tables_clean[i] = tables[i]
                bboxes_clean[i] = bboxes[i]
            
    return tables_clean, bboxes_clean



# convert to the format for camelot

def tocamelot(boxes, pdffile):
    
    with pdfplumber.open(pdffile) as pdf:
        pages = pdf.pages
        height = int(pages[0].height)
        
    results = {}
    for page in boxes:
        box = boxes[page]
        
        page_box = []
        
        for ele in box:
            x1 = str(int(ele[0]))
            y1 = str(height - int(ele[1]))
            x2 = str(int(ele[2]))
            y2 = str(height - int(ele[3]))
            page_box.append([x1 + ',' + y1 + ',' + x2 + ',' + y2])
                    
        results[page] = page_box
    return results

def tocamelot_page(boxes, page):
    page_box = []
    height = int(page.height)
    for ele in boxes:
        x1 = str(int(ele[0]))
        y1 = str(height - int(ele[1]))
        x2 = str(int(ele[2]))
        y2 = str(height - int(ele[3]))
        page_box.append([x1 + ',' + y1 + ',' + x2 + ',' + y2])    
    return page_box



# In[40]:


def get_table_stream_bbox(pdffile, boxes, remove_non_table = False):
    
    tables = {}
    bboxes_clean = {}
  
    if pdffile.endswith('.pdf'):
        with pdfplumber.open(pdffile) as pdf:
            pages = pdf.pages
        for page in boxes:
            tables[page] = []
            areas = tocamelot_page(boxes[page], pages[page])
            
            for area in areas:
                table = camelot.read_pdf(pdffile, flavor='stream', pages=str(page+1), row_tol = 9, table_areas = area) # rol_tol = 12
                if table[0] == []:
                    continue
                tables[page].append(table[0].df)
            if remove_non_table == True:
                tables[page], bboxes_clean[page] = check_non_table(tables[page], boxes[page])
                tables[page] = clean_table_stream(tables[page])
                tables[page] = merge_row_stream(tables[page])
    
        #alltables[file] = tables
            
    return tables, bboxes_clean


# In[ ]:



def save_tables(tables, outpath = '../results/'):
    
    if tables == {}:
        return None
    
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    
    for file in alltables:
        if file.endswith('.pdf'):
            savefile = os.path.join(outpath, file[:-4] +'_tables.xlsx')
            tables = alltables[file]
            pages = len(tables)
            with pd.ExcelWriter(savefile) as writer:
                for page_n in range(pages):    
                    if tables[page_n] != []:
                        for idx in range(len(tables[page_n])):
                            table = tables[page_n][idx]
                            table.to_excel(writer, sheet_name='page_' + str(page_n + 1) + '_id_' + str(idx + 1))
# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




