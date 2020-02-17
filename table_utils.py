#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle

import os
import re

import pandas as pd

import pdfplumber
import camelot
import PyPDF2
import pdb

# import cv2

from scipy.stats import mode

import tempfile

import pickle


# In[ ]:





# In[ ]:


# use pdfplumber + pypdf2 to get text

def get_text_word(pdffile, page_nums = None):
    alltext = [] # format of x0, x1, top, bottom, text
    # can be 'SCHEDULE OF INVESTMENT ASSETS' or "SCHEDULE OF ASSETS"
    allwords = []
    alltables = []
    alllines = []
    allrects = []
    
    table_method = {"vertical_strategy": "text", "horizontal_strategy": "lines"}
    
    with pdfplumber.open(pdffile) as pdf:
        pages = pdf.pages
        page_n = len(pages)
        
        if page_nums is None:
            for i in range(page_n):

                text = pages[i].extract_text()
                words = pages[i].extract_words()
                tables = pages[i].find_tables(table_settings = table_method)
                lines = pages[i].lines
                rects = pages[i].rects
                alltext.append(text)
                allwords.append(words)
                alltables.append(tables)
                alllines.append(lines)
                allrects.append(rects)
        else:
            for i in page_nums:
                try:
                    text = pages[i].extract_text()
                    words = pages[i].extract_words()
                    tables = pages[i].find_tables(table_settings = table_method)
                    lines = pages[i].lines
                    rects = pages[i].rects
                    alltext.append(text)
                    allwords.append(words)
                    alltables.append(tables)
                    alllines.append(lines)
                    allrects.append(rects)
                except:
                    continue

    #text = page.extract_text()
    #text.split('\n')
    
    return alltext, allwords, alltables, alllines, allrects, pages


# this is another way to get text, sometimes they have different (but overlapping) results
def get_text_word_pypdf2(pdffile):
    pdf_in = open(pdffile, 'rb')
    pdf_reader = PyPDF2.PdfFileReader(pdf_in)
    totalpage = pdf_reader.getNumPages()
    
    alltext = []
    for page_n in range(totalpage):
        page = pdf_reader.getPage(page_n)
        text = page.extractText()
        alltext.append(text)
    pdf_in.close()
    return alltext


# In[ ]:


# search for keywords, and return page number

#pattern1 = r'.*schedule\s*of\s*investments*.*' # any character + schedule of investment(s) + any character
#pattern2 = r'.*schedule\s*of\s*assets*.*' # any character + schedule of asset(s) + any character

def trackpage(alltext, pages, patterns):
    page_tracker = []
    
    for i in range(len(pages)):
        if alltext[i] is not None:
            page_text = alltext[i].split('\n')   
            for line in page_text:
                for pattern in patterns: 
                    if re.match(pattern, str.lower(line)): #or re.match(pattern2, str.lower(line))
                        page_tracker.append(i)
                        break 
    return list(set(page_tracker))


# In[ ]:


def refinepages(alltext, page_tracker, keywords, thresh = 3):
    
    page_tracker_clean = []
    
    for page in page_tracker:
        wordcounter = 0
        page_text = alltext[page].split('\n')   
        for line in page_text: 
            if wordcounter < thresh:
                for word in keywords:
                    if word in str.lower(line):
                        wordcounter += 1
                        if wordcounter >= thresh:
                            page_tracker_clean.append(page)
                            break
            else:
                break
    return page_tracker_clean


# In[ ]:


def refinepages_1(alltext, page_tracker, thresh = 0.1):
    
    page_tracker_clean = []
    
    for page in page_tracker:
        digits = 0
        chars = 0
        for text in alltext[page]:
            cur_digit = np.sum([k.isdigit() for k in text])
            cur_char = len(text)
            digits += cur_digit
            chars += cur_char
        if digits/chars > thresh:
            page_tracker_clean.append(page)

    return page_tracker_clean


# In[ ]:


# check rotation
# rotation indicator: in a few rows,  many single chars appear, 
# use char_counter method

# 0 = normal, 1 = rotated


# use to count short chars on a single page
# use in "checkrotation" function 
def findshortchars(text, threshold = 3):
    char_counter = 0
    if text is not None:
        for ele in text.split('\n'): 
            if 0<len(ele.strip()) <= threshold:
                char_counter +=1
    return char_counter


def checkrotation(alltext, pages, threshold = 3, mincount = 9, ratio = 5):
    
    rotation = {}
    for page in pages:
        rot_cur_page = 0
        char_counter = findshortchars(alltext[page], threshold = threshold)
        row_len = len(alltext[page].split('\n'))
        if char_counter >= mincount and row_len/char_counter < ratio:
            rot_cur_page = 1
        rotation[page] = rot_cur_page

        
    # can make the assumption that, if more than half is rotated, all the tables are actually rotated?. 
    if np.sum([k ==1 for k in rotation.values()]) > 0.5*(len(rotation.values())):
        for page in rotation.keys():
            rotation[page] = 1 

    return rotation


# In[ ]:


# a function to group close lines

def grouplines(cols, min_sep):
    grouped = []
    i = 0
    while i < len(cols):
        checked = [i]
        cur_group = [cols[i]]
        j = i+1
        while j < len(cols):
            if cols[j] - cols[i] <= min_sep:
                cur_group.append(cols[j])
                checked.append(j)
                i = j
                j = j+1
            else:
                j +=1
        grouped.append(cur_group)
        i = checked[-1] + 1

    cols_clean = []
    for i in range(len(grouped)):
        #col = np.mean(grouped[i])
        #col = np.max(grouped[i])
        col = np.min(grouped[i])
        cols_clean.append(int(col))
        
    return cols_clean


# In[ ]:


## function to find edge candidates

# col_max = 10: pick top 10 candidates
# col_sep = 10: if two col-lines are within 30, merge them as 1 line
# row_max and row_sep have similar definitions, but for rows 

def find_edges(allwords, pages, page, rotation, col_max=10, row_max=10, col_sep=30, row_sep = 10):
    
    # 0 - normal, 1 - rotated
    
    page_w = int(pages[page].width)
    page_h = int(pages[page].height)
    
    left_x = []
    top_y = []
    for word in allwords[page]:
        left_x.append(int(word['x0']))
        ##left_x.append(int(word['x1']))##
        top_y.append(int(word['bottom']))
        ##top_y.append(int(word['top']))##
    
    #fig, ax = plt.subplots(1, 2)
    # ax0 is used to find cols, ax1 is used to find tabel edges
    if rotation[page] == 0: 
        #c, d, _ = plt.hist(top_y, bins = range(0, page_h, 5)) #rows
        c, d = np.histogram(top_y, bins=np.arange(0, page_h, 5))
        #a, b, _ = plt.hist(left_x, bins = range(0, page_w, 5)) #cols
        a, b = np.histogram(left_x, bins=np.arange(0, page_w, 5))
    else: 
        #a, b, _ = plt.hist(top_y, bins = range(0, page_h, 5)) #cols
        a, b = np.histogram(top_y, bins=np.arange(0, page_h, 5))
        #c, d, _ = plt.hist(left_x, bins = range(0, page_h, 5)) #rows
        c, d = np.histogram(left_x, bins=np.arange(0, page_w, 5))
    
    
    pick_col_indx = np.argsort(a)[::-1][:col_max]
    cols = sorted(b[pick_col_indx])
    
    pick_row_indx = []
    for i in range(len(c)):
        if c[i] > row_max:
            pick_row_indx.append(i)
            
    rows = [d[k] for k in pick_row_indx]
    rows.sort()
    
    # clean cols, group them, then find mean/max
    cols_clean = grouplines(cols, col_sep)
    rows_clean = grouplines(rows, row_sep)
    
    ############### prepare cols for camelot

    # first one is the edge, skip that
    
    col_use = ', '.join([str(cols_clean[k]) for k in range(1, len(cols_clean))])
    col_use = [col_use]
    

    row_use = ', '.join([str(rows_clean[k]) for k in range(1, len(rows_clean))])
    #row_use = ', '.join([str(rows_clean[k]) for k in range(0, len(rows_clean))])
        
    row_use = [row_use] 

    
    
    return col_use, row_use


# In[ ]:


# find edges for all pages

def find_edge_all(allwords, pages, page_tracker, rotation, col_max=10, row_max = 10, col_sep=30, row_sep = 10):
    
    all_results = []
    for page in page_tracker:
        col_use, row_use = find_edges(allwords, pages, page, rotation, col_max, row_max, col_sep, row_sep)
        
        if row_use == ['']:
            row_use = ['30' + ', ' + str(int(pages[page].height))]
            
        all_results.append([page, col_use, row_use])
    return all_results

# all_results have: [page_n, col_edge, page_edge]


# In[ ]:


# based on the edges for all pages, find the most common pattern as this doc's "table template"

def find_comon_edge(all_results):
    cols = [result[1] for result in all_results]
    rows = [result[2] for result in all_results]
    
    # cols
    common_cols = []
    for result in cols:
        if len(result[0]) == 1:
            common_cols.append(1)
        else:
            common_cols.append(len(result[0].split(', ')))  
    common_col_n = mode(common_cols)[0][0]

    picked_col = []
    for result in cols:
        if len(result[0]) > 1 and len(result[0].split(', ')) == common_col_n:
            picked_col.append(result[0].split(', '))
            
    final_cols = []
    for i in range(common_col_n):
        temp_col = []
        for ele in picked_col:
            temp_col.append(int(ele[i]))

        final_cols.append(mode(temp_col)[0][0])
        #final_cols.append(np.median(temp_col))
    final_cols = list(set(final_cols))
    final_cols.sort()
    #final_cols = [', '.join(str(k) for k in final_cols)]
    
    
    # rows
    
    row_top = []
    row_bottom = []
    
    for result in rows: 
        if len(result[0]) == 1:
            row_top.append(int(result[0]))
            row_bottom.append(int(result[0]))
        elif len(result[0]) == 0:
            continue
        else:
            row_top.append(int(result[0].split(', ')[0]))
            row_bottom.append(int(result[0].split(', ')[-1]))
            
    final_rows = []
    
    final_rows.append(min(row_top))
    final_rows.append(max(row_bottom))
 
    #final_rows = [', '.join(str(k) for k in final_rows)]
    
    if final_rows[0] == final_rows[-1]:
        final_rows[0] = 30
    
    return final_cols, final_rows


# In[ ]:


# find table edges based on final_rows

def find_table_edge(final_rows, pages, page, rotation, top_tol = 15, bottom_tol = 10):
    
    #final_rows = final_rows[0].split(', ')
    #rows = [int(k) for k in final_rows]
    
    rows = final_rows
    
    table_edge_all = []
    for ele in page: 
        page_w = int(pages[ele].width)
        page_h = int(pages[ele].height)
        
        if rotation[ele] == 0: 
            table_left_edge = 0
            table_top_edge = rows[0] - top_tol # 10 is to consider tolenrance 
            table_right_edge = page_w
            table_bottom_edge = rows[-1] + bottom_tol # str(0)
            
            if table_bottom_edge < page_h - 100:
                table_bottom_edge = page_h - 15
        else:
            table_left_edge = 0
            table_top_edge = rows[0] - top_tol
            table_right_edge = page_h
            table_bottom_edge = rows[-1] + bottom_tol # str(0)
            
            if table_bottom_edge < page_w - 100:
                table_bottom_edge = page_w - 15
 
        #table_edge = ', '.join([table_left_edge, table_top_edge, table_right_edge, table_bottom_edge])
        #table_edge = [table_edge]
        table_edge = [table_left_edge, table_top_edge, table_right_edge, table_bottom_edge]
        table_edge_all.append(table_edge)
    
    return table_edge_all


# In[ ]:


### visualize col and row projection: optional
def showprojection(allwords, pages, page, rotation):
    
    # 0 - normal, 1 - rotated
    
    page_w = int(pages[page].width)
    page_h = int(pages[page].height)
    
    left_x = []
    top_y = []
    for word in allwords[page]:
        left_x.append(int(word['x0']))
        ##left_x.append(int(word['x1']))##
        top_y.append(int(word['bottom']))
        ##top_y.append(int(word['top']))##
    
    fig, ax = plt.subplots(1, 2)
    # ax0 is used to find cols, ax1 is used to find tabel edges
    if rotation[page] == 0: 
        c, d, _ = ax[1].hist(top_y, bins = range(0, page_h, 5), orientation = 'horizontal') #rows
        ax[1].invert_yaxis()
        #c, d = np.histogram(top_y, bins=np.arange(0, page_h, 5))
        a, b, _ = ax[0].hist(left_x, bins = range(0, page_w, 5)) #cols
        #a, b = np.histogram(left_x, bins=np.arange(0, page_w, 5))
    else: 
        a, b, _ = ax[0].hist(top_y, bins = range(0, page_h, 5)) #cols
        #a, b = np.histogram(top_y, bins=np.arange(0, page_h, 5))
        c, d, _ = ax[1].hist(left_x, bins = range(0, page_h, 5), orientation = 'horizontal') #rows
        ax[1].invert_yaxis()
        #c, d = np.histogram(left_x, bins=np.arange(0, page_w, 5))
    ax[0].set_xlabel('column projections')
    ax[1].set_xlabel('row projections')


# In[ ]:





# In[ ]:





# In[ ]:


# post-processing functions: if page is rotated, correct it

def correctrotation(pdffile, allwords, alllines, allrects, rotation):
    
    pdf_writer = PyPDF2.PdfFileWriter()
    pdf_reader = PyPDF2.PdfFileReader(pdffile)
    
    pages = list(rotation.keys())
    
    # write all rotated pages into a temporary file 
    
    rotated = []
    for i in range(len(pages)):
        page_n = pages[i]
        if rotation[page_n] == 1: 
            rotated.append(i)
            page = pdf_reader.getPage(page_n).rotateClockwise(90)
            pdf_writer.addPage(page)
    
    if rotated == []:
        return allwords, alllines, allrects
    
    with tempfile.NamedTemporaryFile(delete=False) as fp:
        pdf_writer.write(fp)
        fp.seek(0)
        pdf = pdfplumber.open(fp.name)
        
        for i in rotated:
            page_n = pages[i]
            page = pdf.pages[i]
            words = page.extract_words()
            allwords[page_n] = words
            
            lines = page.lines
            alllines[page_n] = lines
            
            rects = page.rects
            allrects[page_n] = rects
        
    return allwords, alllines, allrects


# In[ ]:


# post-processing functions: find rows and group words
# row_tol: used to consider font variance in a row
# col_tol: if two words are within col_tol, connect them. 

def groupwords(allwords, row_tol = 1.5, col_tol = 6):
    
    allwords_clean = []
    
    for words in allwords:
        
        if words == []:
            allwords_clean.append([])
            continue
        
        # find rows first
        words[0]['row'] = 0
        for i in range(1, len(words)):
            if abs(words[i]['top'] - words[i-1]['top']) <= row_tol or                 abs(words[i]['bottom'] - words[i-1]['bottom']) <= row_tol:
                words[i]['row'] = words[i-1]['row']
            else:
                words[i]['row'] = words[i-1]['row'] + 1
                
        # then group words
        words_clean = []
        i = 0
        while i < len(words):

            x0 = words[i]['x0']  # need some tolerence?
            x1 = words[i]['x1']
            text = words[i]['text']
            top = words[i]['top']
            bottom = words[i]['bottom']
            row = words[i]['row']

            j = i + 1
            
            while j < len(words) and words[j]['row'] == words[j-1]['row'] and                 abs(words[j]['x0'] - words[j-1]['x1']) <= col_tol:
                x0 = min(x0, words[j-1]['x0'])
                x1 = max(x1, words[j]['x1'])
                text = text + ' ' + words[j]['text']
                j = j + 1
            
            # do some cleaning for $ sign
            #if words[j-1]['text'].strip() == '$':
                #x1 = min(x1, words[j-2]['x1'])
                #text = text.strip().strip('$').strip()
            
            words_clean.append({'x0': int(x0), 'x1': int(x1), 'top': int(top), 'bottom': int(bottom), 
                                 'text': text, 'row': int(row)})
            i = j
          
        allwords_clean.append(words_clean)
    
    return allwords_clean


# In[ ]:


def refine_cols(alllines, allrects, col_use, allpages, pages, thresh = 2, edge_l = 0, edge_r = 70, col_tol = 30):
    
    col_use_refine = []
    
    for page in allpages:
        x = {}
        
        '''
        if alllines[page] !=[]: 
            for ele in alllines[page]:
                if ele['x0'] not in x and ele['x0'] == ele['x1']:
                    x[ele['x0']] = 1
                if ele['x0'] in x and ele['x0'] == ele['x1']:
                    x[ele['x0']] += 1  
                    
        '''
        if alllines[page] !=[]: 
            for ele in alllines[page]:
                if ele['x0'] not in x:
                    x[ele['x0']] = 1
                else:
                    x[ele['x0']] += 1
                if ele['x1'] not in x:
                    x[ele['x1']] = 1
                else:
                    x[ele['x1']] += 1
        
        if allrects[page] !=[]:           
            for ele in allrects[page]:
                if ele['x0'] not in x:
                    x[ele['x0']] = 1
                else:
                    x[ele['x0']] += 1
                if ele['x1'] not in x:
                    x[ele['x1']] = 1
                else:
                    x[ele['x1']] += 1   

        cols = []
        if x != {}:
            for ele in x.items():
                if ele[1] >= thresh:
                    cols.append(int(ele[0]))
            cols.sort()
       
        #print(page, cols)   
        if cols != []:
            #col_use_1 = [edge_l] + col_use + [col_use[-1] + edge_r]
            
            col_use_1 = [edge_l] + col_use + [int(pages[page].width)]
            
            # clean cols first 
            cols_clean = []
            for col_add in cols:
                can_add = 1
                for ele in col_use_1:
                    if abs(col_add - ele) <= 15:
                        can_add = 0
                        break
                if can_add and col_add not in cols_clean:
                    cols_clean.append(col_add)
            
            #print(cols, '\n', cols_clean)
            
            col_need_add = []
            for i in range(len(col_use_1)-1):
                col_left = []
                for col_add in cols_clean:
                    if col_use_1[i] <= col_add <= col_use_1[i+1]:
                        col_left.append(col_add)
                    if col_add > col_use_1[i+1]:
                        break
                if col_left != []:
                    if len(col_left) == 1:
                        if (col_left[0] - col_use_1[i]) > col_tol or (col_use_1[i+1] - col_left[0]) > col_tol:
                            col_need_add.append(col_left[0])
                    if len(col_left) > 1:
                        if (col_left[0] - col_use_1[i]) > col_tol and (col_use_1[i+1] - col_left[-1]) > col_tol:
                            col_need_add.append(col_left[0])
                        if (col_left[0] - col_use_1[i]) <= col_tol and (col_left[-1] - col_use_1[i]) > col_tol:
                            col_need_add.append(col_left[-1])
                
            cur_col_use = col_use + col_need_add
            cur_col_use.sort()
           
        else: 
            cur_col_use = col_use
            
        col_use_refine.append(cur_col_use)
    
   
    for k in range(len(col_use_refine)):  
        if col_use_refine[k][0] < col_use[0]:
            col_use_refine[k] = col_use_refine[k][1:]

   
    #print(col_use_refine)
    ## now find the common pattern
    
    new_col_vals = []
    for i in range(len(col_use)-1):
        cur_add_vals = []
        for cols in col_use_refine:
            for col in cols[1:]:
                if col_use[i] < col < col_use[i+1]:
                    cur_add_vals.append(col)
                if col > col_use[i+1]:
                    break
        cur_add_vals = np.unique(cur_add_vals)
        if len(cur_add_vals) > 1:
            new_col_vals.append(sorted(cur_add_vals)[1]) ## pick the second small one
            #new_col_vals.append(min(cur_add_vals)+10)
        if len(cur_add_vals) == 1:
            new_col_vals.append(cur_add_vals[0])
            
    ## deal with the last element
    last_add = []
    for cols in col_use_refine:
        for col in cols[1:]:
            if col > col_use[-1]:
                last_add.append(col)
    
    last_add = np.unique(last_add)
    if len(last_add) > 1:
        new_col_vals.append(sorted(last_add)[1]) ## pick the second small one
    if len(last_add) == 1:
        new_col_vals.append(last_add[0])
    
        
    final_cols = col_use + new_col_vals
    final_cols.sort()
   
    return final_cols, col_use_refine


# In[ ]:


def refine_cols_clean(final_cols, col_use, col_tol = 40):
    final_col_use_clean = col_use.copy()
    for ele in final_cols[1:]:
        for i in range(len(col_use)-1):
            if col_use[i] < ele < col_use[i+1]:
                if (ele - col_use[i]) > col_tol and (col_use[i+1] - ele) > col_tol:
                    final_col_use_clean.append(ele)       
            if col_use[i] > ele:
                break
        if ele - col_use[-1] > col_tol:
            final_col_use_clean.append(ele) 
    #final_col_use_clean = list(set(final_col_use_clean))
    final_col_use_clean.sort()
    return final_col_use_clean


# In[ ]:


def refine_rows(alllines, allrects, row_use, allpages, thresh = 2, row_tol = 30):
    
    row_use_refine = []
    
    for page in allpages:
        y = {}
        
        '''
        if alllines[page] !=[]: 
            for ele in alllines[page]:
                if ele['y0'] not in y and ele['y0'] == ele['y1']:
                    y[ele['y0']] = 1
                if ele['y0'] in y and ele['y0'] == ele['y1']:
                    y[ele['y0']] += 1
                    
        '''
        
        if alllines[page] !=[]: 
            for ele in alllines[page]:
                if ele['y0'] not in y:
                    y[ele['y0']] = 1
                else:
                    y[ele['y0']] += 1
                if ele['y1'] not in y:
                    y[ele['y1']] = 1
                else:
                    y[ele['y1']] += 1
                    
        if allrects[page] !=[]:           
            for ele in allrects[page]:
                if ele['y0'] not in y:
                    y[ele['y0']] = 1
                else:
                    y[ele['y0']] += 1
                if ele['y1'] not in y:
                    y[ele['y1']] = 1
                else:
                    y[ele['y1']] += 1   

        rows = []
        if y != {}:
            for ele in y.items():
                if ele[1] >= thresh:
                    rows.append(int(ele[0]))
            rows.sort()
        
        if rows != []:
            row_need_add = []
            for ele in rows:
                if row_use[0] - ele > row_tol: 
                    row_need_add.append(ele)
            if row_need_add != []:
                row_add = max(row_need_add)
            else:
                row_add = 0
            row_use_refine.append(row_add)
        else:
            row_use_refine.append(0)
    
    if max(row_use_refine) == 0:
        final_row_use = row_use
    else:
        final_row_use = [max(row_use_refine)] + [k for k in row_use]
    
    return final_row_use


# In[ ]:


def refinetableedge(final_row_use, table_edge, max_dep = 100, row_tol = 50):
    table_edge_refine = []
    toprow = final_row_use[0]
    
    for ele in table_edge:
        
        if ele[1] - toprow > row_tol:
            if toprow > max_dep:
                toprow = 40
            table_edge_refine.append([ele[0], toprow, ele[2], ele[3]])
        else:
            if ele[1] > max_dep:
                ele[1] = 40
            table_edge_refine.append(ele)
            
    return table_edge_refine


# In[ ]:


## post-processing: get dataframe 

def preparedf(allwords, allpages, col_seperators, all_table_edges):
    
    alltables = []
    
    k = 0 
    
    print('Processing tables on each page...', end = ' ')
    for page in allpages:
    
        if k == len(allpages) - 1:
            print('Done')
        #if k == len(allpages) - 1:
            #print('%d of %d' % (k, len(allpages)), end='. ')
        #else:
            #print('%d of %d' % (k, len(allpages)), end=', ')
        
        cur_table_edge = all_table_edges[k]
        col_use = [0] + col_seperators + [cur_table_edge[2]]
        
        
        # find word in this row and col 
        picked = []
        for word in allwords[page]:
            if word['top'] >= cur_table_edge[1] and word['bottom'] <= cur_table_edge[3]:
                for i in range(len(col_use)-1):
                    if col_use[i] + 2  <= 0.5*(word['x0'] + word['x1']) <= col_use[i+1] + 2:
                        word['col'] = i              
                            
                picked.append(word)
        
        
        # col correction:
        for i in range(1, len(picked)):
            
            if picked[i]['row'] == picked[i-1]['row'] and picked[i]['col'] <= picked[i-1]['col']:
    
                if picked[i]['col'] == 0 or picked[i]['col'] == len(col_seperators):
                    picked[i]['text'] = picked[i-1]['text'] + ' ' + picked[i]['text']
                
                if 2<=i<= len(picked)-2 and picked[i-1]['col'] == picked[i-2]['col'] + 1                     and picked[i]['col']+1 == picked[i+1]['col']:
                    picked[i]['text'] = picked[i-1]['text'] + ' ' + picked[i]['text']
                    
                if 2<=i<= len(picked)-2 and picked[i-1]['col'] == picked[i-2]['col'] + 1                     and picked[i]['col']+1 < picked[i+1]['col']:
                    picked[i]['col'] += 1
                 
                # add this: 
                if 2<=i<= len(picked)-2 and picked[i-1]['col'] == picked[i-2]['col'] + 1                     and picked[i]['col']+1 > picked[i+1]['col']:
                    picked[i]['text'] = picked[i-1]['text'] + ' ' + picked[i]['text']
                # add end
                
                if 2<=i<= len(picked)-2 and picked[i-1]['col'] > picked[i-2]['col'] + 1                     and picked[i]['col']+1 == picked[i+1]['col']:
                    picked[i-1]['col'] -= 1
                    
                if 2<=i<= len(picked)-2 and picked[i-1]['col'] > picked[i-2]['col'] + 1                     and picked[i]['col']+1 < picked[i+1]['col']:
                    picked[i]['col'] += 1
                    
                # add this: 
                if 2<=i<= len(picked)-2 and picked[i-1]['col'] < picked[i-2]['col'] + 1                     and picked[i]['col']+1 == picked[i+1]['col']:
                    picked[i]['text'] = picked[i-1]['text'] + ' ' + picked[i]['text']
                # add end
        
                
        # find row and col numbers
        delta = picked[0]['row']

        for ele in picked:
            ele['row'] = ele['row']- delta

        maxrow = 0
        maxcol = 0
        for ele in picked:
            maxrow = max(maxrow, ele['row'])
            maxcol = max(maxcol, ele['col'])
            
        
        # fill the dataframe
        table_df = pd.DataFrame('', index=np.arange(maxrow+1), columns=np.arange(maxcol+1))
        for ele in picked: 
            i = ele['row']
            j = ele['col']
            table_df.iloc[i, j] = ele['text']
            
        
        
        # merge cols
        for i in range(table_df.shape[0]):
            for j in range(1, table_df.shape[1]-1):
                cur_ele = []
                next_ele = []
                for ele in picked:
                    if ele['row'] == i and ele['col'] == j:
                        cur_ele = ele
                    if ele['row'] == i and ele['col'] == j+1:
                        next_ele = ele
                    if cur_ele !=[] and next_ele !=[]:
                        break
                        
                if cur_ele !=[] and next_ele !=[]:
                    cur_left, cur_right = cur_ele['x0'], cur_ele['x1']
                    next_left, next_right = next_ele['x0'], next_ele['x1']
                    
                    if cur_right > next_left:
                        table_df.iloc[i, j] = table_df.iloc[i, j] + ' ' + table_df.iloc[i, j+1]
                        table_df.iloc[i, j+1] = ''
        
        
        # find each's edges and median line, for fine tuning 
        col_data = []
        for i in range(table_df.shape[1]):
            col_p_left = []
            col_p_right = []
            col_p_median = []
            for ele in picked:
                if ele['col'] == i:
                    col_p_left.append(ele['x0'])
                    col_p_right.append(ele['x1'])
                    col_p_median.append((ele['x0']+ele['x1'])/2)
            if col_p_left != [] and col_p_right !=[]:
                ##### line 11122
                col_p = [np.median(col_p_left), np.median(col_p_right), np.median(col_p_median)]
                ##### col_p = [np.min(col_p_left), np.median(col_p_right), np.median(col_p_median)]
                #col_m = np.median(col_p_median)
            else: 
                col_p = []
                #col_m = []
            col_data.append(col_p)
            
        #print(col_data[0])
        
        
        # merge close cols and drop empty cols
        first_drop = []
    
        for i in range(table_df.shape[1]-1):
            if col_data[i] != [] and col_data[i+1] != []:
                if col_data[i+1][0] < col_data[i][2] < col_data[i+1][1]:
                #if col_data[i+1][0] < col_data[i][0] and col_data[i+1][1] > col_data[i][1]  or \
                        #col_data[i+1][0] > col_data[i][0] and col_data[i+1][1] < col_data[i][1]:
                    first_drop.append(i)
                    table_df.iloc[:, i+1] = table_df.iloc[:, i] + ' ' + table_df.iloc[:, i+1]
                
        #print(first_drop)
        table_df.drop(first_drop, axis = 1, inplace = True)    
        table_df.columns = range(table_df.shape[1])    
        
        
        # further clean the df
        second_drop = []
        
        for i in range(table_df.shape[1]):
            #table_df[i] = [' '.join(table_df[i][k].split('\n')) for k in range(table_df.shape[0])]
            if table_df[i].isnull().all() or np.all( table_df[i] == ''):
                second_drop.append(i)
        table_df.drop(second_drop, axis = 1, inplace = True)
        table_df.columns = range(table_df.shape[1])
        
       
    
        # clean rows
        row_drop = []
        for i in range(table_df.shape[0]):
            if table_df.iloc[i, :].isnull().all() or np.all(table_df.iloc[i, :] == ''):
                row_drop.append(i)
        
        table_df.drop(row_drop, axis = 0, inplace = True)
        table_df.index = range(table_df.shape[0])
    
        # save the results
        alltables.append(table_df)
        
        k += 1
        
    return alltables


# In[ ]:


## post-processing: get dataframe 

def preparedf_single(allwords, pages, page, col_seperators, table_edge):
    
    alltables = []
    
    #print('Processing table in this area:  ', table_edge, '...', end = ' ')
        
    cur_table_edge = [0, table_edge[0], int(pages[page].width), table_edge[1]]
    col_use = [0] + col_seperators + [cur_table_edge[2]]
        
        
    # find word in this row and col 
    #picked = [word for word in allwords]
    
    picked = []
    for word in allwords:
        #if word['top'] >= cur_table_edge[1] and word['bottom'] <= cur_table_edge[3]:
        if word['top'] >= cur_table_edge[1] and word['bottom'] <= cur_table_edge[3]:
            for i in range(len(col_use)-1):
                if col_use[i] + 2  <= 0.5*(word['x0'] + word['x1']) <= col_use[i+1] + 2:
                    word['col'] = i              

            picked.append(word)

        
    # col correction:
    for i in range(1, len(picked)):
            
        if picked[i]['row'] == picked[i-1]['row'] and picked[i]['col'] <= picked[i-1]['col']:
    
            if picked[i]['col'] == 0 or picked[i]['col'] == len(col_seperators):
                picked[i]['text'] = picked[i-1]['text'] + ' ' + picked[i]['text']
                
            if 2<=i<= len(picked)-2 and picked[i-1]['col'] == picked[i-2]['col'] + 1                 and picked[i]['col']+1 == picked[i+1]['col']:
                picked[i]['text'] = picked[i-1]['text'] + ' ' + picked[i]['text']
                    
            if 2<=i<= len(picked)-2 and picked[i-1]['col'] == picked[i-2]['col'] + 1                 and picked[i]['col']+1 < picked[i+1]['col']:
                picked[i]['col'] += 1
                 
            # add this: 
            if 2<=i<= len(picked)-2 and picked[i-1]['col'] == picked[i-2]['col'] + 1                 and picked[i]['col']+1 > picked[i+1]['col']:
                picked[i]['text'] = picked[i-1]['text'] + ' ' + picked[i]['text']
            # add end
                
            if 2<=i<= len(picked)-2 and picked[i-1]['col'] > picked[i-2]['col'] + 1                 and picked[i]['col']+1 == picked[i+1]['col']:
                picked[i-1]['col'] -= 1
                    
            if 2<=i<= len(picked)-2 and picked[i-1]['col'] > picked[i-2]['col'] + 1                 and picked[i]['col']+1 < picked[i+1]['col']:
                picked[i]['col'] += 1
                    
            # add this: 
            if 2<=i<= len(picked)-2 and picked[i-1]['col'] < picked[i-2]['col'] + 1                 and picked[i]['col']+1 == picked[i+1]['col']:
                picked[i]['text'] = picked[i-1]['text'] + ' ' + picked[i]['text']
            # add end
             
    # find row and col numbers
    delta = picked[0]['row']

    for ele in picked:
        ele['row'] = ele['row']- delta

    maxrow = 0
    maxcol = 0
    for ele in picked:
        maxrow = max(maxrow, ele['row'])
        maxcol = max(maxcol, ele['col'])
            
        
    # fill the dataframe
    table_df = pd.DataFrame('', index=np.arange(maxrow+1), columns=np.arange(maxcol+1))
    for ele in picked: 
        i = ele['row']
        j = ele['col']
        table_df.iloc[i, j] = ele['text']
            
        
        
    # merge cols
    for i in range(table_df.shape[0]):
        for j in range(1, table_df.shape[1]-1):
            cur_ele = []
            next_ele = []
            for ele in picked:
                if ele['row'] == i and ele['col'] == j:
                    cur_ele = ele
                if ele['row'] == i and ele['col'] == j+1:
                    next_ele = ele
                if cur_ele !=[] and next_ele !=[]:
                    break
                        
            if cur_ele !=[] and next_ele !=[]:
                cur_left, cur_right = cur_ele['x0'], cur_ele['x1']
                next_left, next_right = next_ele['x0'], next_ele['x1']
                    
                if cur_right > next_left:
                    table_df.iloc[i, j] = table_df.iloc[i, j] + ' ' + table_df.iloc[i, j+1]
                    table_df.iloc[i, j+1] = ''
        
        
    # find each's edges and median line, for fine tuning 
    col_data = []
    for i in range(table_df.shape[1]):
        col_p_left = []
        col_p_right = []
        col_p_median = []
        for ele in picked:
            if ele['col'] == i:
                col_p_left.append(ele['x0'])
                col_p_right.append(ele['x1'])
                col_p_median.append((ele['x0']+ele['x1'])/2)
        if col_p_left != [] and col_p_right !=[]:
            ##### line 11122
            col_p = [np.median(col_p_left), np.median(col_p_right), np.median(col_p_median)]
            ##### col_p = [np.min(col_p_left), np.median(col_p_right), np.median(col_p_median)]
            #col_m = np.median(col_p_median)
        else: 
            col_p = []
            #col_m = []
        col_data.append(col_p)
            
    #print(col_data[0])
        
        
    # merge close cols and drop empty cols
    first_drop = []
    
    for i in range(table_df.shape[1]-1):
        if col_data[i] != [] and col_data[i+1] != []:
            if col_data[i+1][0] < col_data[i][2] < col_data[i+1][1]:
            #if col_data[i+1][0] < col_data[i][0] and col_data[i+1][1] > col_data[i][1]  or \
                    #col_data[i+1][0] > col_data[i][0] and col_data[i+1][1] < col_data[i][1]:
                first_drop.append(i)
                table_df.iloc[:, i+1] = table_df.iloc[:, i] + ' ' + table_df.iloc[:, i+1]
                
    #print(first_drop)
    table_df.drop(first_drop, axis = 1, inplace = True)    
    table_df.columns = range(table_df.shape[1])    
        
        
    # further clean the df
    second_drop = []
        
    for i in range(table_df.shape[1]):
        #table_df[i] = [' '.join(table_df[i][k].split('\n')) for k in range(table_df.shape[0])]
        if table_df[i].isnull().all() or np.all( table_df[i] == ''):
            second_drop.append(i)
    table_df.drop(second_drop, axis = 1, inplace = True)
    table_df.columns = range(table_df.shape[1])
        
       
    
    # clean rows
    row_drop = []
    for i in range(table_df.shape[0]):
        if table_df.iloc[i, :].isnull().all() or np.all(table_df.iloc[i, :] == ''):
            row_drop.append(i)
        
    table_df.drop(row_drop, axis = 0, inplace = True)
    table_df.index = range(table_df.shape[0])
    
    # save the results
    alltables.append(table_df)
        
    #print('Done')
        
    return alltables


# In[ ]:


# post-processing: convert strings to int 

def convert2int(table_df):
    
    table_df_new = table_df.copy()
    row = table_df_new.shape[0]
    col = table_df_new.shape[1]
    
    for i in range(row):
        for j in range(col):
            ele = table_df_new.iloc[i, j]
            new_ele = ele.strip().strip('()').strip('$').strip()
            new_ele = new_ele.strip().strip('()').strip('$').strip()
            if ',' in new_ele:
                new_ele_clean = []
                for ele in new_ele.split(','):
                    for k in ele.split():
                        new_ele_clean.append(k)
                new_ele = ''.join(new_ele_clean)
            
            try:
                new_ele = int(new_ele)
                table_df_new.iloc[i, j] = new_ele
            except:
                try: 
                    new_ele = float(new_ele)
                    table_df_new.iloc[i, j] = new_ele
                except ValueError:
                    continue
        
            
    return table_df_new


# In[ ]:


# post-processing: remove non-table-like pagse and convert number-like-string to int 

def finalclean(alltables, col_use, page_tracker, tol = 1):
    
    alltables_clean = []
    page_tracker_clean = []
    
    for i in range(len(alltables)):
        
        if alltables[i].shape[1] > 1 and alltables[i].shape[1] >= len(col_use) - tol:
            page_tracker_clean.append(page_tracker[i])
            
            table_new = convert2int(alltables[i])
            alltables_clean.append(table_new)
            
    return alltables_clean, page_tracker_clean
        


# In[ ]:


# header_tracker is a list, each ele = [header start, header end]

def getheader(alltables_clean, header_key_words, header_start = 6):

    header_tracker = []

    for table in alltables_clean:
        cur_header = []
        for i in range(min(header_start, len(table))):
            foundflag = 1
            j = 0
            #keyword_count = 0
            while foundflag and j < table.shape[1]:
                if type(table.iloc[i, j]) == str:
                    for keyword in header_key_words:
                        if keyword in str.lower(table.iloc[i, j]).split(' '):
                            #print(i, j, keyword)
                            #keyword_count +=1
                            cur_header.append(i)
                            foundflag = 0
                            break
                j +=1
        
        
        # header must be continous number
        stopper = len(cur_header)
        for i in range(len(cur_header)-1):
            if cur_header[i] + 1 != cur_header[i+1]:
                stopper = i + 1
        cur_header = cur_header[:stopper]        
        
        if cur_header == []:
            cur_header = [0]
        header_tracker.append(cur_header) # record the begin and end of header
        
    return header_tracker


# In[ ]:


# only check last 2 rows
def getfootnote(alltables_clean, footnote_stop = 3):
    
    footnote_tracker = []

    for table in alltables_clean:
        cur_footnote = []
        for i in range(len(table)-1, len(table)-footnote_stop, -1):
            cur_col_n = np.sum(table.iloc[i, :] != '') - np.sum(table.iloc[i, :] == ' ')
            if cur_col_n == 1:
                cur_footnote.append(i)
        footnote_tracker.append(cur_footnote)
    
    return footnote_tracker


# In[ ]:


def clean_single_header_footnote(table, header_stop = 3, footnote_stop = 3):
    
    header_tracker = []
    footnote_tracker = []

    for i in range(len(table)-1, len(table)-footnote_stop, -1):
        cur_len = max([len(k) for k in table.iloc[i, :]])
        cur_col_n = np.sum(table.iloc[i, :] != '') - np.sum(table.iloc[i, :] == ' ')
        if cur_col_n == 1 or cur_len >=50:
            footnote_tracker.append(i)
            
            
    for i in range(header_stop):
        cur_len = max([len(k) for k in table.iloc[i, :]])
        #cur_col_n = np.sum(table.iloc[i, :] != '') - np.sum(table.iloc[i, :] == ' ')
        if cur_len >=50: #cur_col_n == 1 or 
            header_tracker.append(i)
    
    if footnote_tracker != []:
        end_row = min(footnote_tracker)
    else:
        end_row = -1
        
    if header_tracker != []:  
        start_row = max(header_tracker)+1
    else:
        start_row = 0
    
    alltables_clean = table.iloc[start_row:end_row, :]
    alltables_clean.index = range(alltables_clean.shape[0])
    
    return alltables_clean



# clean header and footnote

def clean_header_footnote(alltables_clean, header_tracker, footnote_tracker):
    
    for i in range(len(alltables_clean)):
        
        # skip small tables
        if len(alltables_clean[i])>=10:
            #table = alltables_clean[i]
            if header_tracker[i] != []:
                start_row = header_tracker[i][0]
                header_h =  header_tracker[i][-1]
            else:
                start_row = 0
                header_h = 0

            if footnote_tracker[i] != [] and footnote_tracker[i][-1] - header_h > 1:
                end_row = footnote_tracker[i][-1]
            else:
                end_row = len(alltables_clean[i])

            alltables_clean[i] = alltables_clean[i].iloc[start_row:end_row, :]
            alltables_clean[i].index = range(alltables_clean[i].shape[0])
        
    return alltables_clean


# In[ ]:





# In[ ]:


def colcorrection(alltables_clean, header_tracker, ratio = 2):
    
    alltables_clean_1 = []
    
    for i in range(len(alltables_clean)):
        table = alltables_clean[i]
        no_header_col = []
        for j in range(1, table.shape[1]):
            no_header = True
           
            #if np.sum(table[j] != '') < len(table)/ratio:
            '''
            if len([k for k in table[j] if k != '' and k != ' ']) < len(table)/ratio:
                for h_row in header_tracker[i]:
                    if pd.notnull(table.iloc[h_row, j]) and table.iloc[h_row, j] != '' and table.iloc[h_row, j] != ' ':
                        no_header = False
                if no_header:
                    no_header_col.append(j)
            '''
            ## replace previous a step, if current has no header, merge with previous one
            for h_row in header_tracker[i]:
                if pd.notnull(table.iloc[h_row, j]) and table.iloc[h_row, j] != '' and table.iloc[h_row, j] != ' ':
                    no_header = False
            if no_header:
                no_header_col.append(j)
                    
            
        not_remove_col = []
        if no_header_col != []:
            for k in no_header_col:
                # table[k-1] = table[k-1] + ' ' + table[k]
                for m in range(table.shape[0]):
                    if type(table.iloc[m, k-1]) == type(table.iloc[m, k]):
                        try:
                            new_col_val = table.iloc[m, k-1] + ' ' + table.iloc[m, k]
                        except:
                            #print('error when trying to combine columns')
                            #print('table id:', i, '\trow number:', m, '\tcol id:', k, '\n')
                            not_remove_col.append(k)
                            break #continue
                        else:
                            table.iloc[m, k-1] = new_col_val
                    else:
                        if type(table.iloc[m, k-1]) == str:
                            if table.iloc[m, k-1] == '' or table.iloc[m, k-1] == ' ':
                                table.iloc[m, k-1] = table.iloc[m, k]
                            else: 
                                table.iloc[m, k-1] = table.iloc[m, k-1] + ' ' + str(table.iloc[m, k])
                        elif type(table.iloc[m, k]) == str:
                            if table.iloc[m, k] == '' or table.iloc[m, k] == ' ':
                                table.iloc[m, k-1] = table.iloc[m, k-1]
                            else:
                                table.iloc[m, k-1] = str(table.iloc[m, k-1]) + ' ' + table.iloc[m, k]
                        else:
                            #print('table id:', i, '\trow number:', m, '\tcol id:', k, '\n')
                            #print('error when trying to combine columns')
                            continue
            
            final_drop_col = [k for k in no_header_col if k not in not_remove_col]
            table.drop(final_drop_col, axis = 1, inplace = True)
            table.columns = range(table.shape[1])
        
        alltables_clean_1.append(table)
             
    return alltables_clean_1


# In[ ]:





# In[ ]:


import copy
import re
cusippattern = '[0-9A-Za-z]{9}'#### '[0-9]{5}[A-Za-z]{3}[0-9]' # incorrect

def addcusipcol(alltables_clean, header_tracker):
    
    
    alltables_cusip = []
    
    for i in range(len(alltables_clean)): 
        if header_tracker[i] !=[]:
            startrow = header_tracker[i][-1]+1
        else:
            startrow = 0
            
        table = copy.deepcopy(alltables_clean[i].iloc[startrow:,:])
        row_n = alltables_clean[i].shape[0] 
        col_n = alltables_clean[i].shape[1]
        
        #cusipvals = pd.DataFrame(np.zeros((row_n, 1)))
        cusipvals = pd.DataFrame(['']*row_n)
        rowremove = []
        
        # only check 1-3 cols? or all? 
        for j in range(table.shape[0]):
            for k in range(table.shape[1]):
                if type(table.iloc[j, k]) == str and 'cusip' in table.iloc[j, k].lower():
                    indx = str.find(table.iloc[j, k].lower(), 'cusip')
                    strings = table.iloc[j, k][indx+5:]
                    cusiploc = re.findall(cusippattern, strings)
                    if cusiploc:
                        if (np.sum(table.iloc[j, :] == '') + np.sum(table.iloc[j, :] == ' ')) == col_n - 1:
                            
                            # can remove this row
                            if len(re.findall('[0-9A-Za-z]+', table.iloc[j, k])) == 2:
                                cusipvals.iloc[startrow+j-1] = cusiploc[0]
                                rowremove.append(startrow+j)
                            # this row has 'cusip' + 'numbers' + 'something else', cannot be removed 
                            else:
                                cusipvals.iloc[startrow+j] = cusiploc[0]
                        else:
                            cusipvals.iloc[startrow+j] = cusiploc[0]
        
        # no matter we find cusip or not, need to match all tables. 
        
        #if np.any(cusipvals !=0):
            
        cusipvals.iloc[startrow-1] = 'CUSIP'
        table_add = pd.concat((alltables_clean[i], cusipvals), axis = 1)
            
        if rowremove:
            table_add.drop(rowremove, axis = 0, inplace = True)
            
        table_add.columns = range(table_add.shape[1])
        table_add.index = range(table_add.shape[0])
        alltables_cusip.append(table_add)
        
      
    return alltables_cusip
        
 


# In[ ]:


# need to remove headers and stack all first

import copy

def stacktables(alltables_clean, header_tracker):
    
    if header_tracker[0] !=[]:
        alltables_clean_1 = [copy.deepcopy(alltables_clean[0].iloc[header_tracker[0][0]:, :])]
    else:
        alltables_clean_1 = [copy.deepcopy(alltables_clean[0])]

    for i in range(1, len(alltables_clean)):
        
        if header_tracker[i] != []:
            table = copy.deepcopy(alltables_clean[i].iloc[header_tracker[i][-1]+1:, : ])
        else:
            table = copy.deepcopy(alltables_clean[i])

        drop_col = []
        for j in range(table.shape[1]):
            #if table[j].isnull().all() or np.all(table[j] == ''):
            if alltables_clean[i].iloc[:, j].isnull().all() or np.all(alltables_clean[i].iloc[:, j] == '') or                    np.all(alltables_clean[i].iloc[:, j] == ' '):
                drop_col.append(j)
        table.drop(drop_col, axis = 1, inplace = True)
        table.columns = range(table.shape[1])

        alltables_clean_1.append(table)

    alltables_clean_1 = pd.concat(alltables_clean_1, axis = 0, ignore_index=True)

    return alltables_clean_1


# now only have total and subheader 
# alltables_clean_1


# In[ ]:


# detect total and subheader
# need to track row + col #

def getheaderinfo(alltables_clean_1, header_tracker):

    subheader_section = []
    subheader_total = []
    subheader_other = []

    for i in range(header_tracker[0][-1]+1, len(alltables_clean_1)):
        mask = (alltables_clean_1.iloc[i, :] != '') & (alltables_clean_1.iloc[i, :] != ' ')
        cur_col_n = np.sum(mask == True)
        #cur_col_n = np.sum(alltables_clean_1.iloc[i, :].apply(lambda s: s.strip()) != '')
        
        if cur_col_n == 1:
            subheader_section.append(i)

        elif 1 < cur_col_n < alltables_clean_1.shape[1]:
            cur_row = alltables_clean_1.iloc[i, :]
            total_found = 0
            for k in cur_row:
                if type(k) == str and 'total' in k.lower():
                    subheader_total.append([i, cur_col_n])
                    total_found = 1
                    break
            
            if total_found == 0:       
            #if i not in total_rows:
                subheader_other.append([i, cur_col_n])

        else:
            continue
            
    return subheader_total, subheader_section, subheader_other


# In[ ]:



def getheaderdetails(alltables_clean_1, header_tracker, subheader_total, subheader_section, subheader_other):
    subheader_total_detail = []
    subheader_section_detail = []
    subheader_other_detail = []

    for i in range(header_tracker[0][-1]+1, len(alltables_clean_1)):
        total_rows = [ele[0] for ele in subheader_total]
        section_rows = [ele for ele in subheader_section]
        other_rows = [ele[0] for ele in subheader_other]

        if i in total_rows:
            #for j in range(alltables_clean_1.shape[1]):
                #if type(alltables_clean_1.iloc[i, j]) == int or type(alltables_clean_1.iloc[i, j]) == float:
            subheader_total_detail.append([i, alltables_clean_1.iloc[i, :]])

        if i in section_rows:
             #for j in range(alltables_clean_1.shape[1]):
                    #if alltables_clean_1.iloc[i, j] != '':
            subheader_section_detail.append([i, alltables_clean_1.iloc[i, :]])
                        
        if i in other_rows:
             #for j in range(alltables_clean_1.shape[1]):
                    #if alltables_clean_1.iloc[i, j] != '' and alltables_clean_1.iloc[i, j] != ' ':
            subheader_other_detail.append([i, alltables_clean_1.iloc[i, :]])
    return subheader_total_detail, subheader_section_detail, subheader_other_detail


# In[ ]:


# check and remove subheaders
# choose 50 as tolerance??

# [cur_row_id, col_id, stop_row_id, this_col's total]

def tracktotalvalue(alltables_clean_1, header_tracker, subheader_total, tol = 5.0):
    total_tracker = []
    total_rows = [ele[0] for ele in subheader_total]

    for i in range(header_tracker[0][-1]+1, len(alltables_clean_1)):

        if i in total_rows:
            for j in range(alltables_clean_1.shape[1]):
                if type(alltables_clean_1.iloc[i, j]) == int or type(alltables_clean_1.iloc[i, j]) == float:
                    remain = alltables_clean_1.iloc[i, j]
                    k = i - 1
                    while k >=0:
                        if type(alltables_clean_1.iloc[k, j]) == int or type(alltables_clean_1.iloc[k, j]) == float: 
                            remain -= alltables_clean_1.iloc[k, j]
                            if remain == 0.0 or abs(remain) < tol:
                                total_tracker.append([i, j, k, alltables_clean_1.iloc[i, j], remain])
                                break
                            k -=1
                        else:
                            k -=1

        #if i == 1309:
            #print(i, j, k, remain)
    return total_tracker


# In[ ]:


# confirm that the total_row can be removed

def findremovecol(subheader_total, total_tracker):
    row_can_remove = []
    row_cannot_remove = []

    for ele1 in subheader_total:
        cur_col_count = 0
        for ele2 in total_tracker:
            if ele1[0] == ele2[0]:
                cur_col_count +=1
        # this is very strict check
        #if cur_col_count == ele1[1] - 1:
        
        # a loose check is
        if cur_col_count >= 1:
            row_can_remove.append(ele1[0])
        else: 
            row_cannot_remove.append(ele1[0])
    return row_can_remove, row_cannot_remove


# In[ ]:


import copy
def rmtotalvalue(alltables_clean_1, row_can_remove):
    alltables_clean_2 = copy.deepcopy(alltables_clean_1)

    alltables_clean_2.drop(row_can_remove, axis = 0, inplace = True)
    alltables_clean_2.index = range(alltables_clean_2.shape[0])
    
    return alltables_clean_2


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# deal with $ sign % sign
def removesign(allwords):
    signchar = []
    for i in range(len(allwords)):
        
        #word_df = pd.DataFrame(allwords).groupby('row').count().reset_index()
        #single_ele_row = word_df[word_df['text'] == 1]['row'].to_list()
        
        if (allwords[i]['text'].strip() == '$' or allwords[i]['text'].strip() == '%' or \
            allwords[i]['text'].strip() == 'x' or \
            '--' in allwords[i]['text'].strip()):
            # or allwords[i]['text'].strip() == '*' 
            #and \(allwords[i]['row'] not in single_ele_row):
            signchar.append(i)

    allwords = [allwords[k] for k in range(len(allwords)) if k not in signchar]
    
    return allwords


# In[ ]:


# find all potential row seperators on one single page

def find_page_structure(allwords_df, page, tol = 5, min_sep = 3, method = 'min'): 
    
    use_df = allwords_df.copy()
    cur_page_df = pd.DataFrame(use_df[page])
    
    # get coordinate information 
    page_positions = cur_page_df[['row', 'top', 'bottom']]
    page_positions = page_positions.drop_duplicates().sort_values(by = 'row')
    page_positions.index = np.arange(len(page_positions))
    ############################
    
    # find overlapping rows
    i = 0
    repeat_row_id = []
    while i < len(page_positions) - 1:
        if page_positions.loc[i, 'row'] != page_positions.loc[i+1, 'row']:
            i += 1
            continue
        else: 
            cur_rw_id = [i]
            while i < len(page_positions) - 1 and page_positions.loc[i, 'row'] == page_positions.loc[i+1, 'row']:
                i += 1
            cur_rw_id.append(i)
            repeat_row_id.append(cur_rw_id)

    ###########################
    # merge overlapping rows
    
    new_rows = []
    for row_group in repeat_row_id:
        cur_row = page_positions.loc[row_group[0], 'row']
        cur_top = []
        cur_bottom = []
        for row in row_group:
            cur_top.append(page_positions.loc[row, 'top'])
            cur_bottom.append(page_positions.loc[row, 'bottom'])
        new_rows.append({'row': cur_row, 'top': max(cur_top), 'bottom': min(cur_bottom)})
    new_rows = pd.DataFrame(new_rows, columns = ['row', 'top', 'bottom'])    

    ##########################
    # drop and merge
    
    allid = []
    for k in repeat_row_id:
        allid += k

    page_positions= page_positions.drop(allid, axis = 0)
    page_positions = pd.concat([page_positions, new_rows], axis = 0)
    page_positions = page_positions.sort_values(by = 'row')
    page_positions.index = np.arange(len(page_positions))

    ##########################
   
    page_positions['linechange'] = 0
    for i in range(1, len(page_positions)):
        page_positions.loc[i, 'linechange'] = page_positions.loc[i, 'top'] - page_positions.loc[i-1, 'bottom']

    breaklines = []
    # do not count in the last row
    for i in range(len(page_positions) - 1):
        if page_positions.loc[i, 'linechange'] >= tol:
            breaklines.append(page_positions.loc[i, 'row'])
   
    ##########################
    # clean row seperators

    breaklines_clean = grouplines(breaklines, min_sep) #, method
    
    ##########################
    # find possible table boundaries
    table_edges = []
    if breaklines_clean[0] >= 3:
        table_edges.append([0, breaklines_clean[0]])
    for i in range(len(breaklines_clean) - 1):
        table_edges.append([breaklines_clean[i], breaklines_clean[i+1]])
    if breaklines_clean[-1] <= page_positions['row'].values[-1] - 3:
        table_edges.append([breaklines_clean[-1], page_positions['row'].values[-1]])
            
    return table_edges, breaklines_clean, page_positions


# In[ ]:


def check_connect_row(page_positions, min_sep = 1):
    page_positions['merge'] = -1
    
    # find most common line space
    common_linespace = page_positions['linechange']
    common_linespace = pd.value_counts(common_linespace)
    common_linespace = sorted(common_linespace.items(), key = lambda x: x[1], reverse = True)
    
    common_1 = common_linespace[0]
    common_2 = common_linespace[1]
    
    if common_1[1] > 1.5 * common_2[1]:
        common_linespace = common_1[0]
    else:
        common_linespace = common_2[0]
    
    if common_linespace < 0:
        common_linespace = 0
    i = 0
    while i < len(page_positions):
        j = i + 1
        while j < len(page_positions) and page_positions.loc[j, 'linechange'] <= min_sep and             page_positions.loc[j, 'linechange'] < common_linespace:
            page_positions.loc[j, 'merge'] = i
            j += 1
        i = j
    return page_positions


# In[ ]:


def merge_row_df(table, page_positions, tol_ratio = 2):
    row_drop = []
    for i in range(1, len(table)):
        merge_row = page_positions.loc[i, 'merge']
        if merge_row != -1:
            
            pre_ele = len([k for k in table.iloc[i-1, :] if k != '' and k != ' ']) 
            cur_ele = len([k for k in table.iloc[i, :] if k != '' and k != ' ']) 
            
            if pre_ele <= int(table.shape[1]/tol_ratio) or cur_ele <= int(table.shape[1]/tol_ratio):
                row_drop.append(i)
                for j in range(table.shape[1]):
                    table.iloc[merge_row, j] = str(table.iloc[merge_row, j]) + ' ' + str(table.iloc[i, j])

    table.drop(row_drop, axis = 0, inplace = True)
    table.index = np.arange(len(table))
    
    return table


# In[ ]:


def get_common_col(cols):
    return mode(cols)[0][0]

def get_common_sep(seperators, col_n):
    
    select_sep = []
    for sep in seperators:
        if len(sep) == col_n - 1:
            select_sep.append(sep)
    
    common_sep = []
    for i in range(col_n-1):
        cur_common = []
        for sep in select_sep:
            cur_common.append(sep[i])
        cur_common = np.median(cur_common)
        common_sep.append(int(cur_common))
        
    return common_sep


# In[2]:


def find_table_structure(allwords, page_positions, table_edges, min_sep = 25):
    
    curdf = pd.DataFrame(allwords)
    
    # group row eles
    row_record = {}
    for row in range(len(page_positions)):
        row_eles = []
        for i in range(len(curdf)):
            if curdf.loc[i, 'row'] > row:
                break
            if curdf.loc[i, 'row'] == row and curdf.loc[i, 'row'] >= table_edges[0] and curdf.loc[i, 'row'] <= table_edges[1]:
                row_eles.append([curdf.loc[i, 'x0'], curdf.loc[i, 'x1']])
        row_record[row] = row_eles
    
    # find col num
    ele_n = []
    for row in row_record:
        ele_n.append(len(row_record[row]))
    #col_n = mode(ele_n)[0][0]
    
    cols = pd.value_counts(ele_n)
    cols_sort = sorted(cols.items(), key = lambda x: x[1], reverse = True)
    
    col_n = cols_sort[0][0]
    if col_n <=2:
        col_id = 1
        while col_id < len(cols_sort) and col_n <= 2:
            col_n = cols_sort[col_id][0]
            col_id += 1

    # find col seperators
    col_breaks = []
    for row in row_record:
        row_eles = row_record[row]
        if len(row_eles) == col_n:
            for i in range(1, len(row_eles)):
                #col_break = int(row_eles[i][1])
                col_break = int(0.5*(row_eles[i][0] +  row_eles[i-1][1]))
                col_breaks.append(col_break)
    
    
    # clean col seperators
    col_breaks = np.unique(col_breaks)
    
    col_break_clean = grouplines(col_breaks, min_sep)
    
    num_run = 0 
    while num_run<50 and len(col_break_clean) > col_n - 1:
        num_run += 1
        min_sep += 5
        col_break_clean = grouplines(col_breaks, min_sep)
    
    return col_break_clean, col_n, row_record, ele_n


# In[ ]:


def check_non_table(alltables, page_tracker, thresh = 20, ratio = 0.05):
    page_tracker_clean = []
    alltables_clean = []
    for i in range(len(alltables)):
        long_ele = 0
        for j in range(alltables[i].shape[0]):
            for k in range(alltables[i].shape[1]):
                if len(alltables[i].iloc[j, k]) >= 50:
                    long_ele += 1
        
        if long_ele < thresh and long_ele/(alltables[i].shape[0]*alltables[i].shape[1]) < ratio:
            page_tracker_clean.append(page_tracker[i])
            alltables_clean.append(alltables[i])
        
    return alltables_clean, page_tracker_clean

# In[ ]:


def checkrotation_1(alltext_1, allwords, pages):
    
    rotation = {}
    for page in pages:
        rot_cur_page = 0
        if len(allwords[page]) > 50 and len(alltext_1[page].split('\n')) < 2:
            rot_cur_page = 1
        rotation[page] = rot_cur_page

    return rotation

# post-processing functions: if page is rotated, correct it

def correctrotation_1(pdffile, allwords, alllines, allrects, rotation):
    
    pdf_writer = PyPDF2.PdfFileWriter()
    pdf_reader = PyPDF2.PdfFileReader(pdffile)
    
    pages = list(rotation.keys())
    
    # write all rotated pages into a temporary file 
    
    rotated = []
    for i in range(len(pages)):
        page_n = pages[i]
        if rotation[page_n] == 1: 
            rotated.append(i)
            page = pdf_reader.getPage(page_n).rotateClockwise(90)
            pdf_writer.addPage(page)
    
    if rotated == []:
        return allwords, alllines, allrects
    
    with tempfile.NamedTemporaryFile(delete=False) as fp:
        pdf_writer.write(fp)
        fp.seek(0)
        pdf = pdfplumber.open(fp.name)
        
        for i in range(len(rotated)):
            page_n = pages[rotated[i]] 
            page = pdf.pages[i]
            words = page.extract_words()
            allwords[page_n] = words
            
            lines = page.lines
            alllines[page_n] = lines
            
            rects = page.rects
            allrects[page_n] = rects
        
    return allwords, alllines, allrects



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


if __name__ == "__main__":
    with open('./allwords_df.pkl', 'rb') as f:
        allwords_df = pickle.load(f)
    find_page_structure(allwords_df, 0, tol = 5, min_sep = 1)     
        
