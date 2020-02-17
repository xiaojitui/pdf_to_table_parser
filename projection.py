#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pdfplumber
import os
import pandas as pd
import numpy as np
import copy
import table_utils 
import sys
#sys.path.insert(0, './pdf_parser')
from table_utils import check_non_table
import pdb
import pickle


# In[23]:

def non_max_suppression_fast(boxes, probs, overlap_thresh=0.7, max_boxes=300):

    if len(boxes) == 0:
        return []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes	
    pick = []

    # calculate the areas
    area = (x2 - x1) * (y2 - y1)

    # sort the bounding boxes 
    idxs = np.argsort(probs)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the intersection

        xx1_int = np.maximum(x1[i], x1[idxs[:last]])
        yy1_int = np.maximum(y1[i], y1[idxs[:last]])
        xx2_int = np.minimum(x2[i], x2[idxs[:last]])
        yy2_int = np.minimum(y2[i], y2[idxs[:last]])

        ww_int = np.maximum(0, xx2_int - xx1_int)
        hh_int = np.maximum(0, yy2_int - yy1_int)

        area_int = ww_int * hh_int

        # find the union
        area_union = area[i] + area[idxs[:last]] - area_int

        # compute the ratio of overlap
        overlap = area_int/(area_union + 1e-6)

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlap_thresh)[0])))

        if len(pick) >= max_boxes:
            break

    # return only the bounding boxes that were picked using the integer data type
    boxes = boxes[pick].astype("int")
    probs = probs[pick]
    return boxes, probs


def clean_tablelocs(tablelocs):
    tablelocs_clean = {}
    for i in range(len(tablelocs)):
        tablelocs_clean[i] = []
        
        for tableloc in tablelocs[i]:
            
            if tableloc == []:
                tablelocs_clean[i] = []
                continue
                
            box = tableloc.bbox
            box = [int(k) for k in box]
            box[0] = 20
            box[2] = 600
            tablelocs_clean[i].append(box)
        
        tablelocs_clean[i] = np.array(tablelocs_clean[i])
        probs = np.array([1] * len(tablelocs_clean[i]))
        tablelocs_clean[i] = non_max_suppression_fast(tablelocs_clean[i], probs, overlap_thresh=0.7, max_boxes=300)
        if tablelocs_clean[i] != []:
            tablelocs_clean[i] = tablelocs_clean[i][0].tolist()
    return tablelocs_clean


def merge_row_projection(tables, tol_s = 2):
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






def get_table_project(pdffile, page_nums = None, remove_non_table = False):
    alltext, allwords, tablelocs, alllines, allrects, pages = table_utils.get_text_word(pdffile, page_nums)
    
    if tablelocs != [[]]:
        tablelocs = clean_tablelocs(tablelocs)
    
    
    if len(pages) == 0:
        return {}
    
    if page_nums is None:
        page_tracker = np.arange(len(pages))
    else:
        page_tracker = page_nums
 
    allwords_df = copy.deepcopy(allwords) # 
    #alllines_df = copy.deepcopy(alllines)
    #allrects_df = copy.deepcopy(allrects)

    allresults = {}
    for idx in range(len(page_tracker)):
        page = page_tracker[idx]
        allresults[page] = {}
        
        #allwords_df[page] = utils.removesign(allwords_df[page])
        allwords_df = table_utils.groupwords(allwords_df)
        try:
            table_edges, _, page_positions = table_utils.find_page_structure(allwords_df, idx, tol = 5, min_sep = 1) #page
            table_edges = [table_edges[0][-1], table_edges[-1][-1]]
            page_positions = table_utils.check_connect_row(page_positions)

            col_break_clean, col_n, _, ele_n = table_utils.find_table_structure(allwords_df[idx], page_positions, table_edges) #page

            allresults[page]['ele_n'] = ele_n
            allresults[page]['col_n'] = col_n
            allresults[page]['col_sep'] = col_break_clean
            #allresults[page]['table_edge'] = table_edges
            allresults[page]['page_position'] = page_positions

            if col_break_clean == []:
                continue

            col_sep = [col_break_clean[0] - 50] + col_break_clean + [col_break_clean[-1] + 50]
            if tablelocs[idx] == []: #page
                curtable = table_utils.preparedf_single(allwords_df[idx], pages, page, col_sep, [20, 780]) #page
                if curtable != []:
                    curtable[0] = table_utils.clean_single_header_footnote(curtable[0])
                    curtable[0] = table_utils.merge_row_df(curtable[0], allresults[page]['page_position'])
                allresults[page]['table'] = [curtable[0]]
                allresults[page]['bbox'] = [[20, 20, int(pages[page].width)-20, 780]]
            else:
                allresults[page]['table'] = []
                allresults[page]['bbox'] = []
                for tableloc in tablelocs[idx]: #page
                    #t1 = max(0, int(tableloc.bbox[1]) - 20)
                    #t2 = min(780, int(tableloc.bbox[3]) + 20)
                    t1 = max(0, int(tableloc[1]) - 20)
                    t2 = min(780, int(tableloc[3]) + 20)
                    curtable = table_utils.preparedf_single(allwords_df[idx], pages, page, col_sep, [t1, t2]) #page
                    if curtable != []:
                        curtable[0] = table_utils.clean_single_header_footnote(curtable[0])
                    allresults[page]['table'].append(curtable[0])
                    allresults[page]['bbox'].append([20, t1, int(pages[page].width)-20, t2])
        except:
            continue
    
    alltables = {}
    allboxes = {}
    for page in page_tracker:
        if 'table' in allresults[page]:
            tables = allresults[page]['table']
            bboxes = allresults[page]['bbox']
            if remove_non_table == True:
                tables, bboxes = check_non_table(tables, bboxes)
               
                tables = merge_row_projection(tables)
            if tables != []:
                alltables[page] = tables
                allboxes[page] = bboxes
                
    
    return alltables, allboxes


# In[59]:


def get_word_in_box(allword, bbox, tol = 10):
    allwords_clean = []
    for word in allword:
        x0 = word['x0']  # need some tolerence?
        x1 = word['x1']
        top = word['top']
        bottom = word['bottom']

        if x0>=bbox[0]-tol and x1<=bbox[2]+tol and top>=bbox[1]-tol and bottom<=bbox[3]+tol:
            allwords_clean.append(word)
    return [allwords_clean]


# In[63]:


def get_table_project_bbox(pdffile, boxes, remove_non_table = False):
    
    page_nums = list(boxes.keys())
    
    if page_nums == []:
        page_nums = None
        
    alltext, allwords, tablelocs, alllines, allrects, pages = table_utils.get_text_word(pdffile, page_nums)
    
    if len(pages) == 0:
        return {}
    
    if page_nums is None:
        page_tracker = np.arange(len(pages))
    else:
        page_tracker = page_nums
 
    #alllines_df = copy.deepcopy(alllines)
    #allrects_df = copy.deepcopy(allrects)

    allresults = {}
    allboxes = {}
    for idx in range(len(page_tracker)):
        page = page_tracker[idx]
        
        if page not in boxes:
            continue
            
        allword = allwords[idx] #page
        allresults[page] = []
        allboxes[page] = []
        areas = boxes[page]
        
        for area in areas:
            
            try: 
                
                allwords_clean = get_word_in_box(allword, area)

                allwords_df = copy.deepcopy(allwords_clean) # 

                #print(allwords_df)
                #allwords_df[page] = utils.removesign(allwords_df[page])
                allwords_df = table_utils.groupwords(allwords_df)
                if not os.path.exists('allwords_df_debug.pkl'):
                    with open('allwords_df_debug.pkl', 'wb') as f:
                        pickle.dump(allwords_df, f)
                table_edges, _, page_positions = table_utils.find_page_structure(allwords_df, 0, tol = 5, min_sep = 1)
                table_edges = [table_edges[0][-1], table_edges[-1][-1]]
                page_positions = table_utils.check_connect_row(page_positions)
                col_break_clean, col_n, _, ele_n = table_utils.find_table_structure(allwords_df[0], page_positions, table_edges)
                
                if col_break_clean == []:
                    continue

                col_sep = [col_break_clean[0] - 50] + col_break_clean + [col_break_clean[-1] + 50]

                curtable = table_utils.preparedf_single(allwords_df[0], pages, 0, col_sep, [0, 2000])
                if curtable != []:
                    curtable[0] = table_utils.clean_single_header_footnote(curtable[0])
                    curtable[0] = table_utils.merge_row_df(curtable[0], page_positions)
                allresults[page].append(curtable[0])
                allboxes[page].append(area)
                
            except:
                continue


    alltables = {}
    allboxes_clean = {}
    for k, v in allresults.items():
        if remove_non_table == True and v != []:
            tables, clean_boxes = check_non_table(v, allboxes[k])
            tables = merge_row_projection(tables)
        else:
            tables = v
            clean_boxes = allboxes[k]
                
        alltables[k] = tables
        allboxes_clean[k] = clean_boxes
    
    return alltables, allboxes_clean
    
