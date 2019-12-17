# -*- coding: utf-8 -*-
"""
###############################################################################

The module "helper_functions" contains functions that I frequently use.

@author: Tommy Lang

Functions:
    1. update_mod(destination = 'default', source = src) 
    2. get_path(name = 'tom')
    3. set_dir(name = 'tom')
    4. open_db()
    5. read_chunks(query, chunksize, rows)
    6. missing_value_df(df, sort = False)
    7. multilabel_sample(y, size=1000, min_count=5, seed=None)
    8. multilabel_sample_dataframe(df, labels, size, min_count=5, seed=None)
    9. multilabel_train_test_split(X, Y, size, min_count=5, seed=None)
    10. msg(text = 'Finished Running Code')
    
Installation:
    1. Copy the file "helper_functions.py" into a PYTHONPATH directory
        - 'C:/Users/tommy/Dropbox/7. projects/py_helper_functions/helper_functions.py'
        - Run statements below to get a list of PYTHONPATH directories:
            - import sys
            - sys.path
    2. Import the module by using: import helper_functions as hf
    
###############################################################################
"""
#%%
import os
import cx_Oracle
import pandas as pd
import math
import time
import sys
import tkinter as tk
from datetime import datetime
from tkinter import ttk
from shutil import copy

#%%
src = 'C:/Users/tommy/Dropbox/7. projects/py_helper_functions/helper_functions.py'
def update_mod(destination = 'default', source = src):
    '''Copy helper_functions.py file to 4th PYTHONPATH directory'''
    
    if destination == 'default':
        destination = sys.path[3]
        copy(source, destination)
    else:
        copy(source, destination)
    
    print('helper_functions.py file copied to dir: ', destination)

# update_mod()
    
#%%
# Functions for file directory

def get_path(name = 'tom'):
    '''Returns string for file path to directory. Ex: get_path('tom')'''
    if name == 'team':
        ret_path = 'C:/Users/'
    elif name == 'tom':
        ret_path = 'C:/Users/tommy/Dropbox/7. projects/'
    else:
        raise ValueError("Not a valid argument to get_path(). Use 'tom' or 'team'.")
    return ret_path

def set_dir(name = 'tom'):
    if name == 'team':
        os.chdir(get_path('team'))
    elif name == 'tom':
        os.chdir(get_path('tom'))
    else:
        raise ValueError("Not a valid argument to set_dir(). Use 'tom' or 'team'.")
        
#%%
# Functions for DB query
        
# Oracle parameters used to construct the database connection string
hosts = [
            'host_string1.com',
            'host_string2.com',
            'host_string3.com',
            'host_string4.com'    
            ]

port = 123
service_name = 'db_name'
dis = 'user_name'
guise = 'password'

def dsn_list(): 
    '''Creates a list of connection strings to be used by open_db()'''

    host_list = []
    for host in hosts:
        dsn = cx_Oracle.makedsn(host, port, 
                                    service_name = service_name)
        host_list.append(dsn)
    return host_list

def open_db():
    '''Returns the oracle database connection object'''

    last_err = None    
    dsns = dsn_list()
    for dsn in dsns:
        try:
            connection = cx_Oracle.connect(dis, guise, dsn)
        except cx_Oracle.DatabaseError as e:
            last_err = e
            continue        
        return connection        
    raise last_err  
    
def read_chunks(query, chunksize, rows):
    '''Returns query results as DF. Reads query in chunks.
    
    Args:
        query: string for the query to be run
        chunksize: number for the number of rows to be read each iteration
        rows: the total number of rows query returns
    '''
    
    chunks = pd.read_sql(sql = query, con = open_db(), chunksize = chunksize)
    
    chunks_n = math.ceil(rows / chunksize)
    result_list = []
    n = 1
    start = time.time()
    start_i = start
    
    for chunk in chunks:
        result_list.append(chunk)
        time_i = time.time() - start_i
        
        print('read chunk:', n, '/', chunks_n, 
              '   elapsed time:', round(time.time() - start, 2), 
              '   time left:', round(time_i * (chunks_n - n),2))
        
        n +=1 
        start_i = time.time() 
        
    df = pd.concat(result_list)
    
    return(df)

#%%
# Misc functions
    
def missing_value_df(df, sort = False):
    '''
    takes a DF and returns a DF that shows how many missing vals in each col
    '''
    
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    
    mis_val_table = mis_val_table.rename(
        columns = {0 : 'missing_vals', 1 : 'mis_val_percent'})
    
    if sort == True:
        mis_val_table = mis_val_table.sort_values('mis_val_percent', 
                                                  ascending=False)
        
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
        "There are " + str(mis_val_table[mis_val_table.iloc[:,1] != 0].shape[0]) +
          " columns that have missing values.")
    
    return mis_val_table

def multilabel_sample(y, size=1000, min_count=5, seed=None):
    """ Takes a matrix of binary labels `y` and returns
        the indices for a sample of size `size` if
        `size` > 1 or `size` * len(y) if size =< 1.

        The sample is guaranteed to have > `min_count` of
        each label.
    """
    try:
        if (np.unique(y).astype(int) != np.array([0, 1])).any():
            raise ValueError()
    except (TypeError, ValueError):
        raise ValueError('multilabel_sample only works with binary indicator matrices')
    if (y.sum(axis=0) < min_count).any():
        raise ValueError('Some classes do not have enough examples. Change min_count if necessary.')
    if size <= 1:
        size = np.floor(y.shape[0] * size)
    if y.shape[1] * min_count > size:
        msg = "Size less than number of columns * min_count, returning {} items instead of {}."
        warn(msg.format(y.shape[1] * min_count, size))
        size = y.shape[1] * min_count
    rng = np.random.RandomState(seed if seed is not None else np.random.randint(1))
    if isinstance(y, pd.DataFrame):
        choices = y.index
        y = y.values
    else:
        choices = np.arange(y.shape[0])
    sample_idxs = np.array([], dtype=choices.dtype)
    # first, guarantee > min_count of each label
    for j in range(y.shape[1]):
        label_choices = choices[y[:, j] == 1]
        label_idxs_sampled = rng.choice(label_choices, size=min_count, replace=False)
        sample_idxs = np.concatenate([label_idxs_sampled, sample_idxs])
        
    sample_idxs = np.unique(sample_idxs)
    # now that we have at least min_count of each, we can just random sample
    sample_count = int(size - sample_idxs.shape[0])
    # get sample_count indices from remaining choices
    remaining_choices = np.setdiff1d(choices, sample_idxs)
    remaining_sampled = rng.choice(remaining_choices,
                                   size=sample_count,
                                   replace=False)
    return np.concatenate([sample_idxs, remaining_sampled])





def multilabel_sample_dataframe(df, labels, size, min_count=5, seed=None):

    """ Takes a dataframe `df` and returns a sample of size `size` where all
        classes in the binary matrix `labels` are represented at
        least `min_count` times.
    """
    idxs = multilabel_sample(labels, size=size, min_count=min_count, seed=seed)
    return df.loc[idxs]





def multilabel_train_test_split(X, Y, size, min_count=5, seed=None):

    """ Takes a features matrix `X` and a label matrix `Y` and
        returns (X_train, X_test, Y_train, Y_test) where all
        classes in Y are represented at least `min_count` times.
    """
    
    index = Y.index if isinstance(Y, pd.DataFrame) else np.arange(Y.shape[0])
    test_set_idxs = multilabel_sample(Y, size=size, min_count=min_count, 
                                      seed=seed)
    train_set_idxs = np.setdiff1d(index, test_set_idxs)
    test_set_mask = index.isin(test_set_idxs)
    train_set_mask = ~test_set_mask
    
    return (X[train_set_mask], X[test_set_mask], Y[train_set_mask], 
            Y[test_set_mask])

def msg(text = 'Finished Running Code'):
    '''
    creates poopup message with passed in text
    '''
    popup = tk.Tk()
    popup.wm_title("Message")
    
    label = ttk.Label(popup, text=text, font=("Verdana", 12))
    label.pack(pady=10,padx=10)
    cur_time = datetime.now().strftime("%H:%M:%S")
    time = ttk.Label(popup, text= cur_time, font=("Verdana", 12))
    time.pack(pady=10,padx=10)
    space = ttk.Label(popup, text= '', font=("Verdana", 12))
    space.pack(pady=10,padx=10)
    
    B1 = ttk.Button(popup, text="Close", command = popup.destroy)
    B1.pack()
    
    popup.geometry('300x200')
    popup.mainloop()

def set_diff(list_a, list_b):
    in_a_not_b = list(set(list_a) - set(list_b))
    in_b_not_a = list(set(list_b) - set(list_a))
    print('length of in list a but not in list b: ' + str(len(in_a_not_b)))
    print('length of in list b but not in list a: ' + str(len(in_b_not_a)))
    return in_a_not_b, in_b_not_a

#%%
# Examples of running query

if __name__ == '__main__':
    
    # query example 
    query = 'select * from  table where rownum < 10005'
    query_result1 = pd.read_sql(sql = query, con = open_db())
    
    # query example with text file
    dq_query = open('C:/Users/tommy/Dropbox/7. projects/query.txt')
    query_result2 = pd.read_sql(sql = dq_query.read(), con = open_db())
    
    # query example with chunks
    query_result3 = read_chunks(query, chunksize = 5000, rows = 10005)
    
    # query example with parameters
    dq_query = open(get_path() + 'query_w_param.txt')
    query_result4 = pd.read_sql(sql = dq_query.read(), con = open_db(), 
                                params = {'CURRENT_DATE':'20190918', 
										  'PRIOR_DATE':'20190917'})
