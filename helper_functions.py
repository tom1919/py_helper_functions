# -*- coding: utf-8 -*-
"""
###############################################################################

The module "helper_functions" contains functions that I frequently use.

@author: Tommy Lang

TODO:
    1. update docs

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
#%% Load Libs
import os
import cx_Oracle
import pandas as pd
import numpy as np
import math
import time
import sys
import tkinter as tk
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
from tkinter import ttk
from shutil import copy

#%% Update Module

src = 'C:/Users/tommy/Dropbox/7. projects/py_helper_functions/helper_functions.py'
def update_mod(destination = 'default', source = src):
    '''Copy helper_functions.py file to 4th PYTHONPATH directory'''
    
    if destination == 'default':
        destination = sys.path[3]
        copy(source, destination)
    else:
        copy(source, destination)
    
    print('helper_functions.py file copied to dir: ', destination)

if __name__ == '__main__':
    import sys
    from shutil import copy
    update_mod()
    
#%% File Path

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
        
#%% DB query
        
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

#%% Data Wrangling

    
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
    #train_set_idxs = np.setdiff1d(index, test_set_idxs)
    test_set_mask = index.isin(test_set_idxs)
    train_set_mask = ~test_set_mask
    
    return (X[train_set_mask], X[test_set_mask], Y[train_set_mask], 
            Y[test_set_mask])

def set_diff(list_a, list_b):
    in_a_not_b = list(set(list_a) - set(list_b))
    in_b_not_a = list(set(list_b) - set(list_a))
    print('length of in list a but not in list b: ' + str(len(in_a_not_b)))
    print('length of in list b but not in list a: ' + str(len(in_b_not_a)))
    return in_a_not_b, in_b_not_a

def compare_df(left_df, right_df, join_key = None, rounding = 8):
    '''
    Compare differeces betweeen two dataframes. 
    
    Parameters
    ----------
    left_df : Dataframe
        first dataframe to be compared
    right_df : Dataframe
        second dataframe to be compared
    join_key : String or List, optional
        column names to join the DFs. Default value will join by index
    rounding : integer, optional
        number of decimal places to round values being comapred.
        
    Returns
    -------
    Returns a dictionary with the intersecting cols as keys and lists of 
    indcies where the cols in the 2 DFs differ as values. Also includes:
        - left_only_cols: list of col names that are only in left_df
        - right_only_cols: list of col names that are only in the right_df
        - left_only_rows: list of indices that are only in the left_df
        - right_only_rows: list of incdices that are only in the right_df
    '''
    
    # reset index for joining DFs. 
    if join_key != None: # if join_key is none then just join by default index
        
        # if there's index set already then convert them to cols
        if right_df.index.names != [None]:
            right_df.reset_index(inplace = True)
        if left_df.index.names != [None]:
            left_df.reset_index(inplace = True)
        
        # set index to join_key. This will change index of input DFs
        right_df.set_index(join_key, inplace = True)
        left_df.set_index(join_key, inplace = True)
        print('DF index is reset.')
        
    # cols that are only in left DF, right DF and in both DFs
    left_only_cols = list(set(left_df.columns) - set(right_df.columns))
    right_only_cols = list(set(right_df.columns) - set(left_df.columns))
    both_cols = set.intersection(set(left_df.columns), set(right_df.columns))
    
    # subset dfs so they only contain common cols
    left_df2 = left_df.loc[:,both_cols]
    right_df2 = right_df.loc[:,both_cols]
    
    # join dfs on their index
    joined = pd.merge(left_df2, right_df2, how = 'outer',
                      left_index = True, right_index = True,
                      suffixes=('_left', '_right'), indicator = True)
    
    # rows that are only in left DF, right DF and in both DFs
    left_only_rows = joined.loc[joined._merge == 'left_only', :]
    right_only_rows = joined.loc[joined._merge == 'right_only', :]
    both = joined.loc[joined._merge == 'both',: ].round(rounding)
    
    # create dict to store the differences
    diff_dict = {'left_only_cols': left_only_cols,
                  'right_only_cols': right_only_cols,
                  'left_only_rows': list(left_only_rows.index),
                  'right_only_rows': list(right_only_rows.index)}
    
    # row indcies where value differ for each col in the dfs
    for col in both_cols:
        # fill NA bc NaN != NaN with pd.Series.eq()
        equal_bool = both[col + '_left'].fillna(999).eq(both[col + '_right'], 
                                            fill_value = 999)
        diff_idx = equal_bool[equal_bool==False].index
        diff_dict[col] = list(diff_idx)
    
    return(diff_dict)

#%% Code Run Time Log

def popup_msg(code_desc, run_time, start_time, end_time):
    '''
    generates a pop up message to notify that code is done running

    Parameters
    ----------
    code_desc : string
        description of code being run
    run_time : string
        amount of time that code took to run
    start_date : string
        date and time that code started running
    end_date : string
        date and time that code finished running

    Returns
    -------
    None.

    '''
  
    popup = tk.Tk()
    popup.wm_title("Code Runtime Log")
    
    # text in popup message
    desc = ttk.Label(popup, text = code_desc, font=("Verdana", 12))
    desc.pack(pady=3,padx=3)
    runtime = ttk.Label(popup, text= 'Run Time: {}'.format(run_time), 
                        font=("Verdana", 12))
    runtime.pack(pady=3,padx=3)
    starttime = ttk.Label(popup, text= 'Start Time: {}'.format(start_time), 
                          font=("Verdana", 12))
    starttime.pack(pady=3,padx=3)
    endtime = ttk.Label(popup, text= 'End Time: {}'.format(end_time), 
                          font=("Verdana", 12))
    endtime.pack(pady=4,padx=4)
    
    # close button on pop up message
    B1 = ttk.Button(popup, text="Close", command = popup.destroy)
    B1.pack()
    
    # popup.geometry('300x200') # size
    popup.after(1000 * 30, lambda: popup.destroy()) # close after 30 secs
    popup.mainloop() 

def email_msg(code_desc, run_time, start_date, end_date, body):
    '''
    Sends an email message for tracking when code is done running

    Parameters
    ----------
    code_desc : string
        description of code being run
    run_time : string
        amount of time that code took to run
    start_date : string
        date and time that code started running
    end_date : string
        date and time that code finished running
    body : string
        extended description of code being run. is put in body of email

    Returns
    -------
    None.

    '''

    my_email = 'py.notify1@gmail.com'
    prd = 'yourfunny1'
    send_email = 'tommywlang@gmail.com'
    
    email_subject = 'DESC: {} | RUNTIME: {} | START: {} | END: {}'\
        .format(code_desc, run_time, start_date, end_date)
    
    # connect to SMTP server
    s = smtplib.SMTP(host='smtp.gmail.com', port=587)
    s.starttls()
    s.login(my_email, prd)
    
    # create email message
    email = MIMEMultipart()
    email['From'] = my_email
    email['To'] = send_email 
    email['Subject'] = email_subject
    if body != '':
        email.attach(MIMEText(body, 'plain'))
    
    # send email
    s.send_message(email)
    


def msg(start_time = '?', code_desc = '?', extd_desc = '', kind = 'popup'):
    '''
    Logs time for code to run and creates popup or sends email to notify you

    Parameters
    ----------
    start_time : number
        current time in seconds since the Epoch. The default is '?'.
    code_desc : string
        description of code being run. The default is '?'.
    extd_desc : string
        extended description of code being run
    kind : string, optional
        dictates whether a 'popup' or 'email' is made. The default is 'popup'.

    Example
    -------
    start_time = hf.t()
    *** block of code to be logged
    hf.msg(start_time, 'desc of code', 'extended desc of code being run')

    '''
    
    # end time in seconds and end time/date
    end_time = time.time()
    end_date = time.strftime('%m/%d/%y - %H:%M:%S', 
                             time.localtime((end_time)))
    
    # calc start date/time and runtime if possible
    try:
        # date/time of start and elapsed time in seconds
        start_date = time.strftime('%m/%d/%y - %H:%M:%S',
                           time.localtime((start_time)))
        run_time = end_time - start_time
        
        # if run time is greater 180 days then its prob wrong
        if run_time > 180*86400:
            run_time = '?'        
        elif run_time <= 120: # report 2 min or less in seconds
            run_time = str(round(run_time, 1)) + ' sec'
        elif run_time <= 7200: # report 2 hours or less in mins
            run_time = str(round(run_time/60, 1)) + ' min'   
        else: 
            run_time = str(round(run_time/3600, 1)) + ' hr'

    except:
        run_time = '?'
        start_date = '?'
    
    # create pop up message or send email
    if kind == 'popup':
        popup_msg(code_desc, run_time, start_date, end_date)
    if kind == 'email':
        email_msg(code_desc, run_time, start_date, end_date, extd_desc)
    
    # save out to csv file 
    row = {'Code_Desc': code_desc, 'Run_Time': run_time, 
           'Start_Time': start_date, 'End_Time': end_date, 
           'Extended_desc': extd_desc}
    log = pd.read_csv(get_path() + 'code_run_log.csv')
    log = log.append(row, ignore_index=True)
    try:
        log.to_csv(get_path() + 'code_run_log.csv', index = False)
    except:
        log.to_csv(get_path() + 'code_run_log_TEMP.csv', index = False)
        print("saved temp version bc can't overwrite open version")
    
        
def t():
    '''
    returns current time in seconds since the Epoch. used for input to msg()

    '''
    return(time.time())

#%% Run Query Example

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
