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

port = port_number
service_name = db_name
dis = user_name
guise = password

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
