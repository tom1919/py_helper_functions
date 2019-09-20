# -*- coding: utf-8 -*-
"""
The module "helper_functions" contains functions that I frequently use.

@author: tommy

Functions:
    1. get_path(name = 'tom')
    2. set_dir(name = 'tom')
    3. open_db()
    4. close_db()  
    5. update_mod() 
    
Installation:
    1. Copy the file "helper_functions.py" into a PYTHONPATH directory
        - "helper_functions.py" is in '...'
        - Run statements below to get a list of PYTHONPATH directories:
            - import sys
            - sys.path
    2. Import the module by using: import helper_functions as hf
    
TODO:
    4. close_db()  # close db connecntion and remove variables
    5. update_mod() # copy file and put in pythonpath dir
"""

#%%
def get_path(name = 'tom'):
    '''Returns string for file path to directory. Ex: get_path('tom')'''
    if name == 'team':
        ret_path = 'C:/Users/'
    elif name == 'tom':
        ret_path = 'C:/Users/tommy/Dropbox/7. projects/'
    else:
        raise ValueError("Not a valid argument to get_path(). Use 'tom' or 'team'.")
    return ret_path
        
if __name__ == "__main__":
    foo = get_path('tof')
    #get_path??
    #hf??
    set_dir()

#%% 
import os 
def set_dir(name = 'tom'):
    if name == 'team':
        os.chdir(get_path('team'))
    elif name == 'tom':
        os.chdir(get_path('tom'))
    else:
        raise ValueError("Not a valid argument to set_dir(). Use 'tom' or 'team'.")
    

# db connection


# =============================================================================
# put it one of these paths
# import sys
# sys.path
# =============================================================================



