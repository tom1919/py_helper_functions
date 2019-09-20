# -*- coding: utf-8 -*-
"""
lots of documentation
"""

#%%
def get_path(name = 'tom'):
    '''
    Returns string for file path to directory
    '''
    if name == 'team':
        ret_path = 'C:/Users/'
    elif name == 'tom':
        ret_path = 'C:/Users/tommy/Dropbox/7. projects/'
    else:
        raise ValueError("Not a valid argument to get_path(). Use 'tom' or 'team'.")
    return ret_path
        
if __name__ == "__main__":
    foo = get_path('tof')
    hf.get_path??
    hf??
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



