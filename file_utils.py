from time import time
from functools import wraps

def __get_filename__(file):
    lst = str(file).split('\\')[-1].split('/')[-1].split('.')
    filename, filetype = lst[-2], lst[-1]
    return filename, filetype


def get_filename(file):
    lst = str(file).split('\\')[-1].split('/')[-1].split('.')
    return lst[-2]


def file_iterator(folder_path, parse=True, sep=None, encoding='utf8', to_yield=[]):
    ''' 
    Generator for iterating over files
    inputs:
        folder_path:
            - path to the folder with files to parse. If folder path is in fact a file path, it either parses the file
            or yields the filename and filetype (depending on parser input)
        parse: bool
            - parsers supported: pandas with excel & csv  and BeautifulSoup for xml & html (this is done automatically);
            if parse is False it just lazily yields the filename and filetype
        sep: str
            - which separator to use for csv parsing if none is provided it uses commas iff a csv file is parsed
        encoding: str
            - default is 'utf8'
        full_yield: list of strings
            - possible values for list are:
                ['filename', 'number_files','current_count', 'filetypes']
    '''
    import os
    try:
        onlyfiles = [folder_path+f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    except: # if it's not a folder but a file to parse
        onlyfiles = [folder_path]

    if parse or to_yield:
        _, filetypes = map(set,zip(*[__get_filename__(file) for file in onlyfiles]))
        del _
        switch = False
        if 'csv' or 'xlsx' in filetypes:
            import pandas as pd
            switch = True
        if 'html' or 'xml' in filetypes:
            from bs4 import BeautifulSoup
            switch = True
        if 'pickle' or 'pkl' in filetypes:
            import pickle
            switch = True
        if switch == False:
            print('you input parse=True but the files in the directory are not supported for parsing')
            parse = False
    
    counter = -1
    for file in onlyfiles:
        counter += 1
        filename, filetype = __get_filename__(file)
        to_yield_lst = []
        for yieldee in to_yield:
            if yieldee == 'number_files':
                to_yield_lst.append(len(onlyfiles))
            if yieldee == 'current_count':
                to_yield_lst.append(counter)
            if yieldee == 'filetypes':
                to_yield_lst.append(filetypes)
            if yieldee =='filename':
                to_yield_lst.append(filename)
        if filetype == 'csv':
            if sep==None:
                yield (pd.read_csv(file, sep=','),  *to_yield_lst)
            else:
               yield (pd.read_csv(file, sep=sep),  *to_yield_lst)
        elif filetype == 'xlsx':
            yield (pd.read_excel(file, engine='openpyxl'),  *to_yield_lst)
        elif filetype == 'xml':
            with open(file, encoding=encoding, errors='ignore') as xml:
                yield (BeautifulSoup(xml, 'lxml'),  *to_yield_lst)
        elif filetype == 'pickle' or filetype == 'pkl':
            with open(file, 'rb') as handle:
                yield (pickle.load(handle), *to_yield_lst)
        elif parse == False:
            print(f'filename: {filename}, filetype: {filetype}')
            yield (file, *to_yield_lst)
        else:
            print(f'filename: {filename}, filetype: {filetype}')
            yield (file, *to_yield_lst)


def function_timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time()
        res = func(*args, **kwargs)
        end = time()
        return res, end-start
    return wrapper


# path = r'C:\Users\hwx756\Downloads/' 

# # @function_timer
# for i in file_iterator(path, parse=True, to_yield=['number_files','filetypes', 'current_count']):
#     file, number_files, filetypes, current_count = i
#     print(current_count)