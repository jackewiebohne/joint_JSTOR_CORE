import numpy as np
import pandas as pd
from collections import Counter

def normalise(array):
    return (array-np.mean(array))/np.std(array)


def randomised_bin_key_subsampler(dataframe, bin_column, randomise_on, threshold, random_shuffle_bins=False, min_n_files=None, max_count_per_file=None, verbose=True):
    '''
    REQUIRES PANDAS TO WORK! 

    This function randomly subsamples a list of files that has been put into bins
    with pandas.cut. If random_shuffle_bins is False it will simply subsample a
    number of files per bin based on randomise_on. The size of the subsample per bin will
    depend on min_n_files, threshold, and max_count_per_file. If random_shuffle_bins is 
    True it will create simulated (fake) bins that contain a list of files 
    randomly subsampled conditional on bins and the factor to subsample on.
    E.g. if we originally have bins 'Europe' and 'Asia' this function creates
    new bins (as dict) called 'Europe' and 'Asia' where the new 'Europe' contains
    randomly sampled files from the original bins 'Europe' AND 'Asia' based on 
    what the function was supposed to randomise on (see input: randomise_on), 
    a minimal threshold of the sum of randomise_on per bin (see input: threshold),
    and, optionally, the minimum number of files per fake bin (see input: min_n_files).
    It outputs the random list of files so that with this list
    a randomised trial can be run on the files in the list of files.

    Inputs:
        dataframe: pandas dataframe containing following columns:
            - 'filename': filenames
            - should contain column with bins (e.g. time bins). 
            the cell values should be str. The name of this column is specified separately
            in the input 'bin_column'
            - should contain column with values to randomise_on
            the cell values should be int. The name of this column is sepcified separately
            in the input 'ranodmise_on'
        bin_column: str:
            - the column in the dataframe with the bins
        randomise_on: str: 
            - the column in the dataframe based on which we randomly
            subsample. E.g. the count of a keyword per file/filename 
            on which we sample conditional on bin. 
            The cell of randomise_on must hold int

        threshold: int
            - minimum threshold of the total sum of the randomise_on cell
            values that are sampled. E.g. we might want to sample at least
            100 keyword counts per bin
        random_shuffle_bins: bool
            - if True the function will simulate bins for which it subsamples files from 
            original bins. If False (default) it will keep the original bins and subsample
            from those.
        min_n_files: int
            - (optional) sample at least min_n files per bin. might result in 
            much higher threshold than the minimum threshold given as input and
            may result in a more uneven distribution of the sum of randomise_on 
            values over the fake bins
        max_count_per_file: int
            - (optional) maximum count of the randomise_on value for a randomly chosen file.
            this is to prevent that one single file with a high count of the randomise_on value
            dominates the sample. E.g. if we have a total threshold of randomise_on of 100, 
            but one randomly chosen file happens to already have a randomise_on count of 80,
            that one single file might make up 80% of the total random subsample for the fake bin.
            If no value is provided, this number will be int(0.5 * threshold)
        verbose: bool
            - prints out the distribution of actual, original bins over the new fake bins
            - prints out the the sum of randomise_on values per fake bin
    
    Outputs:
        - binned_filenames: dict: list of files per fake bin
    
    '''
    bins = [str(x) for x in dataframe[bin_column].unique() if not isinstance(x, float) and x != None]
    bin_filename_dict = {bin:[] for bin in bins}
    bin_key_dict = {bin:0 for bin in bins}
    if verbose:
        bin_distribution_dict = {bin:Counter() for bin in bins}
        bin_key_distribution_dict = {bin:[] for bin in bins}
    if min_n_files:
        min_files_dict = {bin:0 for bin in bins}
    if max_count_per_file==None:
        max_count_per_file = int(0.5*threshold)

    for single_bin in bins:
        switch = False
        while switch == False:
            if random_shuffle_bins:
                random_bin = np.random.choice(np.array(bins))
                random_filename = np.random.choice(dataframe['filename'][dataframe[bin_column] == random_bin].values)
                sliced_frame = dataframe[dataframe.filename == random_filename]
                if int(sliced_frame[randomise_on]) < max_count_per_file:
                    bin_filename_dict[single_bin].append(random_filename)
                    bin_key_dict[single_bin] += int(sliced_frame[randomise_on])
                    if verbose:
                        bin_distribution_dict[single_bin][sliced_frame[bin_column].str.cat()] += 1
                        bin_key_distribution_dict[single_bin].append(int(sliced_frame[randomise_on]))
                    if min_n_files:
                        min_files_dict[single_bin] += 1
                    if bin_key_dict[single_bin] >= threshold:
                        if min_n_files:
                            if min_files_dict[single_bin] >= min_n_files:
                                switch=True
                        else:
                            switch=True
            else:
                random_filename = np.random.choice(dataframe['filename'][df[bin_column] == single_bin].values)
                sliced_frame = dataframe[dataframe.filename == random_filename]
                if int(sliced_frame[randomise_on]) < max_count_per_file:
                    bin_filename_dict[single_bin].append(random_filename)
                    bin_key_dict[single_bin] += int(sliced_frame[randomise_on])
                    if verbose:
                        bin_distribution_dict[single_bin][sliced_frame[bin_column].str.cat()] += 1
                        bin_key_distribution_dict[single_bin].append(int(sliced_frame[randomise_on]))
                    if min_n_files:
                        min_files_dict[single_bin] += 1
                    if bin_key_dict[single_bin] >= threshold:
                        if min_n_files:
                            if min_files_dict[single_bin] >= min_n_files:
                                switch=True
                        else:
                            switch=True
    if verbose:
        from matplotlib import pyplot as plt
        import math
        if random_shuffle_bins:
            print(f'fake bins and the corresponding sum of randomise_on values: {bin_key_dict} \n')
            print(f'fake bins and the corresponding distribution of summed randomise_on values per original/actual bin: {bin_distribution_dict}')
            print(f'distribution of randomise_on values for files in fake bins: {bin_key_distribution_dict} \n')
            idx = 0
            plot_dims = len(bin_key_distribution_dict.keys())
            fig, axes = plt.subplots(math.ceil(plot_dims/3), 3) # subplot with rows: math.ceil(plot_dims/3) and 3 columns 
            axes = axes.flatten()
            for k,v in bin_key_distribution_dict.items():
                axes[idx].hist(v, bins=len(set(v)))
                axes[idx].set_title(str(k))
                idx += 1
            for idx, ax in enumerate(axes[plot_dims:]): # delete unused axes in case number of subplot%3 != 0
                fig.delaxes(ax)
            fig.tight_layout()
            fig.suptitle('plotted distribution of summed randomise_on values\n')
            fig.subplots_adjust(top=0.85)
            plt.show()
            if min_n_files:
                print(f'number of files per fake bin: {min_files_dict}')
        else:
            print(f'bins and the corresponding sum of randomise_on values: {bin_key_dict}')
            print(f'bins and the corresponding distribution of summed randomise_on values per original/actual bin: {bin_distribution_dict}')
            print(f'distribution of randomise_on values for files in bins: {bin_key_distribution_dict}')
            idx = 0
            plot_dims = len(bin_key_distribution_dict.keys())
            fig, axes = plt.subplots(math.ceil(plot_dims/3), 3) # subplot with rows: math.ceil(plot_dims/3) and 3 columns 
            axes = axes.flatten()
            for k,v in bin_key_distribution_dict.items():
                axes[idx].hist(v, bins=len(set(v)))
                axes[idx].set_title(str(k))
                idx += 1
            for idx, ax in enumerate(axes[plot_dims:]): # delete unused axes in case number of subplot%3 != 0
                fig.delaxes(ax)
            fig.tight_layout()
            fig.suptitle('plotted distribution of summed randomise_on values\n')
            fig.subplots_adjust(top=0.85)
            plt.show()
            if min_n_files:
                print(f'number of files per bin: {min_files_dict}')
    return bin_filename_dict



def __get_filename__(file):
    lst = str(file).split('\\')[-1].split('/')[-1].split('.')
    return lst[-2]


def file_splitter(file_paths, bin_filename_dict, total_file=True, tokenise=True):
    '''
    Creates files, one per bin, based on the output of randomised_bin_key_subsampler

    inputs:
        - file_paths: list
            iterable containing strings with file paths that will be iterated over
        - bin_filename_dict: dict
            output dictionary of randomised_bin_key_subsampler
        - total_file: bool
            if True this will create a file containing the text of all the bins.
            default is True
        - tokenise: bool
            uses simple custom tokeniser to tokenise file text. default is True
    outputs:
        - creates files containing all the texts of the filenames per bin as based 
        on the randomised_bin_key_subsampler (returns: None)
    '''
    if tokenise:
        import nlp_utils as nlp
    if total_file:
        total_filenames = [filename for filename in bin_filename_dict.values()]
        tfile = open(r'./total.txt', 'a', encoding='utf8')

    count = 0
    for filename, single_bin in bin_filenames_dict.items():
        for path in file_paths:
            for file in pl.Path(path).iterdir():
                if file.is_file():
                    _name = __get_filename__(file)
                    if filename == _name:
                        count += 1
                        with open(file, encoding='utf8', errors='ignore') as file_open:
                            f = file_open.read()
                            if tokenise:
                                t = nlp.tokenise(f)
                                t = re.sub('[\(\[\]/\)\'\,\"]','',''.join(str(t)))
                            else:
                                t = f
                                if total_file:
                                    tfile.write(t) # appending to total file
                            with open(r'./'+ str(single_bin) + '.txt', 'w+', encoding='utf8') as file_write:
                                file_write.write(t)
    tfile.close()
    print(f'total number of filenames: {count}')

import pandas as pd
import openpyxl
df = pd.read_excel('./joint_JSTOR_CORE_20.xlsx', engine='openpyxl')
randomised_bin_key_subsampler(df, 'dominant_geolocation', 'keyword_count', 100, False)
