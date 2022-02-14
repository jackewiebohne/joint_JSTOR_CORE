import re, os, string
import unidecode
import numpy as np
from collections import Counter
import spacy
from scipy.stats.distributions import chi2

from nltk.corpus import stopwords
stop_words_en = stopwords.words('english') + ['however','around', 'pp','amp','although','rather','might','im', 'must','dont','tion', 'still', 'ing','also', 'would', 'could', 'ain\'t', 'even', 'us', 'one', 'two', 'three', 'may']
from nltk.stem import PorterStemmer

####
# German stopword list
stop_words_ger = ['und', 'oder', 'wie', 'der', 'die', 'das', 'des','dem', 'ein', 'eine', 'eines', 'einen', 'einem', 'aber', 'von', 'vielfältig', 'vielfaeltig',
'einmal', 'zweimal', 'dreimal', 'viermal', 'fünfmal', 'fuenfmal', 'zigmal', 'einfach', 'zweifach', 'dreifach', 'vierfach', 'fünffach', 'fuenffach', 'zigfach',
'davon', 'dabei', 'schon', 'er', 'ihn', 'ihm', 'es', 'immer', 'mehr', 'gleich', 'darin', 'danach', 'denen', 'deren', 'heran', 'mitunter', 'zugleich', 'dergleichen', 'hinaus', 'heraus', 'vorher', 'nachher', 'hinüber',
'sein', 'seine', 'seines', 'seinem', 'seiner', 'an', 'in', 'auf', 'aus', 'unter', 'über','ueber', 'oben', 'zwischen', 'um', 'jener', 'jene', 'jenes', 'jenem',
'seit', 'zu', 'so', 'worin', 'nach', 'auch', 'den', 'bei', 'daß', 'eben', 'bald', 'wieder', 'weiter', 'weitere', 'weiteres', 'weiterem',
'alle', 'allem', 'allen', 'aller', 'alles', 'als', 'also', 'am', 'ander', 'andere', 'anderem', 'anderen', 'anderer', 'anderes', 'anderm', 'andern',
'anders', 'bin', 'bis', 'bist', 'da', 'damit', 'dann', 'dass', 'derselbe', 'derselben', 'denselben', 'desselben', 'demselben', 'dieselbe', 'dieselben', 'dasselbe',
'dazu', 'dein', 'deine', 'deinem', 'deinen', 'deiner', 'deines', 'denn', 'derer', 'dessen', 'dich', 'dir', 'du', 'dies', 'diese', 'diesem', 'diesen', 'dieser',
'dieses', 'doch', 'dort', 'durch', 'einer', 'einig', 'einige', 'einigem', 'einigen', 'einiger', 'einiges', 'einmal', 'etwas', 'euer',
'eure', 'eurem', 'euren', 'eurer', 'eures', 'für', 'fuer', 'gegen', 'gewesen', 'hab', 'habe', 'haben', 'hat', 'hatte', 'hatten', 'hier', 'hin', 'hinter', 'ich', 'mich',
'mir', 'ihr', 'ihre', 'ihrem', 'ihren', 'ihrer', 'ihres', 'euch', 'im', 'indem', 'ins', 'ist', 'jede', 'jedem', 'jeden', 'jeder', 'jedes', 'jenen', 'jetzt', 'kann',
'kein', 'keine', 'keinem', 'keinen', 'keiner', 'keines', 'können', 'koennen', 'könnte', 'koennte', 'machen', 'man', 'manche', 'manchem', 'manchen', 'mancher', 'manches', 'mein', 'meine',
'meinem', 'meinen', 'meiner', 'meines', 'mit', 'muss', 'musste', 'noch', 'nun', 'nur', 'ob', 'ohne', 'sehr', 'seinen', 'selbst', 'sich', 'sie',
'ihnen', 'sind', 'solche', 'solchem', 'solchen', 'solcher', 'solches', 'soll', 'sollte', 'sondern', 'sonst', 'uns', 'unsere', 'unserem', 'unseren', 'unser',
'unseres', 'viel', 'vom', 'vor', 'während', 'waehrend', 'war', 'waren', 'warst', 'was', 'weg', 'weil', 'welche', 'welchem', 'welchen', 'welcher', 'welches', 'wenn', 'werde',
'werden', 'will', 'wir', 'wird', 'wirst', 'wo', 'wollen', 'wollte', 'würde', 'wuerde', 'würden', 'wuerden', 'zum', 'zur', 'zwar','nicht', 'nichts','non', 'et']
####



def keygram_vec(wordlist, keyword, window_size=5, equalise=False, return_type=None):
    '''
    vectorised n-gram function for a keyword.
    inputs:
        wordlist = ordered list of words (the reason for using list, not str, as input is that for better accuracy
                    of keygram_vec the original string should be tokenised anyway and then the list of tokens can be
                    input into this function)
        keyword (string) = keyword to get ngram for
        window_size (int) = takes window_size many words on either side of keyword
        equalise (boolean) = if e.g. set to False function will only return as much as or less than window_size many words
                            to the right and left of the keyword. if set to True, function will always return (window_sizw*2 + keyword)
                            many words (if keyword_index-window_size is shorter than document, the difference will be added to the right;
                            and vice versa if keyword_index+window_size is longer than document). In short, the keygrams returned will 								all be of equal length, regardless of keywords' index positions in the document
        return_type (str) = possible inputs: 'str': list['str'], 'list': list[list], default: list[arrays]
    '''
    assert(type(wordlist)==list)
    arr = np.array(wordlist)
    L = len(arr)
    indexes = np.where(arr==keyword)[0] # returns tuple(array, array_dtype), but we only want the arrays, hence the [0] slice
    
    if indexes.shape[0]==0: # if nothing is found return empty list
        return []

    left = indexes[0] - window_size # to check if we can grab window_size many words to left of first keygram, if negative, we can't
    right = indexes[-1] + window_size # to check if we can grab window_size many words to right of last keygram, if right > L, we can't
    
    keygram_list = []
    
    # first find of keyword
    if left >= 0:
        keygram_list.append(arr[indexes[0]-window_size:indexes[0]+window_size+1])
    else:
        if not equalise:
            try:
                keygram_list.append(np.hstack((arr[:indexes[0]], arr[indexes[0]:indexes[0]+window_size+1])))
            except:
                # exception needed, in case arr[:indexes[0]] is empty (i.e. keyword is at very beginning of doc)
                keygram_list.append(arr[indexes[0]:indexes[0]+window_size+1])
        else:
            try:
                keygram_list.append(np.hstack((arr[:indexes[0]], arr[indexes[0]:indexes[0]+window_size+abs(left)+1])))
            except:
                keygram_list.append(arr[indexes[0]:indexes[0]+window_size+abs(left)+1])
                
    # any find of keyword other than first and last
    for i in range(1, len(indexes)-1):
        keygram_list.append(arr[indexes[i]-window_size:indexes[i]+window_size+1])
    
    #last find of keyword
    if indexes[0] != indexes[-1]: # make sure first and last keyword are not the same
        if right > L:
            if not equalise:
                try:
                    keygram_list.append(np.hstack((arr[indexes[-1]-window_size-1:indexes[-1]], arr[indexes[-1]:])))
                except:
                    keygram_list.append(arr[indexes[-1]-window_size-1:indexes[-1]])
            else:
                try:
                    keygram_list.append(np.hstack((arr[indexes[-1]-window_size-(right-L)-1:indexes[-1]], arr[indexes[-1]:])))
                except:
                    keygram_list.append(arr[indexes[-1]-window_size-(right-L)-1:indexes[-1]])
        else:
            keygram_list.append(arr[indexes[-1]-window_size-1:indexes[-1]+window_size])

    if not return_type:
        return keygram_list
    elif return_type == 'list':
        keygram_list = [ele.tolist() for ele in keygram_list]
        return keygram_list
    elif return_type == 'str':
        keygram_list = [' '.join(ele) for ele in keygram_list]
        return keygram_list
    else:
        raise ValueError('Only "str" (returns list of str), "list" (list of lists), or "None" (list of np.arrays) allowed')



def tokenise(wordstring, stem=False, lang='en', min_token_len=2):
    '''
    Tokeniser: removes punctuation, numbers, and stopwords. lowercases tokens and returns a list

    stemming works for for English only
    otherwise supports tokenisation for German: lang='ger'
    min_token_len is the minimal length of the token for the token to be kept

    '''

    str1 = PunctuationRemover(wordstring).replace('\\n', '').replace('\\u', '')
    if lang!='en' and lang!='ger':
        raise ValueError('only supports English and German')
    if lang=='ger':
        if stem:
            raise ValueError('stemmer currently only supports English')
        return [w for w in str1.lower().split() if w not in stop_words_ger and w.isalpha() and len(w)>=min_token_len]
    if lang=='en':
        if stem:
            porter = PorterStemmer()
            lst1 = [porter.stem(w) for w in str1.lower().split() if w not in stop_words_en and w.isalpha() and len(w)>=min_token_len]
        else:
            lst1 = [w for w in str1.lower().split() if w not in stop_words_en and w.isalpha() and len(w)>=min_token_len]
        return lst1



def jaro (string1, string2):
    '''AN APPLICATION OF THE FELLEGI-SUNTER MODEL OF RECORD LINKAGE TO THE 1990 U.S. DECENNIAL CENSUS
    William E. Winkler and Yves Thibaudeau U.S. Bureau of the Census '''
    '''or see also: https://www.geeksforgeeks.org/jaro-and-jaro-winkler-similarity/'''
    if len(set(string1) & set(string2)) == 0:
        return 0
    
    if string1 == string2:
        return 1

    # matches
    cnt = (max(len(string1), len(string2))//2) - 1
    assigned1 = np.array(['/']*len(string1))
    assigned2 = np.array(['/']*len(string2))
    
    m = 0
    for i in range(len(string1)):
        for j in range(max(0,(i-cnt)), min(len(string2), i+cnt+1)): # to limit the number of operations. the loop starts at i-max_length and only goes to either i+max_length+1 (since range is exclusive) or (if the end of string2 is closer than max_length) to the end of string2
            if string1[i] == string2[j] and assigned1[i] == '/' and assigned2[j] == '/': # the last two condition are so that there is no overwrite of already assigned values
                assigned1[i] = string1[i]
                assigned2[j] = string2[j]
                m += 1
    if m == 0:
        return 0 
    
    assigned1 = np.delete(assigned1, np.where(assigned1 == '/')) # removing all the spaceholders
    assigned2 = np.delete(assigned2, np.where(assigned2 == '/'))
    
    # transpositions
    t = 0
    if all(assigned1 == assigned2):
        t = 0
    elif all(assigned1 == '/') and all(assigned2 == '/'):
        t = 0
    else:
        t = sum(np.compare_chararrays(assigned1, assigned2, '!=', False).astype(int))//2 # vectorised for minor performance improvement; saves us the for loop commented out below
        
    return  ((m/len(string1) + m/len(string2) + (m-t)/m) / 3)



def jaro_winkler(string1, string2, scaling=0.1):
    jaro_dist = jaro(string1,string2)
    
    if jaro_dist == 1:
        return 1
    
    if jaro_dist >= 0.6:
        prefix = 0
        for i in range(min(5, min(len(string1), len(string2)))):
            if string1[i] == string2[i]:
                prefix += 1
            else:
                break
        return (jaro_dist + prefix * scaling * (1 - jaro_dist))
    
    else:
        return jaro_dist



def word_counter(wordstring):
    return sum(1 for x in wordstring.split())




def PunctuationRemover(wordstring, return_list=False):
    assert type(wordstring) == str
    a = wordstring
    translator = str.maketrans('', '', string.punctuation)
    g = a.translate(translator)
    h = re.sub('[\n”“\-«»’]', '', g)
    if return_list == True:
        h = h.split()
    return h




def named_entity_recog(wordstring):
    """
    requires that spacy is installed and 'en_core_web_trf' is downloaded
    outputs a list
    """
    ocr_split = wordstring.split()
    nlp = spacy.load('en_core_web_trf')#("en_core_web_sm")
    ner_data = []
    _words = ''
    if len(ocr_split) >= 100000:
        for word in ocr_split:
            _words += word + ' '
            if len(_words.split()) == 100000:
                for entity in nlp(_words).ents:
                    if entity.label_ == 'NORP' or entity.label_ == 'LOC' or entity.label_ == 'GPE':
                        ner_data.append((entity.text, entity.label_))
                _words = ''

    else:
        for entity in nlp(wordstring).ents:
            if entity.label_ == 'NORP' or entity.label_ == 'LOC' or entity.label_ == 'GPE':
                ner_data.append((entity.text, entity.label_))
    return ner_data



def cos_similarity(x,y):
    return x@y/(np.linalg.norm(x)*np.linalg.norm(y))



class Collocation_Metrics():
    """
    class to calculate some standard collocation metrics commonly used in 
    computational linguistics. there are a few collocation metrics that are not
    included here, however.

    params::
        self.N = total tokens
        self.C = collocate frequency in total corpus
        self.W = frequency of target token
        self.WC = frequency of co-occurrence of token and collocate


    for these formulas, see Brezina: statistics in corpus linguistics, p. 72. 
    t-score corrected by window size seems to be the least extreme in 
    susceptibility to frequency and exclusivity (see ibid., p. 74)
    """
    def __init__(self,N, C, W, WC):
        self.N = N # total words
        self.C = C # collocate frequency in total corpus
        self.W = W # frequency of target token/word
        self.WC = WC # frequency of co-occurrence of word and collocate
        
    def _expected(self):
        return (self.W*self.C)/self.N
    
    def _corrected_expected(self, window_size):
        return (self.W*self.C*window_size)/self.N
    
    def MI(self):
        if self._expected():
            return np.log2(self.WC/self._expected())
        else:
            return 0
    
    def MI2(self):
        if self._expected():
            return np.log2(self.WC**2/self._expected())
        else:
            return 0
    
    def MI3(self):
        if self._expected():
            return np.log2(self.WC**3/self._expected())
        else:
            return 0
    
    def z_score(self):
        if self._expected():
            return (self.C - self._expected())/np.sqrt(self._expected())
        else:
            return 0
    
    def t_score(self):
        if self.WC:
            return (self.WC - self._expected())/np.sqrt(self.WC)
        else:
            return 0
    
    def corrected_t_score(self, window_size):
        '''
        params::
            window_size = the window of collocations considered 
                (collocation = collocation of target token and collocate token)
        ''' 
        if self.WC:
            return (self.WC - self._corrected_expected(window_size))/np.sqrt(self.WC)
        else:
            return 0

def log_likelihood_corpus_cf(total_tokens_reference_corpus, total_tokens_comparison_corpus,
 target_frequency_reference, target_frequency_comparison, significance_level=0.05, verbose=True):
    """
    Conduct log-likelihood test of word frequencies between a comparison corpus and a reference corpus.
    
    params ::
        arguments are self-explanatory
        if verbose, function prints p-value and log-likelihood and states whether or not H0 should be rejected

    for source see Brezina, Statistics in Corpus Linguistics, pp. 84-5
    returns: tuple of the raw log-likelihood and the p-value at the given significance level (using chi-square approx. 
    with 1 degree of freedom)
    """
    TR = total_tokens_reference_corpus
    TC = total_tokens_comparison_corpus
    O21 = target_frequency_reference
    O11 = target_frequency_comparison
    E11 = (TC * (O11 + O21))/(TC + TR)
    E21 = (TR * (O11 + O21))/(TC + TR)
    LL = 2 * (O11 * np.log(O11/E11) + O21 * np.log(O21/E21))
    p_val = chi2.sf(LL, 1) # 1 = degree of freedom
    if verbose:
        print('Log-likelihood: ', LL, '\np < ', p_val)
        if p_val <= significance_level:
            print('reject H0')
        else:
            print('accept H0')
    return (LL, p_val)