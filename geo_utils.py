import unidecode
import pandas as pd
from collections import Counter
import pathlib as pl
import re, os, time, string
import numpy as np


convert_dict_temp = {'Broadway': 'New York City', 'D.C.': 'Washington', 'Washington D.C.': 'Washington', 'D. C.': 'Washington', 'New York': 'New York City',
           'West Oakland' : 'Oakland', 'East Oakland': 'Oakland', 'The City of Chicago':'Chicago', 'U. S. A' : 'United States', 'U.S.A': 'United States', 'U.S.': 'United States', 'N.Y.C': "New York City", 'N.Y.': "New York City", 'U. S.': 'United States',
           'St. Kitts-Nevis':'St Kitts','Saint Kitts and Nevis':'St Kitts', 'NYC': 'New York City', 'N.Y.C.':'New York City',
           'New York': 'New York City','Federal Republic of Germany': 'Germany', 'Quebec City': 'Quebec', 'U.K.': 'United Kingdom', 'United States of America': 'United States', 
           'N.Y.': 'New York City', 'South Central L.A.': 'Los Angeles', 'South L.A.': 'Los Angeles', 'l.a.':'Los Angeles',
           'Minneap': 'United States', 'Ore.': 'United States', 'Ark.':'United States', 'PRC': 'China', 'P.R.C.': 'China',
            'People\'s Republic of China': 'China', 'Burma':'Myanmar', 'Democratic Republic of the Congo': 'Dr Congo', 'DRC': 'Dr Congo',
            'American Samoa' : 'United States overseas','Guam': 'United States overseas', 'Northern Mariana Islands': 'United States overseas',
            'Wake Islands':'United States overseas', 'Cook Islands': 'New Zealand', 'French Polynesia': 'France overseas', 'New Caledonia':'France overseas',
            'Niue':'New Zealand', 'Tokelau':'New Zealand', 'Pitcairn': 'United Kingdom overseas',  'Pitcairn Island': 'United Kingdom overseas',
            'Anguilla':'United Kingdom overseas', 'Aruba':'Netherlands overseas','Bermuda':'United Kingdom overseas','Bonaire':'Netherlands overseas',
             'British Virgin Islands':'United Kingdom overseas', 'Cayman Islands':'United Kingdom overseas','Clipperton Island':'France overseas',
             'Curacao':'Netherlands overseas','Greenland':'Denmark overseas','Guadeloupe':'France overseas','Martinique':'France overseas',
             'Montserrat':'United Kingdom overseas','Navassa Island':'United States overseas','Puerto Rico':'United States overseas',
             'Saba':'Netherlands overseas','Saint Barthelemy':'France overseas','Saint Martin':'France overseas', 'Saint Pierre and Miquelon':'France overseas',
            'Sint Eustatius':'Netherlands overseas','Sint Maarten':'Netherlands overseas','Turks and Caicos':'United Kingdom overseas',
            'US Virgin Islands':'United States overseas', 'Palestinian Territory':'Palestine', 'Republic Of The Congo':'Congo-Brazzaville',
            'United States Minor Outlying Islands':'United States overseas', 'Republic Of The Congo': 'Congo-Brazzaville', 'Guam': 'United States overseas',
            'Vatican City': 'Vatican', 'Curacao':'Netherlands overseas', 'Netherlands Antilles': 'Netherlands overseas', 'the Congo': 'Congo-Brazzaville',
            'American Samoa': 'United States overseas', 'French Polynesia':'France overseas', 'Congo Republic':'Congo-Brazzaville',
            'Isle of Man': 'United Kingdom', 'French Southern Territories': 'France overseas','U.S. Virgin Islands':'United States overseas',
            'British Virgin Islands': 'United Kingdom overseas', 'Faroe Islands':'Denmark overseas', 'Greenland': 'Denmark overseas',
            'Falkland Islands': 'United Kingdom overseas', 'Saint Helena': 'United Kingdom overseas', 'Saint Kitts and Nevis': 'St Kitts',
            'Saint Kitts': 'St Kitts', 'Turks and Caicos Islands': 'United Kingdom overseas','Bermuda':'United Kingdom overseas', 'Martinique':'France overseas',
            'New Caledonia': 'France overseas', 'Wallis and Futuna': 'France overseas','Saint Barthelemy':'France overseas','Reunion':'France overseas',
            'Aruba':'Netherlands overseas', 'Saint Martin':'France overseas', 'French Guiana':'France overseas','Republic of the Congo': 'Congo-Brazzaville',
            'Tahiti':'France overseas','Aland Islands':'Finland', 'Åland': 'Finland', 'Åland Islands': 'Finland',
            'Gibraltar':'United Kingdom overseas', 'Anguilla':'United Kingdom overseas','Svalbard and Jan Mayen':'Norway',
            'Svalbard':'Norway', 'Jan Mayen':'Norway','South Georgia and the South Sandwich Islands':'United Kingdom overseas',
            'South Georgia':'United Kingdom overseas', 'South Sandwich Islands':'United Kingdom overseas','Bouvet Island':'Norway',
            'Sint Maarten':'Netherlands overseas', 'Montserrat':'United Kingdom overseas', 'Bonaire, Saint Eustatius and Saba':'Netherlands overseas',
            'Mayotte':'France overseas', 'Cabo Verde': 'Cape Verde', 'British Indian Ocean Territory':'United Kingdom overseas',
            'Jersey':'United Kingdom overseas', 'Norfolk Island':'Australia', 'Antigua':'Antigua and Barbuda', 'Cocos Islands':'Australia',
            'Saint Pierre and Miquelon':'France overseas','Guadeloupe':'France overseas','East Timor':'Timor Leste','Guernsey':'United Kingdom overseas',
            'Christmas Island':'Australia', 'Heard Island and McDonald Islands':'Australia', 'Heard Island':'Australia', 'McDonald Islands':'Australia',
            'Bonaire, Saint Eustatius and Saba ':'Netherlands overseas', 'East Germany': 'Germany', 'GDR': 'Germany', 'G.D.R':'Germany',
            'Britain': 'United Kingdom', 'Great Britain': 'United Kingdom', 'England':'United Kingdom', 'Scotland':'United Kingdom','Wales':'United Kingdom',
            'Northern Ireland':'United Kingdom', 'u.s.s.r.':'ussr', 'u. s. s. r.':'ussr','Soviet Union':'USSR', 'Union of Soviet Socialist Republics': 'USSR', 'Northamerica': 'North America', 'Southamerica':'South America',
            'Big Apple':'New York City','N. Y. City':'New York City','N.Y. City': 'New York City', 'N.Y.City': 'New York City','Cote D\'Ivoire':'Ivory Coast',
            'Soviet Block': 'USSR', 'Subsaharan Africa':'Africa', 'West Africa': 'Africa', 'East Africa':"Africa", 'North Africa':'Africa', 'Federal Germany': 'Germany',
            'Wild West': 'United States', 'New England':'United States', 'Macedonia': 'North Macedonia', 'German Reich': 'Germany', 'Third Reich':'Germany',
                     'punjab': 'India', 'pacific region': 'Oceania','pacific asia':'Oceania', 'new south wales': 'Australia', 'north sea':'Europe', 'calif':'United States', 'america':'united states',
                     'westkingston':'Kingston', 'uk': 'United Kingdom', 'mass.':'United States', 'new york state':'United States', 'newyorkpagepage': 'New York City',
                    'Pomerania':'Poland', 'surinam':'Suriname', 'mich.':'United States', 'Prussia':'Germany', 'u.s.a.':'United States', 'washington dc':'Washington',
                     'mediterranean':'Europe', 'calcutta':'kolkata', 'dutchantilles':'Netherlands overseas', 'conn.':'United States', 'czechoslovakia':'Czechia', 'ill.':'United States',
                     'westgermany':'Germany', 'westindies' : 'Caribbean', 'orient':'Asia', 'southbronx':'New York City', 'witwatersrand':'Johannesburg',
                     'longisland':'New York City', 'wis.':'United States', 'xinjiang': 'China', 'ind.':'United States', 'zaire':'dr congo', 'southflorida':'United States',
                     'britishcolumbia':'Canada', 'britishempire':'United Kingdom', 'minn.':'United States', 'colo':'United States', 'unitedstatesv':'united states',
                     'britishguiana':'United kingdom overseas', 'new guinea': 'Australia', 'zealand':'Denmark', 'republicofafghanistan':'Afghanistan', 'islamicrepublicofpakistan':'Pakistan'
                    } #'Serbia and Montenegro':'Serbia',


convert_dict = {}
for key, value in convert_dict_temp.items():
    convert_dict[key.replace('\n', '')] = value



def geotokenise(wordstring):
    """
	this function tokenises geodata. it keeps acronyms, but removes other clutter around geodata (see below)
    requires the unicode module
    """
    starts = [('the', ''), ('this', ''),('greater', ''),('Northern', ""), ('Southern', ''), ('Eastern', ''), ('Western', ''), 
               ('Northwestern',''), ('Northeastern', ''), ('Southeastern',''), ('Southwestern', ''), 
               ('lower',  ''), ('higher', ''), ('upper', ''), ('central', ''), ('Republic of', ''), ('city of', ''),
             ('district of', ''), ('kingdom of', ''), ('kingdom of the', ''), ('islamic republic of', ''), ('area of', ''),
             ('county of', ''), ('west end of', ''), ('west end', ''), ('westend', ''), ('westend of', ''), ('east end of', ''),
             ('east end', ''), ('eastend of', ''), ('eastend', ''), ('southeast', ''), ('st.', 'Saint'), ('st', 'Saint')] 
    ends = [('\'s', ''), ('district', ''), ('area', ''), ('county', '')]
    cleanstr = ''
    
    if type(wordstring) != str: 
        return
    
    ## remove all numbers and odd chars   
    wordstring = wordstring.lower()
    
    if wordstring == 'western sahara' or wordstring == 'westernsahara':
        return 'westernsahara'
    
    if wordstring == 'the west' or wordstring == 'thewest':
        return 'thewest'
    
    if wordstring == 'the east' or wordstring == 'theeast':
        return 'theeast'
    
    if wordstring == 'republic of the congo' or wordstring == 'republicofthecongo':
        return 'congobrazzaville'
        
    for k,v in starts:
        if wordstring.startswith(k.lower()):
            wordstring=wordstring.replace(k.lower(),v.lower())
            
    for k,v in ends:
        if wordstring.endswith(k.lower()):
            wordstring=wordstring.replace(k.lower(), v.lower())
    
    f = re.search(r'([a-zA-Z]\.){2,}',wordstring)# matches acronyms; round brackets mean it will search for a sequence of these letters; curly brackets + number + comma mean 2 or more times
    if f:
        wordstring = re.sub(r'[^a-zA-Z.]', '', wordstring)
    else:
        wordstring = re.sub(r'[^a-zA-Z]', '', wordstring) # keeps everything within the square brackets, the ^ is a negation sign, i.e. everything inside square brackets after the ^ sign will not be removed

    cleanstr = unidecode.unidecode(wordstring) # unidecode turns unicode chars into their nearest ascii equivalent

    return cleanstr


def cf_geodata_vec(token_str):
    """
    this function is not context free and requires that the lookup excel 
    (with districts, cities, countries and continents) is loaded as a dataframe (see above).
    the function outputs a tuple as follows:
    	unknown (i.e. not found in lookup table), city, country, continent 
    
    """
    if type(token_str) != str:
        return None, None, None, None # the return pattern is always: unidentified, city, country, continent
    
    convtoken = ''
    for k,v in convert_dict.items(): ## convert geodata: e.g. NYC = New York City
        if token_str == geotokenise(k):
            convtoken = geotokenise(v)
        else:
            continue
            
    if not convtoken:
        convtoken = token_str
    
    if convtoken in tokenstates: # US states
        return None, None,'unitedstates', 'northamerica' # the return pattern is always: unidentified/number, city, country, continent
    
    result = df_cf[['city', 'country', 'continent']].loc[(convtoken == df_cf['sub_city'].values)| \
                                                         (convtoken == df_cf['city'].values)|\
                                                         (convtoken == df_cf['country'].values) | \
                                                         (convtoken == df_cf['continent'].values)].values
    if result.shape[0] != 0:
        return None, result[0][0], result[0][1], result[0][2]
    else:
        return convtoken, None, None, None



def evaluate_geodata(geostring):
    '''
    This is the toplevel function that does it all. The above are only helpers
    Input:
        takes an unevaluated geostring based on the spaCy NER parser (see my named entity parser function in nlp_utils)
    Output:
    	Counter() object for each of the following: unknown (i.e. not found in lookup table), continent, country, city, denomym
    	(demonym is whatever spaCy's classifier labels 'NORP')

    also tested with:
        res = list(map(lambda x: cf_geodata_vec(geotokenise(x[0])) if x[1] != 'NORP' else \
        (x[0][:-1] if x[0].endswith('s') else x[0]), eval(geostring)))
        
        return Counter([tple[0] for tple in res if type(tple) == tuple]), Counter([tple[1] for tple in res if type(tple) == tuple]),\
        Counter([tple[2] for tple in res if type(tple) == tuple]),Counter([tple[3] for tple in res if type(tple) == tuple]),\
        Counter([string for string in res if type(string) == str])

    but slower performance: 203 ms ± 32.2 ms per loop (mean ± std. dev. of 7 runs, 1 loop each) 
    as opposed to current: 186 ms ± 31.2 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    '''
    unknown, continent,country,city, demonym = [],[],[],[],[]
    if geostring:
        for geoentity, classifier in eval(geostring):
            if classifier != 'NORP':
                token = geotokenise(geoentity)
                _unknown, _city,_country,_continent  = cf_geodata_vec(token)
                unknown.append(_unknown)
                country.append(_country)
                continent.append(_continent)
                city.append(_city)
            else:
                if geoentity.endswith('s'):
                    demonym += [geoentity[:-1]]
                else:
                    demonym += [geoentity]
    return Counter(unknown), Counter(continent),Counter(country),Counter(city), Counter(demonym) 



#### crucial lookup tables here ###

states = pd.read_csv('./JSTOR data/US states.txt')
states = np.concatenate((states['State'].values, states['Ab1'].values, states['Ab2'].values))
tokenstates = np.array([geotokenise(x) for x in states if type(x)!= float], dtype=str)
df_cf = pd.read_excel('./JSTOR data/general_geo/geotokenised_districts_cities_countries_conts.xlsx', engine='openpyxl')