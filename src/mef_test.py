import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from scipy import stats
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import textdistance
from collections import Counter
import os
import pandas as pd
import numpy as np
import pickle

def make_binary(data, col_name, pos_value):
    return data[col_name].map(lambda x: 1 if x == pos_value else 0)




def calc_significance(data, ind_col_name, ind_value, dep_col_name, dep_value):
    sample_1 = make_binary(data, dep_col_name, dep_value)
    sample_2 = make_binary(data[data[ind_col_name]==ind_value], dep_col_name, dep_value)
    
    return stats.ttest_ind(sample_1, sample_2)[1]




def calc_prcnts(data, col_name=None, value=None):
    if col_name:
        data = data[data[col_name] == value]
    
    total = len(data)
    funct = make_binary(data, 'status_group', 'functional')
    non_funct = make_binary(data, 'status_group', 'non functional')
    repair = make_binary(data, 'status_group', 'functional needs repair')

    prcnt_funct = round(funct.mean()*100, 2)
    prcnt_non_funct = round(non_funct.mean()*100, 2)
    prcnt_repair = round(repair.mean()*100, 2)
    
    return prcnt_funct, prcnt_repair, prcnt_non_funct




def create_replacer(list_of_misspellings, correct_spelling):
    return {key:correct_spelling for key in list_of_misspellings}




def calc_prcnt_by_status(df, status, value=None, col_name=None):
    if col_name:
        df = df[df[col_name] == value]
    total = len(df)
    num_status = len(df[df['status_group'] == status])
    
    return round(num_status/total*100, 2)



def analyze_by_categorical(dataframe, col_name):
    
    possible_values = list(dataframe[col_name].unique())
    column = col_name
    years = dataframe.groupby(col_name)['construction_year'].median()
    
    fig, ax = plt.subplots(len(possible_values), 1, figsize=(8,7*len(possible_values)))

    for index in range(len(possible_values)):
        value = possible_values[index]
        median_year = years[value]
        
        other_wells = dataframe[dataframe[col_name]!= value]
        
        labels = ['Functional', 'Functional - Needs Repair', 'Non-functional']
        data_set_prcnts = calc_prcnts(other_wells)
        subset_percents = calc_prcnts(dataframe, col_name=column, value=value)

        x = np.arange(len(labels))  # the label locations
        width = 0.3  # the width of the bars


        rects1 = ax[index].bar(x + width/2, data_set_prcnts, width, label=f'With {col_name} not equal to {value}', align='center')
        rects2 = ax[index].bar(x - width/2, subset_percents, width, label=f'With {col_name} equal to {value}', align='center')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax[index].set_ylabel('Percent of Rating')
        ax[index].set_xlabel(f'Median Construction Year: {median_year}')
        
        ax[index].set_title(f'Rates of functionality for Water Points\n With {col_name} equal to {value}\nvs the Whole Data Set')
        ax[index].set_xticks(x)
        ax[index].set_xticklabels(labels)
        ax[index].legend()
        ax[index].set_ylim(0,100)

        autolabel(rects1, ax, index=index)
        autolabel(rects2, ax, index=index)

    fig.tight_layout()

    plt.show()
    
def analyze_subset(dataframe, col_name, value):
    
    possible_values = list(dataframe[col_name].unique())
    column = col_name

    fig, ax = plt.subplots(1, 1, figsize=(12,12))
    labels = ['Functional', 'Functional - Needs Repair', 'Non-functional']
    data_set_prcnts = calc_prcnts(dataframe)
    subset_percents = calc_prcnts(dataframe, col_name=column, value=value)

    x = np.arange(len(labels))  # the label locations
    width = 0.3  # the width of the bars


    rects1 = ax.bar(x + width/2, data_set_prcnts, width, label='Total Data Set', align='center')
    rects2 = ax.bar(x - width/2, subset_percents, width, label=f'With {col_name} equal to {value}', align='center')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Percent of Rating')
    ax.set_title(f'Rates of functionality for Water Points\n With {col_name} equal to {value}\nvs the Whole Data Set')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0,100)


    autolabel(rects1, ax)
    autolabel(rects2, ax)


    fig.tight_layout()

    plt.show()
    
    
    
def autolabel(rects, ax, index = None):
    """Attach a text label above each bar in *rects*, displaying its height."""
    if index!=None:
        for rect in rects:
            height = rect.get_height()
            ax[index].annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    else:
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    
    
def find_misspellings(word, list_of_words):
    Ratios = process.extract(word, set(list_of_words), limit=100)
    total_entries = len(list_of_words)
    misspellings = []
    for item in Ratios:
        if item[1]>85:
            misspellings.append(item[0])
    return misspellings

def create_replacer(list_of_misspellings, correct_spelling):
    return {key:correct_spelling for key in list_of_misspellings}







import os
import pandas as pd
import numpy as np
import pickle


def load_train_add_target_df():
    '''This function loads the three
    datasets from the data folder and
    creates a dataframe for each dataset.
    The train data and train target 
    datasets are merged together to 
    complete the training dataset with
    the target label'''
    train_data = pd.read_csv('../data/train_data.csv')
    train_target = pd.read_csv('../data/train_targets.csv')
    train = train_data.merge(train_target, on='id', how='inner')
    return train


def load_test_df():
    test = pd.read_csv('../data/test_set_values.csv')
    return test


def numeric_status_group():
    status_group_numeric = {'functional': 2,
                        'functional needs repair': 1,
                        'non functional': 0}
    return status_group_numeric


def categorize_funder(train):
    '''This function will go through every row in
    the dataframe column funder and if the value
    is any of the top 7 fields, will return those
    values. If the value does not equal any of these
    top 7 fields, it will return other.'''
    if train['funder'] == 'Government Of Tanzania':
        return 'govt'
    elif train['funder'] == 'Danida':
        return 'danida'
    elif train['funder'] == 'Hesawa':
        return 'hesawa'
    elif train['funder'] == 'Rwssp':
        return 'rwssp'
    elif train['funder'] == 'World Bank':
        return 'world_bank'
    elif train['funder'] == 'Kkkt':
        return 'kkkt'
    elif train['funder'] == 'World Vision':
        return 'world_vision'
    else:
        return 'other'

    
def categorize_installer(train):
    '''This function will go through
    every row in the installer column
    and if the value is equal to any of
    the top 7, will return those values.
    If not, will return other.'''
    if train['installer'] == 'DWE':
        return 'dwe'
    elif train['installer'] == 'Government':
        return 'govt'
    elif train['installer'] == 'RWE':
        return 'rwe'
    elif train['installer'] == 'Commu':
        return 'commu'
    elif train['installer'] == 'DANIDA':
        return 'danida'
    elif train['installer'] == 'KKKT':
        return 'kkkt'
    elif train['installer'] == 'Hesawa':
        return 'hesawa'
    else:
        return 'other'

    
def numeric_public(row):
    if row['public_meeting'] == True:
        return 1
    else:
        return 0

    
def categorize_scheme(row):
    '''This function will go through each
    row in the scheme management column
    and if the value is equal to any of the 
    top 7, will return those values. If not,
    will categorize the value as other.'''
    if row['scheme_management'] == 'VWC':
        return 'vwc'
    elif row['scheme_management'] == 'WUG':
        return 'wug'
    elif row['scheme_management'] == 'Water authority':
        return 'water_authority'
    elif row['scheme_management'] == 'WUA':
        return 'wua'
    elif row['scheme_management'] == 'Water Board':
        return 'water_board'
    elif row['scheme_management'] == 'Parastatal':
        return 'parastatal'
    elif row['scheme_management'] == 'Private operator':
        return 'private_operator'
    else:
        return 'other'

    
def permit(row):
    
    if row['permit'] == True:
        return 1
    else:
        return 0

    
def categorize_contruction(row):
    if row['construction_year'] < 1970:
        return '1960s'
    elif row['construction_year'] < 1980:
        return '1970s'
    elif row['construction_year'] < 1990:
        return '1980s'
    elif row['construction_year'] < 2000:
        return '1990s'
    elif row['construction_year'] < 2010:
        return '2000s'
    elif row['construction_year'] < 2020:
        return '2010s'

    
def load_processed_train_df():
    '''This function takes the inital 
    dataframe created with the first 
    function in this file, imputes the 
    missing values in the dataframe, bins
    a few columns into categories, and
    removes columns that provide the same 
    information as another column in the
    dataframe'''
    train = load_train_add_target_df()
    #Creating the status column for numerically transformed status_group
    train['status'] = train.status_group.replace(numeric_status_group)
    #Binning funder field into 7 categories
    train['funder'] = train.apply(lambda x: categorize_funder(x), axis=1)
    #Binning installer field into 7 categories
    train['installer'] = train.apply(lambda x: categorize_installer(x), axis=1)
    #Removing subvillage field
    train = train.drop(columns=['subvillage'], axis=1)
    #Filling 9 percent of na values with False
    train['public_meeting'].fillna(False, limit=300, inplace=True)
    #Filling 91 percent of na values with True
    train['public_meeting'].fillna(True, inplace=True)
    #Binning scheme management field into 7 categories
    train['scheme_management'] = train.apply(lambda x: categorize_scheme(x), axis=1)
    #Removing scheme name field
    train = train.drop(columns=['scheme_name'], axis=1)
    #Filling 31 percent of na values with False
    train['permit'].fillna(False, limit=947, inplace=True)
    #Filling 69 percent of na values with True
    train['permit'].fillna(True, inplace=True)
    #Removing wpt name field
    train = train.drop(columns=['wpt_name'], axis=1)
    #Removing region field
    train = train.drop(columns=['region'], axis=1)
    #Removing recorded by field
    train = train.drop(columns=['recorded_by'], axis=1)
    #Changing permit field to numerical data
    train['permit'] = train.apply(lambda x: permit(x), axis=1)
    #Binning construction year by decade
    train['construction_year'] = train.apply(lambda x: categorize_contruction(x), axis=1)
    #Removing extraction type and extraction type group fields
    train = train.drop(columns=['extraction_type', 'extraction_type_group'], axis=1)
    #Removing management group field
    train = train.drop(columns=['management_group'], axis=1)
    #Removing payment type group field
    train = train.drop(columns=['payment_type'], axis=1)
    #Removing payment type field
    train = train.drop(columns=['water_quality'], axis=1)
    #Removing water quality field
    train = train.drop(columns=['quantity_group'], axis=1)
    #Removing source type field
    train = train.drop(columns=['source_type'], axis=1)
    #Removing waterpoint type group field
    train = train.drop(columns=['waterpoint_type_group'], axis=1)
    
    
   # ohe_features = ['funder', 'installer', 'basin', 
               #'region_code', 'district_code', 'lga', 'public_meeting',
               #'scheme_management', 'permit', 'construction_year', 
               #'extraction_type_class', 'management',
               #'payment', 'quality_group',
              # 'quantity', 'source', 'source_class', 'waterpoint_type']
    #cont_features = ['amount_tsh', 'gps_height', 
                 #'num_private', 'public_meeting',
                  #'population']
    
    #ohe = OneHotEncoder()
    #ss = StandardScaler()
    
    #train_cat = train[ohe_features]
    #train_cont = train[cont_features].astype(float)
    
    #train_ohe = ohe.fit_transform(train_cat)
    #train_scl = pd.DataFrame(ss.fit_transform(train_cont), columns=train[cont_features].columns)
    #columns = ohe.get_feature_names(input_features=train_cat.columns)
    #train_processed = pd.DataFrame(train_ohe.todense(), columns=columns)
    #train_all = pd.concat([train_scl, train_processed], axis=1)
    #train_all['status'] = train['status']
    
    return train

    
    
    
"""
==================================================================================================================
Max's cleaning Functions
==================================================================================================================
"""
def combiner(row, col_1, col_2):
    if row[col_1]!=row[col_2]:
        return f'{row[col_1]}/{row[col_2]}'
    else:
        return row[col_1]
    
def fill_unknown(row, col_1, col_2, unknown):
    if (row[col_1] in unknown) &\
       (row[col_2] in unknown):
        row[col_1] = 'unknown'
        row[col_2] = 'unknown'
        return row
    elif row[col_1] in unknown:
        row[col_1] = row[col_2]
    elif row[col_2] in unknown:
        row[col_2] = row[col_1]
    return row


def combine_managements(df):
    col_1 = 'scheme_management'
    col_2 = 'management'

    df[col_1] = df[col_1].fillna('na')
    df[col_2] = df[col_2].fillna('na')

    df[col_2] = df[col_2].map(lambda x: x.lower())
    df[col_1] = df[col_1].map(lambda x: x.lower())

    df = df.apply(lambda row: fill_unknown(row, col_1, col_2, ['na', 'other', 'none', 'unknown']), axis=1)
    df['scheme_management/management'] = df.apply(lambda row: combiner(row, col_1, col_2), axis=1)
    top = df['scheme_management/management'].value_counts()[df['scheme_management/management'].value_counts()>100]
    
    df['scheme_management/management'] = df['scheme_management/management'].map(lambda x: x if x in top.index else 'binned')
    df.drop([col_1, col_2], axis=1, inplace=True)
    
    return df


def combine_waterpoint(df):
    df['waterpoint_type/group'] = df.apply(lambda row: combiner(row, 'waterpoint_type', 'waterpoint_type_group'), axis=1)

    df['waterpoint_type/group'].value_counts()
    df.drop(['waterpoint_type', 'waterpoint_type_group'], axis=1, inplace=True)
    return df


misspellings = {'dwe&': 'dwe',
 'dwe': 'dwe',
 'dwe/': 'dwe',
 'dwe}': 'dwe',
 'dw#': 'dwe',
 'dw$': 'dwe',
 'dw': 'dwe',
 'dw e': 'dwe',
 'dawe': 'dwe',
 'dweb': 'dwe',
 'government': 'central government',
 'government of tanzania': 'central government',
 'gove': 'central government',
 'tanzanian government': 'central government',
 'governme': 'central government',
 'goverm': 'central government',
 'tanzania government': 'central government',
 'cental government': 'central government',
 'gover': 'central government',
 'centra government': 'central government',
 'go': 'central government',
 'centr': 'central government',
 'central govt': 'central government',
 'cebtral government': 'central government',
 'governmen': 'central government',
 'govern': 'central government',
 'central government': 'central government',
 'olgilai village community': 'community',
 'maseka community': 'community',
 'kitiangare village community': 'community',
 'sekei village community': 'community',
 'igolola community': 'community',
 'comunity': 'community',
 'mtuwasa and community': 'community',
 'village community members': 'community',
 'district community j': 'community',
 'marumbo community': 'community',
 'ngiresi village community': 'community',
 'community': 'community',
 'village community': 'community',
 'commu': 'community',
 'ilwilo community': 'community',
 'communit': 'community',
 'taboma/community': 'community',
 'oldadai village community': 'community',
 'villagers': 'community',
 'kkkt': 'kkkt',
 'kkkt dme': 'kkkt',
 'kkkt-dioces ya pare': 'kkkt',
 'kkkt katiti juu': 'kkkt',
 'kkkt leguruki': 'kkkt',
 'kkkt mareu': 'kkkt',
 'kkkt ndrumangeni': 'kkkt',
 'kk': 'kkkt',
 'kkkt church': 'kkkt',
 'kkkt kilinga': 'kkkt',
 'kkkt canal': 'kkkt',
 'kkt': 'kkkt',
 'lutheran church': 'kkkt',
 'luthe': 'kkkt',
 'haidomu lutheran church': 'kkkt',
 'world vision': 'world vision',
 'world vission': 'world vision',
 'world visiin': 'world vision',
 'world division': 'world vision',
 'world': 'world vision',
 'world nk': 'world vision',
 'district council': 'district council',
 'district counci': 'district council',
 'district  council': 'district council',
 'mbozi district council': 'district council',
 'wb / district council': 'district council',
 'mbulu district council': 'district council',
 'serengeti district concil': 'district council',
 'district water department': 'district council',
 'tabora municipal council': 'district council',
 'hesawa': 'hesawa',
 'esawa': 'hesawa',
 'hesaw': 'hesawa',
 'unknown installer': 'unknown'}

def bin_installer(df):
    """
    input: dataframe
    output: returns a new dataframe with a new column, installer_binned, that has a cleaned installer row
    """
    new_df = df.copy()
    new_df['installer_binned'] = new_df['installer']
    new_df['installer_binned'] = new_df['installer_binned'].fillna('unknown')
    new_df['installer_binned'] = new_df['installer_binned'].map(lambda x: x.lower())
    new_df['installer_binned'] = new_df['installer_binned'].replace(misspellings)
    
    new_df.drop(['installer'], axis=1, inplace=True)
    
    return new_df

def bin_funder(df):
    """
    input: dataframe
    output: returns a new dataframe with a new column, installer_binned, that has a cleaned installer row
    """
    new_df = df.copy()
    new_df['funder_binned'] = new_df['funder']
    new_df['funder_binned'] = new_df['funder_binned'].fillna('unknown')
    new_df['funder_binned'] = new_df['funder_binned'].map(lambda x: x.lower())
    new_df['funder_binned'] = new_df['funder_binned'].replace(misspellings)
    
    new_df.drop(['funder'], axis=1, inplace=True)
    
    return new_df


def combine_installer_funder(df):
    new_df = bin_installer(df)
    new_df = bin_funder(new_df)

    col_1 = 'funder_binned'
    col_2 = 'installer_binned'

    new_df[col_1] = new_df[col_1].fillna('na')
    new_df[col_2] = new_df[col_2].fillna('na')

    new_df[col_2] = new_df[col_2].map(lambda x: x.lower())
    new_df[col_1] = new_df[col_1].map(lambda x: x.lower())


    new_df = new_df.apply(lambda row: fill_unknown(row, col_1, col_2, ['0', 'other', 'unknown']), axis=1)
    new_df['funder/installer'] = new_df.apply(lambda row: combiner(row, col_1, col_2), axis=1)


    top = new_df['funder/installer'].value_counts()[new_df['funder/installer'].value_counts() > 200]
    new_df['funder/installer'] = new_df['funder/installer'].map(lambda x: x if x in top.index else 'binned')
    new_df.drop([col_1, col_2], axis=1, inplace=True)
    
    return new_df

def clean_permit(df):
    df['permit'].fillna('na')
    df['permit'] = df['permit'].map(str)
    
    return df



def combine_extraction(df):
    col_1 = 'extraction_type'
    col_2 = 'extraction_type_group'

    df[col_1] = df[col_1].fillna('na')
    df[col_2] = df[col_2].fillna('na')

    df[col_2] = df[col_2].map(lambda x: x.lower())

    df[col_1] = df[col_1].map(lambda x: x.lower())


    df['extraction_type/group'] = df.apply(lambda row: combiner(row, col_1, col_2), axis=1)


    top = df['extraction_type/group'].value_counts()[:100]
    top

    df['extraction_type/group'] = df['extraction_type/group'].map(lambda x: x if x in top.index else 'binned')
    df.drop([col_1, col_2], axis=1, inplace=True)
    
    # Extraction iteration two

    col_1 = 'extraction_type_class'
    col_2 = 'extraction_type/group'

    df[col_1] = df[col_1].fillna('na')
    df[col_2] = df[col_2].fillna('na')

    df[col_2] = df[col_2].map(lambda x: x.lower())

    df[col_1] = df[col_1].map(lambda x: x.lower())

    df[df[col_2]==df[col_1]]

    df['extraction_type/group/class'] = df.apply(lambda row: combiner(row, col_1, col_2), axis=1)
    df.drop([col_1, col_2], axis=1, inplace=True)
    
    return df
    
def bin_year(year):
    if year<1960:
        return 'unknown'
    elif year>=1960 and year<1970:
        return '1960-1970'
    elif year>=1970 and year<1980:
        return '1970-1980'
    elif year>=1980 and year<1990:
        return '1980-1990'
    elif year>=1990 and year<2000:
        return '1990-2000'
    elif year>=2000 and year<2010:
        return '2000-2010'
    elif year>=2010 and year<2020:
        return '2010-2020'
    else:
        return year
    
def create_decade_columns(decade, target_decade):
    if decade == target_decade:
        return 1
    else:
        return 0
    
    
    
def encode_and_concat_feature_train(X_train_all_features, feature_name):
    """
    Helper function for transforming training data.  It takes in the full X dataframe and
    feature name, makes a one-hot encoder, and returns the encoder as well as the dataframe
    with that feature transformed into multiple columns of 1s and 0s
    """
    # make a one-hot encoder and fit it to the training data
    ohe = OneHotEncoder(categories="auto", handle_unknown="ignore")
    single_feature_df = X_train_all_features[[feature_name]]
    ohe.fit(single_feature_df)
    
    # call helper function that actually encodes the feature and concats it
    X_train_all_features = encode_and_concat_feature(X_train_all_features, feature_name, ohe)
    
    return ohe, X_train_all_features

def encode_and_concat_feature(X, feature_name, ohe):
    """
    Helper function for transforming a feature into multiple columns of 1s and 0s. Used
    in both training and testing steps.  Takes in the full X dataframe, feature name, 
    and encoder, and returns the dataframe with that feature transformed into multiple
    columns of 1s and 0s
    """
    # create new one-hot encoded df based on the feature
    single_feature_df = X[[feature_name]]
    feature_array = ohe.transform(single_feature_df).toarray()
    ohe_df = pd.DataFrame(feature_array, columns=ohe.categories_[0])
    
    # drop the old feature from X and concat the new one-hot encoded df
    X = X.drop(feature_name, axis=1)
    X = pd.concat([X, ohe_df], axis=1)
    
    return X


def create_decades(df):
    df['decade'] = df['construction_year'].map(bin_year)

    non_zeros = df[df['construction_year']>0]
    decades = ['2000-2010',
    '1990-2000',
    '1980-1990',
    '2010-2020',
    '1970-1980',
    '1960-1970']
    for decade in decades:
        non_zeros[decade] = non_zeros['decade'].map(lambda val: create_decade_columns(val, decade))
    
    non_zeros = non_zeros.loc[:,decades]
    
    
    
    zeros = df[df['construction_year']==0]
    zeros_index = zeros.index
    zeros = zeros.reset_index().drop("index", axis=1)

#     zeros_clean = combine_managements(zeros)
#     zeros_clean = combine_waterpoint(zeros_clean)
#     zeros_clean = clean_permit(zeros_clean)
#     zeros_clean = combine_installer_funder(zeros_clean)
#     zeros_clean = combine_extraction(zeros_clean)

    zeros_clean = zeros.loc[:,['waterpoint_type/group', 'scheme_management/management', 'basin', 'region_code', 
                        'funder/installer', 'extraction_type/group/class',  'source', 'source_type', 
                       'management_group', 'permit', 'district_code']]

    funder_installer_ohe = pickle.load(open('funder_installer_ohe.sav', 'rb'))
    extraction_type_group_class_ohe = pickle.load(open('extraction_type_group_class_ohe.sav', 'rb'))
    scheme_management_management_ohe = pickle.load(open('scheme_management_management_ohe.sav', 'rb'))
    management_group_ohe = pickle.load(open('management_group_ohe.sav', 'rb'))
    source_type_ohe = pickle.load(open('source_type_ohe.sav', 'rb'))
    source_ohe = pickle.load(open('source_ohe.sav', 'rb'))
    waterpoint_type_group_ohe = pickle.load(open('waterpoint_type_group_ohe.sav', 'rb'))
    permit_ohe = pickle.load(open('permit_ohe.sav', 'rb'))
    basin_ohe = pickle.load(open('basin_ohe.sav', 'rb'))
    district_code_ohe = pickle.load(open('district_code_ohe.sav', 'rb'))
    region_code_ohe = pickle.load(open('region_code_ohe.sav', 'rb'))

    X_test_all_features = zeros_clean.copy().reset_index().drop("index", axis=1)
    X_test_all_features = encode_and_concat_feature(X_test_all_features, 'funder/installer', funder_installer_ohe)
    X_test_all_features = encode_and_concat_feature(X_test_all_features, "extraction_type/group/class", extraction_type_group_class_ohe)
    X_test_all_features = encode_and_concat_feature(X_test_all_features, "scheme_management/management", scheme_management_management_ohe)
    X_test_all_features = encode_and_concat_feature(X_test_all_features, "management_group", management_group_ohe)
    X_test_all_features = encode_and_concat_feature(X_test_all_features, "source_type", source_type_ohe)
    X_test_all_features = encode_and_concat_feature(X_test_all_features, "source", source_ohe)
    X_test_all_features = encode_and_concat_feature(X_test_all_features, "waterpoint_type/group", waterpoint_type_group_ohe)
    X_test_all_features = encode_and_concat_feature(X_test_all_features, "permit", permit_ohe)
    X_test_all_features = encode_and_concat_feature(X_test_all_features, "basin", basin_ohe)
    X_test_all_features = encode_and_concat_feature(X_test_all_features, "district_code", district_code_ohe)
    X_test_all_features = encode_and_concat_feature(X_test_all_features, "region_code", region_code_ohe)


    model = pickle.load(open('year_predictor.sav', 'rb'))

    predictions_ = model.predict_proba(X_test_all_features)
    prediction_df = pd.DataFrame(predictions_, columns=model.classes_, index=zeros_index)
    

    full_df = pd.concat([non_zeros, prediction_df])
    full_df.sort_index(inplace=True)
    
    new_df = df.merge(full_df, left_index=True, right_index=True)
    new_df.drop('decade', axis=1, inplace=True)
    return new_df

