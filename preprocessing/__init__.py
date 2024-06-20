import pandas as pd
import warnings


#data is for 2018 so using 2018 to calculate total credit age
data_year = 2018

def parse_date_columns(data,cols):
    warnings.simplefilter("ignore")
    for col in cols:
        new_col_name = col + '__monthyear'
        data[col] = pd.to_datetime(data[col])
        data[new_col_name] = data[col].apply(lambda x: data_year*100 - (x.month + x.year*100))
    data = data.drop(columns=cols)
    return data

def encode_categorical_with_dict(data,replacement_dict):
    data['emp_length'] =  data['emp_length'].replace(replacement_dict)
    return data

def get_missing_cols(data):
    # returns a dictionary of columns with missing values
    n = data.shape[0]
    missing_cols = {}
    for col in data.columns:
        data_description = data[col].describe()
        count = data_description[0]
        missing_percent = (n - count)/n
        if missing_percent>0:
            missing_cols[col] = missing_percent
    return missing_cols

def drop_missing(data,threshold):
    # Takes input threshold a decimal number and drops columns with missing fraction > threshold
    cols_missing_val = get_missing_cols(data) 
    missing_cols = [col for col in cols_missing_val if cols_missing_val[col] >= threshold]
    to_drop = missing_cols
    data = data.drop(columns=to_drop)
    return data

def drop_extra(data, to_drop):
    data = data.drop(columns=to_drop)
    return data

def normalizer(data,cols,norm):
    for col in cols:
        normalized_col = col+'_normalized'
        data[normalized_col] = data[col]/data[norm]
    data = data.drop(columns = cols)
    return data


def data_preprocessor(data): 
    # Parse date columns
    date_cols_to_parse = ['last_credit_pull_d','last_pymnt_d','issue_d','next_pymnt_d','earliest_cr_line']
    data = parse_date_columns(data,date_cols_to_parse)

    #replacing values in emp_length - to convert text data to numeric (label encoding manually since it is only one column)
    replacement_dict =  {
        '10+ years': 10,
        '2 years': 2,
        '< 1 year': 0.5,
        '3 years':3,
        '1 year':1,
        '5 years':5,
        '4 years':4,
        '7 years':7,
        '8 years':8,
        '6 years':6,
        '9 years':9
    }
    data = encode_categorical_with_dict(data,replacement_dict)

    

    # Dropping extra columns
    identifiers = ['member_id','id','emp_title']
    linearly_related_duplicates = []
    # ['zip_code','loan_amnt','funded_amnt_inv','total_pymnt','desc', 'policy_code', 'title','out_prncp','acc_now_delinq','tot_coll_amt']
    other_removables = ['policy_code','loan_amnt','funded_amnt_inv','total_pymnt','out_prncp','sub_grade']
    to_drop = identifiers + linearly_related_duplicates + other_removables
    data = drop_extra(data,to_drop)
    

    # Dropping columns with large missing chunks (>25%)
    threshold = 0.8
    data = drop_missing(data,threshold)
    


    # Normalize 'tot_cur_bal', 'total_rev_hi_lim', 'total_rec_prncp', 'total_rec_int', 'annual_inc', 'revol_bal'  using loan amount
    to_normalize = ['tot_cur_bal', 'total_rev_hi_lim', 'total_rec_prncp', 'total_rec_int','annual_inc', 'revol_bal']
    normalize_with = 'funded_amnt'
    data = normalizer(data,to_normalize,normalize_with)
    
    # Normalize 'tot_coll_amt','total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_amnt' fees with installment amount
   
   
    to_normalize = ['tot_coll_amt','total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_amnt']
    normalize_with = 'installment'
    data = normalizer(data,to_normalize,normalize_with)
    

    return data