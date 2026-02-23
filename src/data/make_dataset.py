from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from ydata_profiling import ProfileReport
import pandas as pd
import numpy as np
import yaml


def main():
    # Read configuration
    with open('config/params.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    
    # Read raw data
    df = pd.read_csv(config['data']['raw_dataset_csv'])

    # Text cols encoding
    for col in ['sex', 'smoker']:
        lb = LabelBinarizer()
        df[col] = lb.fit_transform(df[col])

    for col in ['region']:
        ohe = OneHotEncoder()
        encoded = ohe.fit_transform(df[[col]]).toarray()
        df = df.join(pd.DataFrame(encoded, columns=[f"{col}_{c}" for c in ohe.categories_[0]], index=df.index))
        df.drop(col, axis=1, inplace=True)

    # Numeric cols scaling
    for col in ['age', 'bmi']:
        scaler = StandardScaler()
        df[col] = scaler.fit_transform(df[[col]])

    df['charges'] = np.log1p(df['charges'])


    df['children'] = df['children'].map(children_category)
    for col in ['children']:
        ohe = OneHotEncoder()
        encoded = ohe.fit_transform(df[[col]]).toarray()
        df = df.join(pd.DataFrame(encoded, columns=[f"{col}_{c}" for c in ohe.categories_[0]], index=df.index))
        df.drop(col, axis=1, inplace=True)

    # Save data
    df.to_csv(config['data']['processed_dataset_csv'], index=False)
    
    # Save data to Numpy
    data_x, data_y = np.array(df.drop('charges', axis=1)), np.array(df['charges'])
    np.save(config['data']['dataset_x_path_np'], data_x)
    np.save(config['data']['dataset_y_path_np'], data_y)
    

    train_df, test_df = train_test_split(
        df,
        test_size=config['data']['test_size'],
        random_state=config['base']['random_state']
    )

    train_df.to_csv(config['data']['train_path'], index=False)
    test_df.to_csv(config['data']['test_path'], index=False)

    return 0

# Num of childrens encoding
def children_category(children):
    if children == 0:
        return 0
    elif children in (1, 2):
        return 1
    return 2

if __name__ == '__main__':
    main()