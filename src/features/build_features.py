from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, StandardScaler
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

    # Generate report
    profile = ProfileReport(df, title="Data Profiling Report", explorative=True)
    profile.to_file(config['reports']['ydata_report_path'])

    # Save data to pandas
    df.to_csv(config['data']['features_path'], index=False)


# Num of childrens encoding
def children_category(children):
    if children == 0:
        return 0
    elif children in [1, 2]:
        return 1
    else:
        return 2


if __name__ == '__main__':
    main()