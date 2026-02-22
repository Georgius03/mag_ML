import pandas as pd
import numpy as np
import yaml


def main():
    # Read configuration
    with open('config/params.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    # Read raw data
    df = pd.read_csv(config['data']['features_path'])

    # Save data to Numpy
    data_x, data_y = np.array(df.drop('charges', axis=1)), np.array(df['charges'])
    np.save(config['data']['dataset_x_path'], data_x)
    np.save(config['data']['dataset_y_path'], data_y)


if __name__ == '__main__':
    main()