import preprocess
import analyze
import pandas as pd

filename = 'dataset_mood_smartphone.csv'
filename_clean = 'cleaned_normalized.csv'

def main():

    data = preprocess.load(filename)

    # print(data['id'])
    clean_data = preprocess.clean(data)

    clean_data = pd.read_csv(filename_clean)

    # print(clean_data['time'])

    # print(clean_data.loc[clean_data['time'] == 840])
    # print(clean_data.loc[clean_data['time'] == 864])

if __name__ == '__main__':
    main()
