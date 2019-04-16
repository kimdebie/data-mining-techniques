import preprocess
import analyze
import pandas as pd

filename = 'dataset_mood_smartphone.csv'
filename_clean = 'cleaned_normalized.csv'

def main():

############## PRE PROCESS DATA (only once) #############################
    #data = preprocess.load(filename)
    #clean_data = preprocess.clean(data)

############## READ CLEANED DATA ########################################
    data = pd.read_csv(filename_clean)
    data = data.drop(columns=["Unnamed: 0"])
    print(data.head())

############## EXTRACT FEATURES #########################################


if __name__ == '__main__':
    main()
