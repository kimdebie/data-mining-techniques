import preprocess
import analyze

filename = 'dataset_mood_smartphone.csv'

def main():

    data = preprocess.load(filename)
    clean_data = preprocess.clean(data)


if __name__ == '__main__':
    main()
