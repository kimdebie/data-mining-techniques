import load
import eda

def main():

    trainingset = '../data/training_set_VU_DM.csv'

    # take the first 1000 lines of the dataset only - use this for testing
    # to make the code less slow! for final plots etc, comment it out
    trainingset = '../data/training_subset.csv'

    # loading in the right file
    data = load.loaddata(trainingset)

    # handling missing values
    data = eda.missing_values(data)

    # create competitor features
    data = eda.create_competitor_features(data)

    # plot distributions
    eda.plot_distributions(data)




if __name__ == '__main__':

    main()
