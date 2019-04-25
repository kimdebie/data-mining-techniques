def create_competitor_features(df):

    '''Create features for the competitor columns.'''


    # instead of absolute difference with competitor, specify this to difference
    # with magnitude: is Expedia cheaper or more expensive?
    for i in range(1, 9):

        ratecol = "comp" + str(i) + "_rate"
        pricecol = "comp" + str(i) + "_rate_percent_diff"
        magncol = "comp" + str(i) + "_rate_percent_diff_mag"

        df[magncol] = df[ratecol] * df[pricecol]

    return df
