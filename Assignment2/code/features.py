import numpy as np

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

def other_features(df):

    '''Create more features.'''

    # historical price of property minus current price per night
    df['ump'] = np.exp(df['prop_log_historical_price']) - df['price_usd']

    # total number of passengers
    df['tot_passengers'] = df['srch_adults_count'] + df['srch_children_count']

    # price per person
    df['price_pp'] = df['price_usd'] * df['srch_room_count'] / df['tot_passengers']

    return df

def relevance(df):
    if df['booking_bool'] == True:
        return 5
    elif df['click_bool'] == True:
        return 1
    else:
        return 0

def relevance_score(df):

    '''Add relevance score based on clicking and booking.'''

    df['relevance'] = 0 #df.apply(relevance, axis=1)
    df[df['booking_bool'] == 1].relevance = 5
    df[(df.click_bool == 0) & (df.booking_bool == 0)].relevance = 1


    return df
