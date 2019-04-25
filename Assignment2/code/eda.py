import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(style="darkgrid")

def missing_values(df):

    '''Plot and handle missing values.'''

    # Plot the columns with missing values
    missing_values = df.isna().sum().sort_values(ascending=False)[df.isna().sum().sort_values() > 0]
    print(missing_values)
    fig, ax = plt.subplots()
    missing_values.plot.bar()

    fig.tight_layout()
    plt.title("Count of missing values per variable")
    plt.ylabel("Number of values missing")
    plt.show()

    # remove variables containing too many NAs? Something else?

    return df


def plot_distributions(df):

    '''Plot distributions of relevant variables.'''

    fig, ax = plt.subplots()


    ## plot all columns, excluding booleans, ids, competitor columns, flags and position ##
    cols_to_plot = [col for col in df.columns if 'bool' not in col and 'id' not in col \
                and 'comp' not in col and 'flag' not in col and 'position' not in col]

    df[cols_to_plot].hist(bins=30)
    plt.show()


    ## plot competitor columns ##
    comp_cols = [col for col in df.columns if '_rate_percent_diff_mag' in col]
    df[comp_cols].hist(bins=30)
    plt.show()


    ## plot distribution over clicked (and not booked), booked, and neither ##
    cnt_booked = df['booking_bool'].sum()
    cnt_clicked = len(df[(df.click_bool == 1) & (df.booking_bool == 0)].index)
    cnt_neither = len(df[(df.click_bool == 0) & (df.booking_bool == 0)].index)

    labels = ['booked', 'clicked', 'neither']
    plt.pie([cnt_booked, cnt_clicked, cnt_neither], labels=labels, autopct='%1.1f%%')
    plt.axis('equal')
    plt.show()


def plot_correlations(df):

    '''Plot correlations between pairs of variables.'''

    ## probability of booking vs price ##

    # bin prices (in 20 bins)
    price_binned = pd.cut(df['price_usd'], bins=20, labels=False)

    # calculate probability
    prob_booked_price = df['booking_bool'].groupby(price_binned).mean()

    plt.scatter(prob_booked_price.index, prob_booked_price)
    plt.xlabel("Price (divided in 20 bins)")
    plt.ylabel("Probability of booking per bin")
    plt.title("Price vs. probability of booking")
    plt.show()


    ## probability of clicking vs price ##

    # calculate probability of clicking
    prob_clicked_price = df['click_bool'].groupby(price_binned).mean()

    plt.scatter(prob_clicked_price.index, prob_clicked_price)
    plt.xlabel("Price (divided in 20 bins)")
    plt.ylabel("Probability of clicking per bin")
    plt.title("Price vs. probability of clicking")
    plt.show()

    ## probability of booking vs review score ##

    prob_booked_review = df['booking_bool'].groupby(df['prop_review_score']).mean()

    plt.scatter(prob_booked_review.index, prob_booked_review)
    plt.xlabel("Review score (rounded to 0.5))")
    plt.ylabel("Probability of booking")
    plt.title("Review score vs. probability of booking")
    plt.show()


    ## probability of booking vs review score ##

    prob_clicked_review = df['click_bool'].groupby(df['prop_review_score']).mean()

    plt.scatter(prob_clicked_review.index, prob_clicked_review)
    plt.xlabel("Review score (rounded to 0.5)")
    plt.ylabel("Probability of clicking")
    plt.title("Review score vs. probability of clicking")
    plt.show()

    ## probability of booking vs stars ##

    prob_booked_stars = df['booking_bool'].groupby(df['prop_starrating']).mean()

    plt.scatter(prob_booked_stars.index, prob_booked_stars)
    plt.xlabel("Star rating of property")
    plt.ylabel("Probability of booking")
    plt.title("Star rating vs. probability of booking")
    plt.show()


    ## probability of booking vs review score ##

    prob_clicked_stars = df['click_bool'].groupby(df['prop_starrating']).mean()

    plt.scatter(prob_clicked_stars.index, prob_clicked_stars)
    plt.xlabel("Star rating of property")
    plt.ylabel("Probability of clicking")
    plt.title("Star rating vs. probability of clicking")
    plt.show()


def plot_competitor_price_impact(df):

    '''Plotting the impact of price of competitors on clicking/booking likelihood'''

    bins = [-np.inf, -30, -20, -10, 0, 10, 20, 30, np.inf]

    x = []
    y = []
    comp = []

    for i in range(1, 9):

        col = "comp" + str(i) + "_rate_percent_diff_mag"

        comp_price_binned = pd.cut(df[col], bins=bins, labels=False)


        comp_price = df['booking_bool'].groupby(comp_price_binned).mean()

        for j, bin in enumerate(bins):

            if bin == np.inf:
                x.append(40)
            elif bin == -np.inf:
                x.append(-40)
            else:
                x.append(bin)

            if float(j) in comp_price.index:
                y.append(comp_price[float(j)])
            else:
                y.append(0)

            comp.append("competitor " + str(i))

    g = sns.lineplot(x=x, y=y, hue=comp)
    plt.xlabel("Price difference with competitor (in %)")
    g.set(xticklabels=[0] + bins)
    plt.ylabel("Probability of booking")
    plt.title("Price difference with competitor vs. probability of booking")
    plt.show()
