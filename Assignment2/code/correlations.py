import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def show_correlations(df):
    correlations = calculate_pvalues(df)

    correlations = correlations.astype(float)
    # correlations = correlations.drop(['srch_id', 'site_id', 'prop_id', 'srch_destination_id', \
    #                     'srch_room_count', 'relevance', 'visitor_location_country_id', 'position', \
    #                     'promotion_flag'], axis=1)
    correlations = correlations.drop(['srch_id', 'site_id', 'prop_id', 'srch_destination_id', \
                        'srch_room_count', 'relevance', 'position', 'click_bool', 'booking_bool'], axis=0)

    plt.figure()
    sns.heatmap(correlations[['click_bool', 'booking_bool', 'relevance']], cmap='coolwarm', center=0.2)
    plt.show()

def calculate_pvalues(df):
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            p_value = round(pearsonr(df[r], df[c])[1], 4)
            pvalues[r][c] = p_value

            # for reading purposes, significant results are marked with an asterix
            # if p_value <= 0.05:
            #     pvalues[r][c] = str(p_value) + '*'
            #
            # else:
            #     pvalues[r][c] = p_value

    return pvalues
