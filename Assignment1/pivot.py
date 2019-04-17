import pandas as pd

def create_lagged_vars(file, colname, lags=5, hours=24):

    df = pd.read_csv(file)
    df = df.drop("Unnamed: 0", axis=1)

    discols = []

    # we need 'lags' consecutive observations - first remove rows that are not
    # part of such a consecutive series
    for lag in range(1, lags+1):

        temp = pd.DataFrame()

        temp["left"] = df.groupby("id")["time"].diff(periods=lag) == hours*lag
        temp["right"] = df.groupby("id")["time"].diff(periods=-lag) == -hours*lag

        discol = "discard" + str(lag)
        df[discol] = temp.any(axis='columns')

        discols.append(discol)

    df = df[df[discols].all(axis='columns')]
    df = df[[col for col in df.columns if not col in discols]]

    # now we create lags and add them as new columns to the dataset!
    for lag in range(lags):

        newcol = colname + "_lag" + str(lag+1)
        df[newcol] = df.groupby("id")[colname].shift(lag+1)


    # many rows are unusable so we drop them
    df = df.dropna()

    print(df.columns)

    df.to_csv('with_lags.csv')

    return df


df = pd.read_csv('cleaned_normalized.csv')
df = df[["id", "time", "mood", "appCat.builtin"]]
df.to_csv('test.csv')

create_lagged_vars('test.csv', "mood")
create_lagged_vars('with_lags.csv', "appCat.builtin")
