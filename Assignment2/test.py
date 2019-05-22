import pandas as pd

df = pd.read_csv('data/upsampled_training_set.csv')
print(df.dtypes)


print(df.isna().sum())

# separate booked, clicked and neither
booked = df[df['booking_bool'] == 1]
clicked = df[(df.click_bool == 1) & (df.booking_bool == 0)]
neither = df[(df.click_bool == 0) & (df.booking_bool == 0)]

print(len(booked.index))
print(len(clicked.index))
print(len(neither.index))
