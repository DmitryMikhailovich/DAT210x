#%%
import pandas as pd


# TODO: Load up the table, and extract the dataset
# out of it. If you're having issues with this, look
# carefully at the sample code provided in the reading
#
df = pd.read_html('http://www.espn.com/nhl/statistics/player/_/stat/points/sort/points/year/2015/seasontype/2', 
                  header=1)[0]
df = df[df['RK'] != 'RK']
#%%
# TODO: Rename the columns so that they match the
# column definitions provided to you on the website
#
df.rename(columns={'G.1':'PPG', 'A.1':'PPA', 'G.2':'SHG','A.2':'SHA'}, 
          inplace=True)

#%%
# TODO: Get rid of any row that has at least 4 NANs in it
#
df.dropna(thresh=4, inplace=True)

#%%
# TODO: At this point, look through your dataset by printing
# it. There probably still are some erroneous rows in there.
# What indexing command(s) can you use to select all rows
# EXCEPT those rows?
#
print(df)
print(df[df.notnull().all(axis=1)])

#%%
# TODO: Get rid of the 'RK' column
#
del df['RK']

#%%
# TODO: Ensure there are no holes in your index by resetting
# it. By the way, don't store the original index
#
df.reset_index(inplace=True)


#%%
# TODO: Check the data type of all columns, and ensure those
# that should be numeric are numeric

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='ignore')
print(df.dtypes)

#%%
# TODO: Your dataframe is now ready! Use the appropriate 
# commands to answer the questions on the course lab page.

