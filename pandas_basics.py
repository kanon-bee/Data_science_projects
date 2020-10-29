import pandas as pd
import numpy as np

# Create a series by passing a list of values.
a = pd.Series([1,3,5,np.nan, 6,8])

# Create a DataFrame by passing a numpy array with a datetime index and labeled column
dates = pd.date_range('20130101', periods= 6)


df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('Abcd'))
# Creating a DataFrame by passing a dict of objects that can be converted to series-like.


df2 = pd.DataFrame({'A': 1., 'B': pd.Timestamp('20130102'), 'C': pd.Series(1, index = list(range(4)), dtype = float),
                    'D': np.array([3] * 4, dtype = 'int32'), 'E': pd.Categorical(['test', 'train', 'test', 'train']),
                    'F': 'Foo'})

# Boolean Indexing
# print(df2[df2.A > 0])
# print(df[df.A > 0])
# print(df[df >0])


# df2 = df.copy()
# df2['E'] = ['One', 'Two', 'One', 'Four', 'Four','Three']
# print(df2)
# print(df2[df2['E'].isin(['One', 'Four'])])

s = pd.DataFrame({'A': [1,2,3,4,5,6],}, index= pd.date_range('20120101', periods= 6))
s.insert(1, 'B', [7,8,9,10,11,12], False)

df.iat[3,1] = 0
df.loc[:, 'd']= np.array([5] * len(df))
# print(df)

df2.loc[:, 'F'] = np.array(range(0,6), dtype=float)
df2 = df.copy()
df2[df2>0] = -df2
# print(df.columns)


df3 = df.reindex(index = dates[0:4], columns = list(df.columns) + ['e'])
df3.loc[dates[0:1], 'e'] = 1

# To Drop Any Rows that are nan
c = df3.dropna(how='any')

# To Fill Any Rows That are nan
e = df3.fillna(value=5)



# Get boolean mask where values are nan
# print(pd.isna(df3))

# To get a average value of a Column
# print(df3.mean())

# To get a average value of a Row
# print(df3.mean(1))

# To Substract a Dataframe from another Dataframe by row or columns
q = pd.DataFrame({'A': [1,2,3,5], 'B': [5,4,7,4], 'C': [5,9,0,2]}, index= ['A1', 'A2', 'A3', 'A4'])
q1 = pd.Series([1,2,3], index=['A', 'B', 'C'])
q2 = q.sub(q1,axis = 1)
print(q)
# print("\n")
# print(q2)

# To apply functions to a Dataframe
df.apply(np.cumsum)
df.apply(lambda x: x.max() - x.min())


# To get the Unique valiue of a dataframe
b = pd.DataFrame([1,2,1,3,5,3,2,8], index= range(0,8))
f = df3.iloc[:, 1]
# print(f)
# print(f.value_counts())


# String method to do String operations
g = pd.Series([1,2,3,'A', 'B', 'C', np.nan])
# print(g.str.lower())


# To break a dataframe into pieces and add them using Concat function
p = pd.DataFrame(np.random.randn(7, 4))
p1 = [p[:3], p[3:7], p[7:]]
p2 = pd.concat(p1)
# print(p2)


# Merging two dataframe into one
r = pd.DataFrame({'key': ['foo', 'bar'], 'ival': [1,2]})
r1 = pd.DataFrame({'key': ['foo', 'bar'], 'ival' : [3,4]})
r2 = pd.merge(r,r1, on='key')
# print(r2)


# Appending two dataframe into one
d = pd.DataFrame(np.random.randn(8,4), columns=['A', 'B', 'C', 'D'])
d1 = d.iloc[3]
d2 = d.append(d1, ignore_index=True)

# print(d, '\n')
# print(d1, '\n')
# print(d2)

# Another example of Appending
h= pd.DataFrame({'A': [1,2,3,4], 'B': [5,6,7,8], 'C': [9,10,11,12]}, index=[0,1,2,3])
h1 = h.iloc[1]
h3 = h.append(h1)

# print(h, '\n')
# print(h1, '\n')
# print(h3)


i = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'bar'],
                  'B': ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
                  'C': np.random.randn(8),
                  'D': np.random.randn(8)})
i1 = i.groupby('B').sum()
i2 = i.groupby(['A', 'B']).sum()

# print(i, '\n')
# print(i2)


# Reshaping Dataframe
tuples = list(zip(*[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'quz'],
                    'one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']))

# index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
