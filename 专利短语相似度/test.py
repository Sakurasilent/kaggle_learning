import numpy as np
import pandas as pd
from pandas import Series,DataFrame
# 定义生成DataFrame的函数
def make_df(cols,index):
    data = {col:[str(col)+str(ind) for ind in index] for col in cols}
    df = DataFrame(data = data,columns = cols ,index = index )
    return df

df1 = make_df(['a','b','c'],[1,2,3])
df5 = make_df(['c','d','e'],[3,4,5])
print(df1)
print(df5)
print(pd.merge(df1,df5, how='left', left_index=True, right_index=True))
