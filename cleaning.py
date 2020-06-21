#import necessary data
df = pd.read_csv('~/lstm_test/input/nu_public_rep.csv', sep=';',
                 parse_dates={'dt' : ['Date', 'Time']}, infer_datetime_format=True,
                 low_memory=False, na_values=['nan','?'], index_col='dt')

#fill nan with the means of the columns to handle missing databases
droping_list_all=[]
for j in range(0,7):
    if not df.iloc[:, j].notnull().all():
        droping_list_all.append(j)
        #print(df.iloc[:,j].unique())
droping_list_all

#This small loop will replace any missing vlaues with means of columns (study this!!)
for j in range(0,7):
        df.iloc[:,j]=df.iloc[:,j].fillna(df.iloc[:,j].mean())

#Get sum of missing calues - sanity check (should be zero)
df.isnull().sum()
