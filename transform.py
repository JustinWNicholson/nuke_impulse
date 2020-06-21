def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    '''
    n_in: Number of lag observations as input (X). Values may be between [1..len(data)] Optional. Defaults to 1.
    n_out: Number of observations as output (y). Values may be between [0..len(data)-1]. Optional. Defaults to 1.
    dropnan: Boolean whether or not to drop rows with NaN values. Optional. Defaults to True.
    '''


    #define number of variables. data.shape[1] returns number of columns
    n_vars = 1 if type(data) is list else data.shape[1]

    #This will ensure that we use a pandas dataframe
    dff = pd.DataFrame(data)

    #create empty lists of names for columns and names
    cols, names = list(), list()

    #input sequence (t-n, ... t-1)

    for i in range(n_in, 0, -1):
        cols.append(dff.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]


    ## forecast sequence (t, t+1, ... t+n)

    for i in range(0, n_out):
        cols.append(dff.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]


    ## put it all together
    #concatinate by column axis
    agg = pd.concat(cols, axis=1)
    #give appropriate names to columns
    agg.columns = names

    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
