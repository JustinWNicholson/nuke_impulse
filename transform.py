
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


def create_timesteps(data, n_features, n_years=3, n_train_rows = 2000, dropnan=True):

    values = data.values
    df.shape
    reframed = series_to_supervised(df, n_years, 1)
    print(reframed.shape)

#####THIS SECTION CREATES TRAIN AND TEST DATA - NEED TO UPDATE
# split into train and test sets
    values = reframed.values
    train = values[:n_train_rows, :]
    test = values[n_train_rows:, :]

    print(train.shape)
    print(test.shape)
# split into input and outputs
    n_obs = n_years * n_features
    train_X, train_y = train[:, :n_obs], train[:, -n_features]
    test_X, test_y = test[:, :n_obs], test[:, -n_features]
    print(train_X.shape, len(train_X), train_y.shape)

# reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_years, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_years, n_features))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    return (train_X, train_y, test_X, test_y)
