import numpy as np
import os.path

class Dataset(object):
    """
    An abstract class for a dataset.
    """
    def __init__(self, out_dir, name='dataset', preprocess='standardise'):
        """
        Args:
            out_dir (dir): directory to save and load data to and from.
            name (str): name for dataset.

        Attributes:
            X_train (nparray) 
            X_test (nparray)
            Y_train (nparray)
            Y_test (nparray): Data BEFORE preprocessing.
        """
        self.out_dir    = out_dir
        self.name       = name
        self.preprocess = preprocess

    def _generate_data(self):
        raise NotImplementedError
    
    def _save_data(self, X_train, Y_train, X_test, Y_test, save=True):
        if save:
            np.save(self.out_dir + self.name + "X_train.npy", X_train, 
                    allow_pickle=False)
            np.save(self.out_dir + self.name + "Y_train.npy", Y_train, 
                    allow_pickle=False)
            np.save(self.out_dir + self.name + "X_test.npy",  X_test, 
                    allow_pickle=False)
            np.save(self.out_dir + self.name + "Y_test.npy",  Y_test, 
                    allow_pickle=False)

    def _load_data(self):
        X_train = np.load(self.out_dir + self.name + 'X_train.npy')
        Y_train = np.load(self.out_dir + self.name + 'Y_train.npy')
        X_test  = np.load(self.out_dir + self.name + 'X_test.npy')
        Y_test  = np.load(self.out_dir + self.name + 'Y_test.npy')

        return [X_train, Y_train, X_test, Y_test]

    def load_or_generate_data(self, force_generate = False, save = True):
        """
        Return the data AFTER preprocessing if preprocessing is not None.
        """
        files_exist =   os.path.isfile(self.out_dir + self.name + \
                                "X_train.npy") and\
                        os.path.isfile(self.out_dir + self.name + \
                                "Y_train.npy") and\
                        os.path.isfile(self.out_dir + self.name + \
                        "X_test.npy")

        if (force_generate) or (not files_exist):
            self.X_train, self.Y_train, self.X_test, self.Y_test = \
                    self._generate_data()
            self._save_data(\
                    self.X_train, self.Y_train, self.X_test, self.Y_test, save)
        else:
            self.X_train, self.Y_train, self.X_test, self.Y_test = \
                    self._load_data()

        if self.preprocess == 'standardise':
            return self._standardise(\
                    self.X_train, self.Y_train, self.X_test, self.Y_test)
        else:
            return [self.X_train, self.Y_train, self.X_test, self.Y_test]

    def _standardise(self, X_train, Y_train, X_test, Y_test):
        X_mean = np.mean(self.X_train, axis=0)
        X_std = np.std(self.X_train, axis=0)
        X_std[X_std ==0] = 1 #Constant attributes will be zero through centering

        Y_mean = np.mean(self.Y_train, axis=0)
        Y_std = np.std(self.Y_train, axis=0)
        Y_std[Y_std ==0] = 1 #Constant attributes will be zero through centering

        sX_train    = (X_train-np.tile(X_mean, (X_train.shape[0], 1)))/\
                        np.tile(X_std, (X_train.shape[0], 1))
        sX_test     = (X_test-np.tile(X_mean, (X_test.shape[0], 1)))/\
                        np.tile(X_std, (X_test.shape[0], 1))
        sY_train    = (Y_train-np.tile(Y_mean, (Y_train.shape[0], 1)))/\
                        np.tile(Y_std, (Y_train.shape[0], 1))
        sY_test     = (Y_test-np.tile(Y_mean, (Y_test.shape[0], 1)))/\
                        np.tile(Y_std, (Y_test.shape[0], 1))

        # Save centering values
        self.Y_mean = Y_mean
        self.Y_std = Y_std

        return [sX_train, sY_train, sX_test, sY_test]

    def rmse_original_units(self, model_y_preprocess):
        """
        Args:
            model_y_preprocess (nparray): model output in preprocessed
                units.
        Returns:
            float representing RMSE in original units.
        """
        unpreprocessed = model_y_preprocess*self.Y_std+self.Y_mean
        return np.sqrt(np.average( (unpreprocessed - self.Y_test)**2 ))

class Boston(Dataset):
    # https://archive.ics.uci.edu/ml/machine-learning-databases/housing/
    def __init__(self, out_dir, train_test_split = 0.8):
        super().__init__(out_dir, 'boston')
        self.train_test_split = train_test_split

    def _generate_data(self):
        data = np.loadtxt(open(self.out_dir + '/boston.csv'))
        np.random.shuffle(data) # in-place
        X = data[:,:13]
        Y = data[:,13].reshape((-1, 1))

        n = X.shape[0]
        train_size = int(self.train_test_split*n)

        X_train = X[:train_size,:]
        Y_train = Y[:train_size,:]
        X_test  = X[train_size:,:]
        Y_test  = Y[train_size:,:]

        return [X_train, Y_train, X_test, Y_test]

class Concrete(Dataset):
    # https://archive.ics.uci.edu/ml/datasets/concrete+compressive+strength
    def __init__(self, out_dir, train_test_split = 0.8):
        super().__init__(out_dir, 'concrete')
        self.train_test_split = train_test_split

    def _generate_data(self):
        data = np.loadtxt(open(self.out_dir + '/concrete.csv'))
        np.random.shuffle(data) # in-place
        X = data[:,:8]
        Y = data[:,8].reshape((-1, 1))

        n = X.shape[0]
        train_size = int(self.train_test_split*n)

        X_train = X[:train_size,:]
        Y_train = Y[:train_size,:]
        X_test  = X[train_size:,:]
        Y_test  = Y[train_size:,:]

        return [X_train, Y_train, X_test, Y_test]

class Energy(Dataset):
    # https://archive.ics.uci.edu/ml/datasets/energy+efficiency
    def __init__(self, out_dir, train_test_split = 0.8):
        super().__init__(out_dir, 'energy')
        self.train_test_split = train_test_split

    def _generate_data(self):
        data = np.loadtxt(open(self.out_dir + '/energy.csv'), delimiter=',')
        np.random.shuffle(data) # in-place
        X = data[:,:8]
        Y = data[:,8].reshape((-1, 1))

        n = X.shape[0]
        train_size = int(self.train_test_split*n)

        X_train = X[:train_size,:]
        Y_train = Y[:train_size,:]
        X_test  = X[train_size:,:]
        Y_test  = Y[train_size:,:]

        return [X_train, Y_train, X_test, Y_test]

class Kin8nm(Dataset):
    def __init__(self, out_dir, train_test_split = 0.8):
        super().__init__(out_dir, 'kin8nm')
        self.train_test_split = train_test_split

    def _generate_data(self):
        data = np.loadtxt(open(self.out_dir + '/kin8nm.csv'))
        np.random.shuffle(data) # in-place
        X = data[:,:8]
        Y = data[:,8].reshape((-1, 1))

        n = X.shape[0]
        train_size = int(self.train_test_split*n)

        X_train = X[:train_size,:]
        Y_train = Y[:train_size,:]
        X_test  = X[train_size:,:]
        Y_test  = Y[train_size:,:]

        return [X_train, Y_train, X_test, Y_test]

class Naval(Dataset):
    # http://archive.ics.uci.edu/ml/datasets/condition+based+maintenance+of+naval+propulsion+plants
    def __init__(self, out_dir, train_test_split = 0.8):
        super().__init__(out_dir, 'naval')
        self.train_test_split = train_test_split

    def _generate_data(self):
        data = np.loadtxt(open(self.out_dir + '/naval.csv'))
        np.random.shuffle(data) # in-place
        X = data[:,:16]
        Y = data[:,17].reshape((-1, 1))

        n = X.shape[0]
        train_size = int(self.train_test_split*n)

        X_train = X[:train_size,:]
        Y_train = Y[:train_size,:]
        X_test  = X[train_size:,:]
        Y_test  = Y[train_size:,:]

        return [X_train, Y_train, X_test, Y_test]

class Power(Dataset):
    # https://archive.ics.uci.edu/ml/datasets/combined+cycle+power+plant
    def __init__(self, out_dir, train_test_split = 0.8):
        super().__init__(out_dir, 'power')
        self.train_test_split = train_test_split

    def _generate_data(self):
        data = np.loadtxt(open(self.out_dir + '/power.csv'), delimiter=',')
        np.random.shuffle(data) # in-place
        X = data[:,:4]
        Y = data[:,4].reshape((-1, 1))

        n = X.shape[0]
        train_size = int(self.train_test_split*n)

        X_train = X[:train_size,:]
        Y_train = Y[:train_size,:]
        X_test  = X[train_size:,:]
        Y_test  = Y[train_size:,:]

        return [X_train, Y_train, X_test, Y_test]

class Protein(Dataset):
    # https://archive.ics.uci.edu/ml/datasets/Physicochemical%2BProperties%2Bof%2BProtein%2BTertiary%2BStructure
    def __init__(self, out_dir, train_test_split = 0.8):
        super().__init__(out_dir, 'protein')
        self.train_test_split = train_test_split

    def _generate_data(self):
        data = np.loadtxt(open(self.out_dir + '/protein.csv'), delimiter=',')
        np.random.shuffle(data) # in-place
        X = data[:,1:10]
        Y = data[:,0].reshape((-1, 1))

        n = X.shape[0]
        train_size = int(self.train_test_split*n)

        X_train = X[:train_size,:]
        Y_train = Y[:train_size,:]
        X_test  = X[train_size:,:]
        Y_test  = Y[train_size:,:]

        return [X_train, Y_train, X_test, Y_test]

class Wine(Dataset):
    # https://archive.ics.uci.edu/ml/datasets/Wine+Quality
    def __init__(self, out_dir, train_test_split = 0.8):
        super().__init__(out_dir, 'wine')
        self.train_test_split = train_test_split

    def _generate_data(self):
        data = np.loadtxt(open(self.out_dir + '/wine.csv'), delimiter=';')
        np.random.shuffle(data) # in-place
        X = data[:,:11]
        Y = data[:,11].reshape((-1, 1))

        n = X.shape[0]
        train_size = int(self.train_test_split*n)

        X_train = X[:train_size,:]
        Y_train = Y[:train_size,:]
        X_test  = X[train_size:,:]
        Y_test  = Y[train_size:,:]

        return [X_train, Y_train, X_test, Y_test]

class Yacht(Dataset):
    # http://archive.ics.uci.edu/ml/datasets/yacht+hydrodynamics
    def __init__(self, out_dir, train_test_split = 0.8):
        super().__init__(out_dir, 'yacht')
        self.train_test_split = train_test_split

    def _generate_data(self):
        data = np.loadtxt(open(self.out_dir + '/yacht.csv'))
        np.random.shuffle(data) # in-place
        X = data[:,:6]
        Y = data[:,6].reshape((-1, 1))

        n = X.shape[0]
        train_size = int(self.train_test_split*n)

        X_train = X[:train_size,:]
        Y_train = Y[:train_size,:]
        X_test  = X[train_size:,:]
        Y_test  = Y[train_size:,:]

        return [X_train, Y_train, X_test, Y_test]






