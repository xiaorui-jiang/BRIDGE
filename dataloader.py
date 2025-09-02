from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch
import copy
import random
from sklearn import preprocessing
from collections import Counter
min_max_scaler = preprocessing.MinMaxScaler()
from numpy.random import dirichlet
from collections import defaultdict
import numpy as np
import random
from collections import defaultdict

def get_boolean_matrix(n, view_num, Complete_index, split_B):
    # Initialize the matrix with False values
    boolean_matrix = np.zeros((n, view_num), dtype=bool)

    # Mark the elements present in each view as True
    for view in range(view_num):
        # Set True for indices in Complete_index for the current view
        for idx in Complete_index:
            boolean_matrix[idx, view] = True
        # Set True for indices in split_B for the current view
        for idx in split_B[view]:
            boolean_matrix[idx, view] = True

    return boolean_matrix

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def add_pure_random_noniid_noise(data, noise_level=0.1, noise_ratio=0.1):

    noisy_data = []

    for v in range(len(data)):
        view_data = data[v].copy()


        n, dim = view_data.shape

        global_mean = np.mean(view_data, axis=0)
        global_std = np.std(view_data, axis=0) + 1e-6


        mask = np.random.rand(n, dim) < noise_ratio


        noise_types = np.random.choice(["normal", "uniform", "poisson", "laplace", "exponential"], size=(n, dim))

        noise = np.zeros_like(view_data)


        for i in range(n):
            for j in range(dim):
                if mask[i, j]:
                    noise_type = noise_types[i, j]
                    mean = np.random.uniform(-0.1, 0.1) * global_mean[j]
                    std = np.random.uniform(0.5, 2) * global_std[j] * noise_level

                    if noise_type == "normal":
                        noise[i, j] = np.random.normal(loc=mean, scale=std)
                    elif noise_type == "uniform":
                        noise[i, j] = np.random.uniform(low=-std, high=std)
                    elif noise_type == "poisson":
                        noise[i, j] = float(np.random.poisson(lam=np.abs(mean) + 1))
                    elif noise_type == "laplace":
                        noise[i, j] = np.random.laplace(loc=mean, scale=std)
                    elif noise_type == "exponential":
                        noise[i, j] = np.random.exponential(scale=std)

        view_data[mask] += noise[mask]

        noisy_data.append(np.float32(view_data))

    return noisy_data

def Form_Incomplete_Data(seed, missrate, X, Y):
    setup_seed(seed)
    size = len(Y[0])
    view_num = len(X)

    # only shuffle one time, and use seed
    t = np.linspace(0, size - 1, size, dtype=int)

    # Please comment out `random.shuffle(t)` if you are using BRIDGE to build your own DIMVC method.
    random.shuffle(t)

    Xtmp = []
    Ytmp = []
    for i in range(view_num):
        Xtmp.append(copy.deepcopy(X[i]))
        Ytmp.append(copy.deepcopy(Y[i]))

    for v in range(view_num):
        for i in range(size):
            Xtmp[v][i] = X[v][t[i]]
            Ytmp[v][i] = Y[v][t[i]]

    X = Xtmp
    Y = Ytmp
    X_i = copy.deepcopy(Xtmp)
    Y_i = copy.deepcopy(Ytmp)

    # complete data index
    index0 = np.linspace(0, round((1 - missrate) * size - 1), num=round((1 - missrate) * size), dtype=int)

    missindex = np.ones((round(missrate * size), view_num))

    # miss data index (do not shuffle)
    index = [[] for _ in range(view_num)]
    miss_begain = round((1 - missrate) * size)

    total_missing_per_view = round(np.ceil(missrate * size / view_num))

    for view in range(view_num):
        start_row = view * total_missing_per_view
        end_row = (view + 1) * total_missing_per_view
        missindex[start_row:end_row, view] = 0

    # incomplete data index
    missindex = missindex.astype(bool)
    for i in range(missindex.shape[0]):
        for j in range(view_num):
                    if missindex[i, j] == False:
                        index[j].append(round(miss_begain + i))

    min_len = copy.deepcopy(min(len(sublist) for sublist in index))
    max_len = copy.deepcopy(max(len(sublist) for sublist in index))
    for i in range(view_num):
        if len(index[i]) == min_len:
            index[i] = index[i] + random.choices(index[i], k=max_len - min_len)

    # imvc data
    for j in range(view_num):
        X_i[j] = X_i[j][index[j]]
        Y_i[j] = Y_i[j][index[j]].ravel()

    # to form complete and incomplete views' data
    for j in range(view_num):
        index[j] = list(index0) + index[j]
        X[j] = X[j][index[j]]
        Y[j] = Y[j][index[j]].ravel()
    print("----------------generate incomplete multi-view data-----------------------")
    return X, Y, X_i, Y_i, index


def Form_Incomplete_Data_NonIID(seed, missrate, X, Y, alpha=1):
    view_num = len(X)
    for i in range(len(Y)):
        Y[i] = Y[i].reshape(len(Y[i])).astype(np.int32)
    n_classes = max(Y[0]) + 1
    alpha = [alpha] * n_classes

    A = list(range(len(Y[0])))

    n_samples = round(len(Y[0]) * missrate)


    label_distribution = dirichlet(alpha)


    label_to_elements = defaultdict(list)
    for idx, label in zip(A, Y[0]):
        label_to_elements[label].append(idx)

    actual_label_counts = {label: int(len(elements) * 0.95) for label, elements in label_to_elements.items()}

    target_label_counts = np.round(label_distribution * n_samples).astype(int)



    adjusted_label_counts = {}
    for label, target_count in enumerate(target_label_counts):
        adjusted_label_counts[label] = min(target_count, actual_label_counts.get(label, 0))

    min_samples_per_class = 100
    adjusted_total = sum(adjusted_label_counts.values())

    if adjusted_total < n_samples:

        diff = n_samples - adjusted_total

        while diff > 0:
            for label in range(n_classes):
                available_space = actual_label_counts[label] - adjusted_label_counts[label]

                if adjusted_label_counts[label] < min_samples_per_class:
                    add_count = min(diff, int(random.uniform(0.5, 2) * (min_samples_per_class - adjusted_label_counts[label])), available_space)
                    adjusted_label_counts[label] += add_count
                    diff -= add_count
            min_samples_per_class += 10



    B = []
    for label, count in adjusted_label_counts.items():
        if count > 0:
            B.extend(np.random.choice(label_to_elements[label], size=count, replace=False))


    label_to_elements = defaultdict(list)
    for index in B:
        label = Y[0][index]
        label_to_elements[label].append(index)

    split_B = [[] for _ in range(view_num)]
    for label, elements in label_to_elements.items():
        np.random.shuffle(elements)

        parts = np.array_split(elements, view_num)
        for i in range(view_num):
            split_B[i].extend(parts[i])


    min_len = min(len(part) for part in split_B)

    extra_elements = []
    for i in range(view_num):
        if len(split_B[i]) > min_len:
            extra_elements.extend(split_B[i][min_len:])
            split_B[i] = split_B[i][:min_len]

    extra_per_view = len(extra_elements) // view_num  # 每个部分额外分配的元素数

    index = 0
    for i in range(view_num):
        split_B[i].extend(extra_elements[index:index + extra_per_view])
        index += extra_per_view
    Complete_index = list(set(A) - set(B))


    Full_X = [np.concatenate((X[i][Complete_index], X[i][split_B[i]]), axis=0) for i in range(view_num)]
    Full_Y = [np.concatenate((Y[i][Complete_index], Y[i][split_B[i]]), axis=0) for i in range(view_num)]
    X_i = [X[i][split_B[i]] for i in range(view_num)]
    Y_i = [Y[i][split_B[i]] for i in range(view_num)]


    print("----------------generate incomplete multi-view data with different distributions-----------------------")
    return Full_X, Full_Y, X_i, Y_i, index


class SubsetDataset(Dataset):
    def __init__(self, X, Y, view, miss_rate, size, isFull):
        self.X = X
        self.Y = Y
        self.isFull = isFull
        self.view = view
        if isFull:
            for v in range(self.view):
                self.X[v] = self.X[v][0:round((1 - miss_rate) * size)]
            self.Y = self.Y[0][0:round((1 - miss_rate) * size)]

    def __len__(self):
        if self.isFull:
            return len(self.Y)
        else:
            return len(self.Y[0])


    def __getitem__(self, idx):
        if self.isFull:
            return [torch.from_numpy(self.X[v][idx]) for v in range(self.view)], torch.from_numpy(np.array(self.Y[idx]))
        else:
            return [torch.from_numpy(self.X[v][idx]) for v in range(self.view)], [torch.from_numpy(np.array(self.Y[v][idx])) for v in range(self.view)]


class BDGP(Dataset):
    def __init__(self, path, miss_rate, seed, flag):
        data1 = scipy.io.loadmat(path + 'BDGP.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path + 'BDGP.mat')['X2'].astype(np.float32)
        labels = scipy.io.loadmat(path + 'BDGP.mat')['Y'].transpose()
        self.view = 2
        self.X1 = min_max_scaler.fit_transform(data1)
        self.X2 = min_max_scaler.fit_transform(data2)
        if flag:
            self.X, self.Y, self.X_i, self.Y_i, self.index = Form_Incomplete_Data(seed, missrate=miss_rate, X=[self.X1, self.X2], Y=[labels for _ in range(self.view)])
            self.X_i = add_pure_random_noniid_noise(self.X_i)
            # or
            # self.X, self.Y, self.X_i, self.Y_i, self.index = Form_Incomplete_Data_NonIID(seed, missrate=miss_rate, X=[self.X1, self.X2], Y=[labels for _ in range(self.view)])

        else:
            self.X, self.Y, self.X_i, self.Y_i, self.index = Form_Incomplete_Data(seed, missrate=miss_rate, X=[self.X1, self.X2], Y=[labels for _ in range(self.view)])


    def __getitem__(self, idx):
        return [torch.from_numpy(self.X[v][idx]) for v in range(self.view)]

    def __len__(self):
        return len(self.X[0])

class Cifar10(Dataset):
    def __init__(self, path, miss_rate, seed,flag):
        self.X1 = scipy.io.loadmat(path + 'cifar10.mat')['data'][0][0].astype(np.float32).T
        self.X2 = scipy.io.loadmat(path + 'cifar10.mat')['data'][1][0].astype(np.float32).T
        self.X3 = scipy.io.loadmat(path + 'cifar10.mat')['data'][2][0].astype(np.float32).T
        self.X1 = min_max_scaler.fit_transform(self.X1)
        self.X2 = min_max_scaler.fit_transform(self.X2)
        self.X3 = min_max_scaler.fit_transform(self.X3)
        self.view = 3
        self.y = scipy.io.loadmat(path + 'cifar10.mat')['truelabel'][0][0].transpose().astype(np.int32).reshape(50000, ) - 1
        if flag:
            self.X, self.Y, self.X_i, self.Y_i, self.index = Form_Incomplete_Data(seed, missrate=miss_rate, X=[self.X1, self.X2, self.X3], Y=[self.y, self.y, self.y])
            self.X_i = add_pure_random_noniid_noise(self.X_i)
            # or
            # self.X, self.Y, self.X_i, self.Y_i, self.index = Form_Incomplete_Data_NonIID(seed, missrate=miss_rate, X=[self.X1, self.X2, self.X3], Y=[self.y, self.y, self.y])
        else:
            self.X, self.Y, self.X_i, self.Y_i, self.index = Form_Incomplete_Data(seed, missrate=miss_rate, X=[self.X1, self.X2, self.X3], Y=[self.y, self.y, self.y])

    def __len__(self):
        return len(self.X[0])

    def __getitem__(self, idx):
        return [torch.from_numpy(self.X[v][idx]) for v in range(self.view)]

class Cifar100(Dataset):
    def __init__(self, path, miss_rate, seed, flag):
        self.X1 = scipy.io.loadmat(path + 'cifar100.mat')['data'][0][0].astype(np.float32).T
        self.X2 = scipy.io.loadmat(path + 'cifar100.mat')['data'][1][0].astype(np.float32).T
        self.X3 = scipy.io.loadmat(path + 'cifar100.mat')['data'][2][0].astype(np.float32).T
        self.X1 = min_max_scaler.fit_transform(self.X1)
        self.X2 = min_max_scaler.fit_transform(self.X2)
        self.X3 = min_max_scaler.fit_transform(self.X3)
        self.view = 3
        self.y = scipy.io.loadmat(path + 'cifar100.mat')['truelabel'][0][0].transpose().astype(np.int32).reshape(50000, ) - 1
        if flag:
            self.X, self.Y, self.X_i, self.Y_i, self.index = Form_Incomplete_Data(seed, missrate=miss_rate, X=[self.X1, self.X2, self.X3], Y=[self.y, self.y, self.y])
            self.X_i = add_pure_random_noniid_noise(self.X_i)
            # or
            # self.X, self.Y, self.X_i, self.Y_i, self.index = Form_Incomplete_Data_NonIID(seed, missrate=miss_rate, X=[self.X1, self.X2, self.X3], Y=[self.y, self.y, self.y])
        else:
            self.X, self.Y, self.X_i, self.Y_i, self.index = Form_Incomplete_Data(seed, missrate=miss_rate, X=[self.X1, self.X2, self.X3], Y=[self.y, self.y, self.y])

    def __len__(self):
        return len(self.X[0])

    def __getitem__(self, idx):
        return [torch.from_numpy(self.X[v][idx]) for v in range(self.view)]


class MNIST_USPS(Dataset):
    def __init__(self, path, miss_rate, seed, flag):
        self.Y = scipy.io.loadmat(path + 'MNIST_USPS.mat')['Y'].astype(np.int32).reshape(5000, )
        self.X1 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X1'].astype(np.float32).reshape(5000, 784)
        self.X2 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X2'].astype(np.float32).reshape(5000, 784)
        self.X1 = min_max_scaler.fit_transform(self.X1)
        self.X2 = min_max_scaler.fit_transform(self.X2)
        if flag:
            self.X, self.Y, self.X_i, self.Y_i, self.index = Form_Incomplete_Data(seed, missrate=miss_rate, X=[self.X1, self.X2], Y=[self.Y, self.Y])
            self.X_i = add_pure_random_noniid_noise(self.X_i)
            # or
            # self.X, self.Y, self.X_i, self.Y_i, self.index = Form_Incomplete_Data_NonIID(seed, missrate=miss_rate, X=[self.X1, self.X2], Y=[self.Y, self.Y])
        else:
            self.X, self.Y, self.X_i, self.Y_i, self.index = Form_Incomplete_Data(seed, missrate=miss_rate, X=[self.X1, self.X2], Y=[self.Y, self.Y])


    def __len__(self):
        return len(self.X[0])

    def __getitem__(self, idx):
        return [torch.from_numpy(self.X[v][idx]) for v in range(self.view)]

class BBCSport(Dataset):
    def __init__(self, path, miss_rate, seed, flag):
        data = scipy.io.loadmat(path + 'bbcsport.mat')
        data1 = data['X1'].astype(np.float32).T
        data2 = data['X2'].astype(np.float32).T
        labels = data['truth'].reshape(544, ) - 1
        self.view = 2
        self.X1 = data1
        self.X2 = data2
        self.Y = labels
        if flag:
            self.X, self.Y, self.X_i, self.Y_i, self.index = Form_Incomplete_Data(seed, missrate=miss_rate, X=[self.X1, self.X2], Y=[self.Y, self.Y])
            self.X_i = add_pure_random_noniid_noise(self.X_i)

            # self.X, self.Y, self.X_i, self.Y_i, self.index = Form_Incomplete_Data_NonIID(seed, missrate=miss_rate, X=[self.X1, self.X2], Y=[self.Y, self.Y])
        else:
            self.X, self.Y, self.X_i, self.Y_i, self.index = Form_Incomplete_Data(seed, missrate=miss_rate, X=[self.X1, self.X2], Y=[self.Y, self.Y])

    def __len__(self):
        return len(self.X[0])

    def __getitem__(self, idx):
        return [torch.from_numpy(self.X[v][idx]) for v in range(self.view)]


class Hdigit():
    def __init__(self, path, miss_rate, seed, flag):
        self.view = 2
        data = scipy.io.loadmat(path + 'Hdigit.mat')
        min_max_scaler = preprocessing.MinMaxScaler()
        self.Y = data['truelabel'][0][0].astype(np.int32).reshape(10000, ) - 1  # NOTE: -1 is vital
        self.X1 = data['data'][0][0].T.astype(np.float32)
        self.X2 = data['data'][0][1].T.astype(np.float32)
        self.X1 = min_max_scaler.fit_transform(self.X1)
        self.X2 = min_max_scaler.fit_transform(self.X2)
        if flag:
            self.X, self.Y, self.X_i, self.Y_i, self.index = Form_Incomplete_Data(seed, missrate=miss_rate, X=[self.X1, self.X2], Y=[self.Y, self.Y])
            self.X_i = add_pure_random_noniid_noise(self.X_i)

            # self.X, self.Y, self.X_i, self.Y_i, self.index = Form_Incomplete_Data_NonIID(seed, missrate=miss_rate, X=[self.X1, self.X2], Y=[self.Y, self.Y])
        else:
            self.X, self.Y, self.X_i, self.Y_i, self.index = Form_Incomplete_Data(seed, missrate=miss_rate, X=[self.X1, self.X2], Y=[self.Y, self.Y])


    def __len__(self):
        return len(self.X[0])

    def __getitem__(self, idx):
        return [torch.from_numpy(self.X[v][idx]) for v in range(self.view)]

class REU(Dataset):
    def __init__(self, path, miss_rate, seed, flag):
        data = scipy.io.loadmat(path + "REU.mat")
        self.x1 = data['X1'].astype(np.float32)
        self.x2 = data['X2'].astype(np.float32)
        self.x3 = data['X3'].astype(np.float32)
        self.x4 = data['X4'].astype(np.float32)
        self.x5 = data['X5'].astype(np.float32)
        self.Y = np.copy(data['Y'][0]).astype(np.int32).reshape(1200,) - 1
        min_max_scaler = preprocessing.MinMaxScaler()
        self.X1 = min_max_scaler.fit_transform(self.x1)
        self.X2 = min_max_scaler.fit_transform(self.x2)
        self.X3 = min_max_scaler.fit_transform(self.x3)
        self.X4 = min_max_scaler.fit_transform(self.x4)
        self.X5 = min_max_scaler.fit_transform(self.x5)
        if flag:
            self.X, self.Y, self.X_i, self.Y_i, self.index = Form_Incomplete_Data(seed, missrate=miss_rate, X=[self.X1, self.X2], Y=[self.Y, self.Y])
            self.X_i = add_pure_random_noniid_noise(self.X_i)
            # self.X, self.Y, self.X_i, self.Y_i, self.index = Form_Incomplete_Data_NonIID(seed, missrate=miss_rate, X=[self.X1, self.X2, self.X3, self.X4, self.X5], Y=[self.Y, self.Y,self.Y, self.Y,self.Y])
        else:
            self.X, self.Y, self.X_i, self.Y_i, self.index = Form_Incomplete_Data(seed, missrate=miss_rate, X=[self.X1, self.X2, self.X3, self.X4, self.X5], Y=[self.Y, self.Y, self.Y, self.Y, self.Y])

        self.view = 5

    def __len__(self):
        return len(self.X[0])

    def __getitem__(self, idx):
        return [torch.from_numpy(self.X[v][idx]) for v in range(self.view)]

class NUSWIDE(Dataset):
    def __init__(self, path, miss_rate, seed, flag):
        data = scipy.io.loadmat(path + "NUSWIDE.mat")
        self.X1 = data['X1'].astype(np.float32)
        self.X2 = data['X2'].astype(np.float32)
        self.X3 = data['X3'].astype(np.float32)
        self.X4 = data['X4'].astype(np.float32)
        self.X5 = data['X5'].astype(np.float32)
        self.Y = np.copy(data['Y']).astype(np.int32).reshape(5000,)
        rep_mapping = {14: 0, 19: 1, 23: 2, 28: 3, 29: 4}
        for i in range(len(self.Y)):
            idy = rep_mapping.get(self.Y[i])
            self.Y[i] = idy
        if flag:
            self.X, self.Y, self.X_i, self.Y_i, self.index = Form_Incomplete_Data(seed, missrate=miss_rate, X=[self.X1, self.X2, self.X3, self.X4, self.X5], Y=[self.Y, self.Y, self.Y, self.Y, self.Y])
            self.X_i = add_pure_random_noniid_noise(self.X_i)
            #self.X, self.Y, self.X_i, self.Y_i, self.index = Form_Incomplete_Data_NonIID(seed, missrate=miss_rate, X=[self.X1, self.X2, self.X3, self.X4, self.X5], Y=[self.Y, self.Y,self.Y, self.Y,self.Y])
        else:
            self.X, self.Y, self.X_i, self.Y_i, self.index = Form_Incomplete_Data(seed, missrate=miss_rate, X=[self.X1, self.X2, self.X3, self.X4, self.X5], Y=[self.Y, self.Y, self.Y, self.Y, self.Y])

        self.view = 5

    def __len__(self):
        return len(self.X[0])

    def __getitem__(self, idx):
        return [torch.from_numpy(self.X[v][idx]) for v in range(self.view)]

def load_data(dataset, miss_rate, seed, flag):
    if dataset == "BDGP":
        dataset = BDGP('./data/', miss_rate, seed, flag)
        dims = [1750, 79]
        view = 2
        data_size = 2500
        class_num = 5
    elif dataset == "Cifar10":
        dataset = Cifar10('./data/', miss_rate, seed, flag)
        dims = [512, 2048, 1024]
        view = 3
        class_num = 10
        data_size = 50000
    elif dataset == "Cifar100":
        dataset = Cifar100('./data/', miss_rate, seed, flag)
        dims = [512, 2048, 1024]
        view = 3
        class_num = 100
        data_size = 50000
    elif dataset == "MNIST_USPS":
        dataset = MNIST_USPS('./data/', miss_rate, seed, flag)
        dims = [784, 784]
        view = 2
        class_num = 10
        data_size = 5000
    elif dataset == "BBCSport":
        dataset = BBCSport('./data/', miss_rate, seed, flag)
        dims = [3183, 3203]
        view = 2
        data_size = 544
        class_num = 5
    elif dataset == "Hdigit":
        dataset = Hdigit('./data/', miss_rate, seed, flag)
        dims = [784, 256]
        view = 2
        data_size = 10000
        class_num = 10
    elif dataset == "REU":
        dataset = REU('./data/', miss_rate, seed, flag)
        dims = [2000, 2000, 2000, 2000, 2000]
        view = 5
        data_size = 1200
        class_num = 6
    elif dataset == "NUSWIDE":
        dataset = NUSWIDE('./data/', miss_rate, seed, flag)
        dims = [65, 226, 145, 74, 129]
        view = 5
        data_size =  5000
        class_num = 5
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num


