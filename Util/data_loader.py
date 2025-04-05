
import torch
from sklearn.model_selection import train_test_split

def Data_Loader(X, Y, TnT = False, shuffle=False, device='cpu'):

    if TnT:
        # Split the dataset into training and testing sets (80% train, 20% test)
        train_input, test_input, train_label, test_label = train_test_split(X, Y, test_size=0.2, shuffle=shuffle)
    else:
        train_input, test_input, train_label, test_label = X, X, Y, Y

    dataset = {}
    dtype = torch.get_default_dtype()
    dataset['train_input'] = torch.from_numpy(train_input).type(dtype).to(device)
    dataset['test_input'] = torch.from_numpy(test_input).type(dtype).to(device)
    dataset['train_label'] = torch.from_numpy(train_label).type(dtype).to(device)
    dataset['test_label'] = torch.from_numpy(test_label).type(dtype).to(device)
    return dataset