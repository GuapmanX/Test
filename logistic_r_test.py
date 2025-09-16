import torch
import torch.nn as nn
import numpy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

device = torch.device('cpu')

X, y = make_classification(
    n_samples=1000,
    n_features=17,
    n_informative=10,
    n_redundant=7,
    n_classes=2,
    random_state=21
)

n_samples, n_features = X.shape #data amount, and the data in the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=18)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

X_train = torch.from_numpy(X_train).type(torch.float32).to(device)
X_test = torch.from_numpy(X_test).type(torch.float32).to(device)
y_train = torch.from_numpy(y_train).type(torch.float32).to(device)
y_test = torch.from_numpy(y_test).type(torch.float32).to(device)