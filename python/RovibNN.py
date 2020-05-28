from torch.utils.data import DataLoader, Dataset
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.preprocessing import PolynomialFeatures
from mpl_toolkits.mplot3d import Axes3D

class Data(Dataset):
    def __init__(self):
#       In this version of MARVEL we have added 3 x 6 new features to represent
#       the categorical nature of each symmetry group        
        xy = np.loadtxt('./NH3_MARVELV18.states', dtype=np.float32)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples
        
    def polynom(self, order):
        poly = PolynomialFeatures(order)
#       generate polynormial features from first v1,v2,v3,v4,l3,l4,J,K,i
        polyx = torch.from_numpy(poly.fit_transform(self.x[:, 0:8])).float()
        return polyx[:, 1:]
    
    def normalize(self, polyx):
        polyx_norm = torch.from_numpy(normalize(polyx, norm='l2')).float()
        return polyx_norm
       
    def add_categorical(self, polyx):
#       concatnate poly features with categorical features (18 columns)
        self.x = torch.cat((polyx, self.x[:, 9:26]), dim=1)
  

class DataPlot:
    def __init__(self, dataset):
       # my_dpi = 100
       # plt.figure(figsize=(1664 / my_dpi, 936 / my_dpi), facecolor='white', dpi=my_dpi)
        plt.scatter(dataset.x[:, 1].cpu(), dataset.y.cpu(), c='b')
        plt.scatter(dataset.x[:, 0].cpu(), dataset.y.cpu(), c='k')
        self.bottom = torch.min(torch.min(dataset.x[:, 0], dataset.y)).item()
        self.top = torch.max(torch.max(dataset.x[:, 0], dataset.y)).item()
        plt.ylim(bottom=self.bottom, top=self.top)
        plt.pause(0.001)

        self.x_axis_ticks = torch.sort(dataset.x.cpu())[0]
        self.line = None
        self.dataset = dataset

    def update_curve(self, preds):
        if self.line is not None:
            self.line.remove()
        plt.scatter(self.dataset.x[:, 1].cpu(), preds.data, c='b')
        plt.scatter(self.dataset.x[:, 0].cpu(), preds.data, c='k')
        
        plt.pause(0.001)
        
    def ener_resids(self, resids):
        plt.scatter(self.dataset.y.cpu(), resids, c='r')

def load_checkpoint(model, optimizer, filename='test-7.chk'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    checkpoint = torch.load(filename)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer


def main():

    epochs = 10**6
    lr_rate = 0.003


    dataset = Data()

    trainloader = DataLoader(dataset=dataset, batch_size=10000, shuffle=True)
    
#     Introduce polynomial features
    # polyx = dataset.polynom(1)
    # print(dataset.x[:, 0:8].size())
    # print(dataset.x[:, 0:8])
    # print(polyx.size())
    # print(polyx)
    
#   Do we normalize our features
    # polyx = dataset.normalize(polyx)
   
#   Concatnate with categorical features
    # dataset.add_categorical(polyx)
    
    dataiter = iter(trainloader)
    data = dataiter.next()
    features, labels = data

#   works well
    model = nn.Sequential(nn.Linear(27, 100),
                          nn.Tanh(),
                          nn.Linear(100,100),
                          nn.Tanh(),
                          nn.Linear(100, 1))
    
#    works well for rotational states. lr_rate = 0.0001
    # model = nn.Sequential(nn.Linear(2, 6),
    #                       nn.Tanh(),
    #                       nn.Linear(6, 18),
    #                       nn.Tanh(),
    #                       nn.Linear(18, 6),
    #                       nn.Tanh(),
    #                       nn.Linear(6, 1))


    loss_func = torch.nn.MSELoss()

    optimizer = torch.optim.Rprop(model.parameters(), lr=lr_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate, momentum=0.7)

    # plot = DataPlot(dataset)
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)
    # load_checkpoint(model, optimizer)
    # lr_rate = 0.001


    for epoch in range(epochs):
        for features, labels in trainloader:
#           Compute the predictions
            y_pred = model(features)

#           Compute the loss
            loss = loss_func(y_pred, labels)

#           Zero the gradients
            optimizer.zero_grad()

#           Compute the gradients
            loss.backward()

#           Gradient step on the parameters
            optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            full_pred = model(dataset.x)
            total_loss = loss_func(full_pred, dataset.y)
            print('epoch: %5d | loss: %10.6f | rate: %10.6f' % (epoch, total_loss.item(), lr_rate))

#       switch optimiser after a certain number of iterations
       # if epoch == 300:
          # optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)
          # optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate, momentum=0.9, nesterov=True)
            

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(dataset.x[:, 0], dataset.x[:, 1], dataset.y)
    with torch.no_grad():
        model.eval()
        plot_pred = model(dataset.x)
    ax.scatter(dataset.x[:, 0], dataset.x[:, 1], plot_pred, c='k', marker = '*')
    plt.show()
    
    fig = plt.figure()
    ax = Axes3D(fig)    
    ax.scatter(dataset.x[:, 0], dataset.x[:, 1], dataset.y-plot_pred, c='r')
    plt.show()

    # with torch.no_grad():
    #     model.eval()
    #     resids = model(dataset.x) - dataset.y
    # plot.ener_resids(resids)

    for param in model.parameters():
        print(param.data)
        
    state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(), 'losslogger': total_loss.item(), }
    torch.save(state, "./test-8.chk")
    

if __name__ == "__main__":
    main()
