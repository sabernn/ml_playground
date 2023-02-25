

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset as BaseDataset
from torch.utils.data import DataLoader, random_split
from torch import Tensor

import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm

import argparse

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_samples', metavar = 'N', type = int, default = 10000, help = 'Number of training and validation samples')
    parser.add_argument('-tp', '--train_perc', metavar = 'Np', type = float, default = 0.8, help = 'Percentage of training samples from total number of samples')
    parser.add_argument('-nt', '--n_test', metavar = 'Nh', type = int, default = 10, help = 'Number of hidden nodes at each hidden layer')
    parser.add_argument('-ep', '--n_epochs', metavar = 'Ep', type = int, default = 200, help = 'Number of epochs')
    parser.add_argument('-nhl', '--n_hid_layers', metavar = 'Nhl', type = int, default = 1, help = 'Number of hidden layers')
    parser.add_argument('-nhn', '--n_hid_nodes', metavar = 'Nhn', type = int, default = 10, help = 'Number of nodes at each hidden layer')
    parser.add_argument('-md', '--mode', metavar = 'Md', type = str, default = 'train', help = 'train or test')

    parser.add_argument('-lr', '--learning_rate', metavar = 'Lr', type = float, default = 1e-4, help = 'Learning rate')
    parser.add_argument('-mom', '--momentum', metavar = 'Mom', type = float, default = 0.9, help = 'Momentum')
    parser.add_argument('-bs', '--batch_size', metavar = 'Bs', type = int, default = 200, help = 'Batch size')

    parser.add_argument('-bstp', '--batch_step', metavar = 'Bstp', type = int, default = 10, help = 'Batch step for reporting when verbose is True')

    parser.add_argument('-plt', '--plotting', metavar = 'Plt', type = bool, default = True, help = 'Plotting the data and results')
    parser.add_argument('-vrb', '--verbose', metavar = 'Vrb', type = bool, default = True, help = 'Reporting the performance during training and evaluation')

    args = parser.parse_args()
    return args





class FCN(nn.Module):
    def __init__(self,m,h,n) -> None:
        super().__init__()
        self.L1 = nn.Linear(m, h, bias=True)
        self.L2 = nn.Linear(h, n, bias=True)

    def forward(self,x: Tensor) -> Tensor:
        out1 = self.L1(x)
        return self.L2(out1)


class Dataset(BaseDataset):
    def __init__(self,x,y) -> None:
        super().__init__()
        assert(x.shape[0]==y.shape[0])
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

if __name__ == "__main__":

    # GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    args = arg_parser()

    # Inputs
    n_samples = args.n_samples
    n_hid_nodes = args.n_hid_nodes
    n_epochs = args.n_epochs
    mode = args.mode
    

    lr = args.learning_rate
    momentum = args.momentum
    batch_size = args.batch_size
    batch_step = args.batch_step

    plotting = args.plotting
    verbose = args.verbose

    

    # Data perparation
    # data_desc = 'linear'
    data_desc = 'sin'

    x = torch.linspace(0, 1, n_samples).unsqueeze(-1)
    # y = x + torch.rand([n_samples, 1])/10 - 0.05
    y = np.sin(4*x) + torch.rand([n_samples, 1])/10 - 0.05

    x_tst = torch.linspace(0, 1, int(n_samples/10)).unsqueeze(-1)
    # y_tst = x_tst + torch.rand([int(n_samples/10), 1])/10 - 0.05
    y_tst = np.sin(4*x_tst) + torch.rand([int(n_samples/10), 1])/10 - 0.05


    x = x.to(device)
    y = y.to(device)
    x_tst = x_tst.to(device)
    y_tst = y_tst.to(device)

    dataset = Dataset(x,y)
    dataset_tst = Dataset(x_tst,y_tst)

    
    dataset_trn_size = int(n_samples * 0.8)
    dataset_val_size = n_samples - dataset_trn_size
    dataset_trn, dataset_val = random_split(dataset, [dataset_trn_size, dataset_val_size])
    


    loader_trn = DataLoader(dataset_trn, batch_size=batch_size, shuffle=True, num_workers=0)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=0)
    loader_tst = DataLoader(dataset_tst, batch_size=1, shuffle=True, num_workers=0)


    model_name = f"best_model_n{n_samples}_hn{n_hid_nodes}_{data_desc}.pth"

    if plotting:
        fig, ax = plt.subplots()
        ax.plot(dataset_trn[:][0].cpu().numpy(),
                dataset_trn[:][1].cpu().numpy(),'*')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend = True
        ax.set_title("Training Data")
        
        # ax.plot(dataset_val[:][0].cpu().numpy(),
        #         dataset_val[:][1].cpu().numpy(),'o')
        
        plt.show()


    if mode == 'train':
        

        # Model instantiation
        model = FCN(1,n_hid_nodes,1)
        
        model = model.to(device)
        if verbose:
            print(model)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Total parameters: {total_params}")


        
        # Blind prediction
        y_hat = model(dataset.x[0:100])

        # Training
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        loss_t_all = torch.ones(n_epochs)
        loss_v_all = torch.ones(n_epochs)
        loss_tst_all = torch.ones(n_epochs)

        for epoch in range(n_epochs):
            # if epoch%batch_step==0:
                # print(f"\nEpoch {epoch}\n")
            # with tqdm(loader_trn) as iterator:
            # train
            best_loss=0
            for x, y in loader_trn:
                optimizer.zero_grad()
                x, y = x.to(device), y.to(device)
                y_pred = model.forward(x)
                loss_t = criterion(y_pred, y)
                loss_t.backward()
                optimizer.step()
            
            # validation
            for x, y in loader_val:
                # optimizer.zero_grad()
                x, y = x.to(device), y.to(device)
                with torch.no_grad():
                    prediction = model.forward(x)
                    loss_v = criterion(prediction, y)
            
            # testing
            for x, y in loader_tst:
                x, y = x.to(device), y.to(device)
                with torch.no_grad():
                    prediction = model.forward(x)
                    loss_tst = criterion(prediction, y)

            # if epoch%batch_step==0:
                # print(f"Train loss: {loss_t}\tValidation loss: {loss_v}")
            loss_t_all[epoch]=loss_t
            loss_v_all[epoch]=loss_v
            loss_tst_all[epoch]=loss_tst

            if loss_t < loss_v:
                print(f"Saving the model at epoch = {epoch}")
                print(f"Train loss: {loss_t}\tValidation loss: {loss_v}")
                torch.save(model, f'./Weights/{model_name}')

        if plotting:
            fig, ax = plt.subplots()
            ax.plot(np.linspace(0,n_epochs,n_epochs),
                    loss_t_all.detach().cpu().numpy())
            
            ax.plot(np.linspace(0,n_epochs,n_epochs),
                    loss_v_all.detach().cpu().numpy())
            
            ax.plot(np.linspace(0,n_epochs,n_epochs),
                    loss_tst_all.detach().cpu().numpy())
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend = True
            ax.set_title("Loss evolution")
            
            plt.show()



    # mode = 'test'

    # Testing


    elif mode == 'test':
        # Data perparation
        # x_tst = torch.linspace(0, 1, int(n_samples/10)).unsqueeze(-1)
        # y_test = x_test + torch.rand([int(n_samples/10), 1])/10
        # y_tst = x_tst 

        # x_tst = x_tst.to(device)
        # y_tst = y_tst.to(device)

        # dataset_tst = Dataset(x_tst,y_tst)

        # loader_tst = DataLoader(dataset_tst, batch_size=1, shuffle=True, num_workers=0)

        criterion = nn.MSELoss()


        model = torch.load(f'./Weights/{model_name}')

        y_test_prd = model.forward(x_tst)
        loss_t = criterion(y_test_prd, y_tst)

        if plotting:
            fig, ax = plt.subplots()
            ax.grid = True
            ax.plot(dataset_tst[:][0].cpu().numpy(),
                    dataset_tst[:][1].cpu().numpy(),'*')
            
            ax.plot(dataset_tst[:][0].cpu().numpy(),
                    y_test_prd.cpu().detach().numpy(),'o')
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.legend = True
            ax.set_title("Test Data")
        
            plt.show()

    else:
        print("Noting to do!")




        

    print("The end")


