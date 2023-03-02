

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset as BaseDataset
from torch.utils.data import DataLoader, random_split
from torch import Tensor

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse
from time import time

import openai



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
    parser.add_argument('-rcl', '--recordlog', metavar = 'Rcl', type = bool, default = True, help = 'Record log in a csv file')

    args = parser.parse_args()
    return args


class FCN(nn.Module):
    def __init__(self,m: int,nhn: int,n: int) -> None:
        super().__init__()
        # Setting bias = False diverges the net for sinosuidal data regression
        self.Lin = nn.Linear(m, nhn, bias=True)
        # TO DO: OrderedDict for nn.Sequential
        self.LH = nn.Linear(nhn, nhn, bias=True)
        self.Lout = nn.Linear(nhn, n, bias=True)


        self.FLOPs = 2*m*n + 2*nhn*nhn + 2*nhn*n

    def forward(self,x: Tensor) -> Tensor:
        # For a sinosuidal regression problem, relu and tanh proved to be more effective
        x = self.Lin(x)
        x = torch.relu(x) 
        # x = torch.tanh(x)
        x = self.LH(x)
        x = torch.relu(x) 
        # x = torch.tanh(x)
        x = self.Lout(x)
        # x = torch.relu(x) 
        # x = torch.sigmoid(x)
        return x
    
    def get_n_param(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params
    
    def get_FLOPs(self):
        return self.FLOPs


class Dataset(BaseDataset):
    def __init__(self,x: Tensor,y: Tensor) -> None:
        super().__init__()
        assert(x.shape[0]==y.shape[0])
        self.x = x
        self.y = y

    def __getitem__(self, index: int):
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
    
    # Learning rate between 1e-6 and 0.1
    lr = args.learning_rate
    momentum = args.momentum
    batch_size = args.batch_size
    batch_step = args.batch_step

    plotting = args.plotting
    verbose = args.verbose
    recordlog = args.recordlog

    

    # Data perparation
    t_dp = time()
    # data_desc = 'linear'
    data_desc = 'sin'

    x = torch.linspace(0, 1, n_samples).unsqueeze(-1)
    # y = x + torch.rand([n_samples, 1])/10 - 0.05
    y = np.sin(10*x) + torch.rand([n_samples, 1])/10 - 0.05

    x_tst = torch.linspace(0, 1, int(n_samples/10)).unsqueeze(-1)
    # y_tst = x_tst + torch.rand([int(n_samples/10), 1])/10 - 0.05
    y_tst = np.sin(10*x_tst) + torch.rand([int(n_samples/10), 1])/10 - 0.05


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

    if verbose:
        print(f"Data preparation time: {np.round(time()-t_dp,3)} s")


    model_name = f"best_model_n{n_samples}_hn{n_hid_nodes}_{data_desc}"

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

                
        
        if verbose:
            print(f"FLOPs : \t{model.get_FLOPs()}")
            print(f"Total parameters: \t{model.get_n_param()[0]}")
            print(f"Trainable parameters: \t{model.get_n_param()[1]}\n")

            print("Architecture:")
            print(model)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Total parameters: {total_params}")


        model = model.to(device)


        # Blind prediction
        y_hat = model(dataset.x[0:100])

        # Training
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        loss_t_all = torch.ones(n_epochs)
        loss_v_all = torch.ones(n_epochs)
        loss_tst_all = torch.ones(n_epochs)

        if recordlog:
            with open(f'./Weights/{model_name}.csv','w') as f:
                f.write('time,epoch,loss_t,loss_v\n')

        t_tr = time()
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
            if plotting:
                loss_t_all[epoch]=loss_t
                loss_v_all[epoch]=loss_v
                loss_tst_all[epoch]=loss_tst

            if recordlog:
                with open(f'./Weights/{model_name}.csv','a') as f:
                    f.write(f"{time()-t_tr},{epoch},{loss_t},{loss_v}\n")

            if loss_t < loss_v:
                print(f"Saving the model at epoch = {epoch}:  \tTrain loss: {loss_t}\tValidation loss: {loss_v}")
                torch.save(model, f'./Weights/{model_name}.pth')
        
        if verbose:
            print(f"Training time: {np.round(time()-t_tr,3)} s")

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


        criterion = nn.MSELoss()

        model = torch.load(f'./Weights/{model_name}.pth')
        t_tst = time()
        y_test_prd = model.forward(x_tst)
        loss_t = criterion(y_test_prd, y_tst)
        print(f"Testing loss: {loss_t}")
        if verbose:
            print(f"Testing time: {np.round(time()-t_tst,3)} s")

        
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


