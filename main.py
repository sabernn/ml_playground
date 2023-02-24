

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset as BaseDataset
from torch.utils.data import DataLoader, random_split
from torch import Tensor

import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm




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


    # Inputs
    n_samples = 10000
    n_hid_nodes = 10
    n_epochs = 200
    mode = 'train'
    data_desc = 'linear'

    lr = 1e-4
    momentum = 0.9
    batch_size = 200
    batch_step = 10

    plotting = True
    verbose = False

    model_name = f"best_model_n{n_samples}_hn{n_hid_nodes}_{data_desc}.pth"


    if mode == 'train':
        # Data perparation
        
        x = torch.linspace(0, 1, n_samples).unsqueeze(-1)
        y = x + torch.rand([n_samples, 1])/10 - 0.05

        x_tst = torch.linspace(0, 1, int(n_samples/10)).unsqueeze(-1)
        y_tst = x_tst + torch.rand([int(n_samples/10), 1])/10 - 0.05


        if plotting:
            fig, ax = plt.subplots()
            ax.plot(x,y)

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

        if plotting:
            fig, ax = plt.subplots()
            ax.plot(dataset_trn[:][0].cpu().numpy(),
                    dataset_trn[:][1].cpu().numpy(),'*')
            
            ax.plot(dataset_val[:][0].cpu().numpy(),
                    dataset_val[:][1].cpu().numpy(),'o')
            
            plt.show()
        

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
            
            plt.show()



    # mode = 'test'

    # Testing


    if mode == 'test':
        # Data perparation
        x_tst = torch.linspace(0, 1, int(n_samples/10)).unsqueeze(-1)
        # y_test = x_test + torch.rand([int(n_samples/10), 1])/10
        y_tst = x_tst 

        x_tst = x_tst.to(device)
        y_tst = y_tst.to(device)

        dataset_tst = Dataset(x_tst,y_tst)

        loader_tst = DataLoader(dataset_tst, batch_size=1, shuffle=True, num_workers=0)

        


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
            
        
            plt.show()




        

    print("The end")


