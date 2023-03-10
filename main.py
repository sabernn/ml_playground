

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
import cv2
import argparse
from time import time
import os

import openai


class FCN(nn.Module):
    def __init__(self,m: int,nhn: int,n: int) -> None:
        super().__init__()
        # Setting bias = False diverges the net for sinosuidal data regression
        self.Lin = nn.Linear(m, nhn, bias=True)
        # TO DO: OrderedDict for nn.Sequential
        self.LH = nn.Linear(nhn, nhn, bias=True)
        self.Lout = nn.Linear(nhn, n, bias=True)


        self.FLOPs = 2*m*n + 2*nhn*nhn + 2*nhn*n

        self.loss_t_all = None
        self.loss_tst_all = None

        self.name = "FCN-best_model-"

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
    
    def train(self, datapack: list, args, name_appnd: str):
        self.name += name_appnd

        loader_trn = datapack[0]
        loader_val = datapack[1]
        # loader_tst = datapack[2]

        if args.verbose:
            print(f"FLOPs : \t{self.get_FLOPs()}")
            print(f"Total parameters: \t{self.get_n_param()[0]}")
            print(f"Trainable parameters: \t{self.get_n_param()[1]}\n")

            print("Architecture:")
            print(self)
            total_params = sum(p.numel() for p in self.parameters())
            print(f"Total parameters: {total_params}")


        self = self.to(args.device)

        optimizer = optim.Adam(self.parameters(), lr=args.learning_rate)
        criterion = nn.MSELoss()

        self.loss_t_all = torch.ones(args.n_epochs)
        self.loss_v_all = torch.ones(args.n_epochs)
        self.loss_tst_all = torch.ones(args.n_epochs)

        if args.recordlog:
            with open(f'./Weights/{self.name}.csv','w') as f:
                f.write('time,epoch,loss_t,loss_v\n')

        t_tr = time()
        for epoch in range(args.n_epochs):
            # if epoch%args.batch_step==0:
                # print(f"\nEpoch {epoch}\n")
            # with tqdm(loader_trn) as iterator:
            # train
            best_loss=0
            for x, y in loader_trn:
                optimizer.zero_grad()
                x, y = x.to(args.device), y.to(args.device)
                y_pred = self.forward(x)
                loss_t = criterion(y_pred, y)
                loss_t.backward()
                optimizer.step()
            
            # validation
            for x, y in loader_val:
                # optimizer.zero_grad()
                x, y = x.to(args.device), y.to(args.device)
                with torch.no_grad():
                    prediction = self.forward(x)
                    loss_v = criterion(prediction, y)
            
            

            # if epoch%args.batch_step==0:
                # print(f"Train loss: {loss_t}\tValidation loss: {loss_v}")
            if args.plotting:
                self.loss_t_all[epoch]=loss_t
                self.loss_v_all[epoch]=loss_v
                # self.loss_tst_all[epoch]=loss_tst

            if args.recordlog:
                with open(f'./Weights/{self.name}.csv','a') as f:
                    f.write(f"{time()-t_tr},{epoch},{loss_t},{loss_v}\n")

            if loss_t < loss_v:
                print(f"Saving the model at epoch = {epoch}:  \tTrain loss: {loss_t}\tValidation loss: {loss_v}")
                torch.save(self, f'./Weights/{self.name}.pth')
        
        if args.verbose:
            print(f"Training time: {np.round(time()-t_tr,3)} s")

        if args.plotting:
            fig, ax = plt.subplots()
            ax.plot(np.linspace(0,args.n_epochs,args.n_epochs),
                    self.loss_t_all.detach().cpu().numpy())
            
            ax.plot(np.linspace(0,args.n_epochs,args.n_epochs),
                    self.loss_v_all.detach().cpu().numpy())
            
            # ax.plot(np.linspace(0,args.n_epochs,args.n_epochs),
            #         self.loss_tst_all.detach().cpu().numpy())
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend = True
            ax.set_title("Loss evolution")
            
            plt.show()
    

    def test(self, loader_tst, args, name_appnd: str):
        self.name += name_appnd

        criterion = nn.MSELoss()

        model = torch.load(f'./Weights/{self.name}.pth')
        model = model.to(args.device)
        t_tst = time()
        # testing
        for x, y in loader_tst:
            # x, y = x.to(args.device), y.to(args.device)
            with torch.no_grad():
                prediction = model.forward(x)
                loss_tst = criterion(prediction, y)
        y_test_prd = model.forward(x_tst)
        loss_t = criterion(y_test_prd, y_tst)
        print(f"Testing loss: {loss_t}")
        if args.verbose:
            print(f"Testing time: {np.round(time()-t_tst,3)} s")

        
        if args.plotting:
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


def arg_parser():
    parser = argparse.ArgumentParser()
    # Data-related parameters
    parser.add_argument('-n', '--n_samples', metavar = 'N', type = int, default = 10000, help = 'Number of training and validation samples')
    parser.add_argument('-sp', '--split', metavar = 'Sp', type = float, default = 0.8, help = 'Training-validation split')
    parser.add_argument('-nt', '--n_test', metavar = 'Nh', type = int, default = 100, help = 'Number of test samples')
    parser.add_argument('-bs', '--batch_size', metavar = 'Bs', type = int, default = 200, help = 'Batch size')

    # Network hyperparameters
    parser.add_argument('-nhl', '--n_hid_layers', metavar = 'Nhl', type = int, default = 1, help = 'Number of hidden layers')
    parser.add_argument('-nhn', '--n_hid_nodes', metavar = 'Nhn', type = int, default = 10, help = 'Number of nodes at each hidden layer')

    # Training-related parameters
    parser.add_argument('-ep', '--n_epochs', metavar = 'Ep', type = int, default = 200, help = 'Number of epochs')
    parser.add_argument('-lr', '--learning_rate', metavar = 'Lr', type = float, default = 1e-4, help = 'Learning rate')
    parser.add_argument('-mom', '--momentum', metavar = 'Mom', type = float, default = 0.9, help = 'Momentum')
    parser.add_argument('-dvc','--device', metavar = 'Dvc', type = str, default = 'cpu', help = 'Training hardware/device')
    
    # Logging and reporting parameters
    parser.add_argument('-bstp', '--batch_step', metavar = 'Bstp', type = int, default = 10, help = 'Batch step for reporting when verbose is True')
    parser.add_argument('-plt', '--plotting', metavar = 'Plt', type = bool, default = True, help = 'Plotting the data and results')
    parser.add_argument('-vrb', '--verbose', metavar = 'Vrb', type = bool, default = True, help = 'Reporting the performance during training and evaluation')
    parser.add_argument('-rcl', '--recordlog', metavar = 'Rcl', type = bool, default = True, help = 'Record log in a csv file')

    # Running parameters
    parser.add_argument('-md', '--mode', metavar = 'Md', type = str, default = 'train', help = 'train or test')


    args = parser.parse_args()
    return args

def print_hypers(args):
    print(39*"#")
    print(11*"#"+" Hyperparameters "+11*"#")
    print(39*"#"+"\n")

    
    print("DATA:")
    print(f"Number of samples:\t\t{args.n_samples}")
    print(f"Training-validation split:\t{args.split}-{round(1.0-args.split,2)}")
    print(f"Batch size:\t\t\t{args.batch_size}")

    print("\nNETWORK:")
    print(f"Number of hidden layers:\t{args.n_hid_layers}")
    print(f"Number of hidden nodes:\t\t{args.n_hid_nodes}")


    print("\nTRAINING:")
    print(f"Mode:\t\t\t\t{args.mode}")
    print(f"Number of epochs:\t\t{args.n_epochs}")
    print(f"Momentum:\t\t\t{args.momentum}")
    
    print("\n"+39*"#")

def sweep_study(args, datapack, param_name: str, range: list, output_param):
    
    for args.param_name in range:
        print(f"{param_name} changed to {args.param_name}")

        # Model instantiation
        input_dim = datapack[0].dataset.dataset.x.shape[1]
        output_dim = datapack[0].dataset.dataset.y.shape[1]
        model = FCN(input_dim,args.n_hid_nodes,output_dim)

        model_name_append = f"n{args.n_samples}-nhn{args.n_hid_nodes}-{param_name}{args.param_name}-{data_desc}"

        output = []

        if args.mode == 'train':

            # # Model instantiation
            # model = FCN(1,args.n_hid_nodes,1)

            model.train(datapack= datapack, args = args, name_appnd= model_name_append)

            output.append(model.loss_t_all)

        elif args.mode == 'test':

            model.test(loader_tst=datapack[2], args= args, name_appnd= model_name_append)

            output.append(model.loss_tst_all)

def generate_sin(n_samples,a = 1,omg = 10,phi = 0, b = 0, noise_factor = 0.1, device = "cuda:0"):
    '''
    Generate sinosouidal function y = a * sin(omg * (x - phi)) + b + Noise
    '''
    data_desc = 'sin'
    x = torch.linspace(0, 1, n_samples).unsqueeze(-1)
    
    y = a * torch.sin(omg*(x-phi)) + b + noise_factor * (torch.rand([n_samples, 1]) - 0.5)
    # x = torch.cat((x,x),1)

    x = x.to(device)
    y = y.to(device)

    return x,y,data_desc

def generate_lin(n_samples, m = 1, h = 0, noise_factor = 0.1, device = "cuda:0"):
    '''
    Generate linear function y = m * x + h + Noise
    '''
    data_desc = 'lin'
    x = torch.linspace(0, 1, n_samples).unsqueeze(-1)
    y = m * x + h + noise_factor * (torch.rand([n_samples, 1]) - 0.5)
    x = torch.cat((x,x),1)

    x = x.to(device)
    y = y.to(device)

    return x,y,data_desc

def load_img(args ,mode = 'train', dataset_name = 'zeiss', flatten = True):
    '''
    Load image segmentation dataset.
    Data structure:
    data
      |-segzesiss
        |-train
          |-image
          |-mask
        |-test
          |-image
          |-mask

    '''
    data_desc = 'img_'+dataset_name

    # This folder should include image and mask subfolders
    data_dir = "./data/" + dataset_name + "/" + mode
    image_dir = os.path.join(data_dir,'image')
    mask_dir = os.path.join(data_dir,'mask')

    ids = os.listdir(image_dir)
    x_fps = [os.path.join(image_dir, image_id) for image_id in ids]
    y_fps = [os.path.join(mask_dir, image_id) for image_id in ids]

    x = torch.tensor([cv2.imread(i) for i in x_fps],dtype=torch.float32)
    y = torch.tensor([cv2.imread(i) for i in y_fps],dtype=torch.float32)

    if flatten:
        x = x.flatten().unsqueeze(-1)
        y = y.flatten().unsqueeze(-1)

    x = x.to(args.device)
    y = y.to(args.device)

    return x,y,data_desc



if __name__ == "__main__":

    # Input parameters
    args = arg_parser()

    # GPU
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {args.device}")

    if args.verbose:
        print_hypers(args= args)

    # Data perparation
    t_dp = time()

    x,y,data_desc = load_img(args, mode= 'train', dataset_name= 'zeiss0')
    x_tst,y_tst,data_desc_tst = load_img(args, mode= 'test', dataset_name= 'zeiss0')

    args.n_samples = x.shape[0]

    # x,y,data_desc = generate_sin(args.n_samples, device= args.device)
    # x_tst,y_tst,data_desc_tst = generate_sin(int(args.n_samples/10), device= args.device)

    # x,y,data_desc = generate_lin(args.n_samples, device= args.device)
    # x_tst,y_tst,data_desc_tst = generate_lin(int(args.n_samples/10), device= args.device)

    # Dataset object instantiation
    dataset = Dataset(x,y)
    dataset_tst = Dataset(x_tst,y_tst)
    
    dataset_trn_size = int(args.n_samples * args.split)
    dataset_val_size = args.n_samples - dataset_trn_size
    dataset_trn, dataset_val = random_split(dataset, [dataset_trn_size, dataset_val_size])

    loader_trn = DataLoader(dataset_trn, batch_size=args.batch_size, shuffle=True, num_workers=0)
    loader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True, num_workers=0)
    loader_tst = DataLoader(dataset_tst, batch_size=1, shuffle=True, num_workers=0)

    datapack = [loader_trn, loader_val, loader_tst]

    if args.verbose:
        print(f"Data preparation time: {np.round(time()-t_dp,3)} s")

    if args.plotting:
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

    # model = FCN(1,args.n_hid_nodes,1)

    output = sweep_study(
                args= args,
                datapack= datapack,
                param_name="n_hid_nodes",
                range= [1],
                output_param= "loss_t_all")
    
    # model_name_append = f"n{args.n_samples}-nhn{args.n_hid_nodes}-{data_desc}"
    
    # model = FCN(1,args.n_hid_nodes,1)
    # model.train(datapack=datapack,
    #             args= args,
    #             name_appnd=model_name_append)
    # model.test(loader_tst= loader_tst,
    #            args= args,
    #            name_appnd= model_name_append)
    

    print("The end")
