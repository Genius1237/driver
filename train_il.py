import torchvision
import torch
import torch.nn as nn
import torch.optim as optim

import argparse
import h5py
import time

def get_model(pretrained,num_classes):
    #model = torchvision.models.alexnet(pretrained)
    model = torchvision.models.AlexNet(num_classes)
    return model

class ImitationLearningDataset(torch.utils.data.Dataset):

    def __init__(self, file_name):
        self.file_name = file_name
        file = h5py.File(file_name,'r')
        self.len = file['X'].shape[0]
        self.transforms = torchvision.transforms.ToTensor()

    def __len__(self):
        return self.len
    
    def __getitem__(self,i):
        with h5py.File(self.file_name,'r') as file:
            X = file['X'][i]
            X = self.transforms(X)
            Y = torch.tensor(file['Y'][i],dtype=torch.long)

        return X,Y

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data",type=str,default='temp.hdf5',help='location of file with training data')
    parser.add_argument("--model", type=str,default='alexnet',help='name of model architecture to use')
    parser.add_argument("--batch-size",type=int,default=64,help='batch size to use for training')
    parser.add_argument("--n-epochs",type=int,default=100,help='number of epochs to train for')
    parser.add_argument("--device",type=str,default='cpu',help='device to run the model on [cpu cuda]')

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("cuda selected, but not available")
        exit()


    dataset = ImitationLearningDataset(args.data)
    dataloader = torch.utils.data.DataLoader(dataset,args.batch_size)

    n_examples = len(dataset)
    n_outputs = 3 + 2
    model = get_model(False,n_outputs)
    model.to(device=args.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(args.n_epochs):
        model.zero_grad()

        loss = 0
        t = time.time()
        for X,Y in dataloader:
            X = X.to(device=args.device)
            Y = Y.to(device=args.device)
            output = model(X)
            loss += criterion(output[...,:2],Y[...,:1].view(-1))
            loss += criterion(output[...,2:],Y[...,1:].view(-1))

        loss.backward()
        optimizer.step()
        print("Epoch {} Loss {}".format(epoch,loss.item()))
    '''
    with h5py.File(args.data,'r') as f:
        data_x = f['X']
        data_y = f['Y']

        n_examples = data_x.shape[0]
        n_outputs = 3 + 2
        model = get_model(False,n_outputs)
        model.to(device=args.device)

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters())
        image_transform = torchvision.transforms.ToTensor()
        for epoch in range(args.n_epochs):
            model.zero_grad()

            loss = 0
            for i in range(0,n_examples,args.batch_size):
                start = i
                end = min(start+args.batch_size,n_examples)

                t = time.time()
                batch_x_t = data_x[start:end]
                batch_x = []
                for i in range(batch_x_t.shape[0]):
                    batch_x.append(image_transform(batch_x_t[i]))
                batch_x = torch.stack(batch_x)
                batch_x = batch_x.to(device=args.device)

                print(time.time()-t)
                continue
                batch_y = torch.tensor(data_y[start:end],device=args.device)

                print(batch_x.shape)
                output = model(batch_x)

                loss += criterion(output[...,:2],batch_y[...,:1])/(end-start)
                loss += criterion(output[...,2:],batch_y[...,1:])/(end-start)
            continue
            loss.backward()
            optimizer.step()

            print("Epoch: {} Loss: {}".format(epoch,loss.value()))
    '''
if __name__ == "__main__":
    main()