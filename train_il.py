import torchvision
import torch
import torch.nn as nn
import torch.optim as optim

import argparse
import h5py
import time
from tqdm import tqdm

def get_model(pretrained, num_classes):
    # model = torchvision.models.alexnet(pretrained=False, num_classes=num_classes)
    # model = torchvision.models.vgg11(pretrained=False, num_classes=num_classes)
    model = torchvision.models.resnet18(pretrained=False, num_classes=num_classes)
    return model

class ImitationLearningDataset(torch.utils.data.Dataset):

    def __init__(self, file_name, test=False):
        self.file_name = file_name
        file = h5py.File(file_name, 'r')
        self.xname = "{}_X".format("test" if test else "train")
        self.yname = "{}_Y".format("test" if test else "train")
        self.len = file[self.yname].shape[0]
        self.transforms = torchvision.transforms.Compose([
                            torchvision.transforms.ToPILImage(mode='RGB'),
                            torchvision.transforms.Resize((427,240)),
                            torchvision.transforms.ToTensor()
                        ])
        

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        with h5py.File(self.file_name, 'r') as file:
            X = file[self.xname][i]
            X = self.transforms(X)
            Y = torch.tensor(file[self.yname][i], dtype=torch.long)

        return X, Y


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--file", type=str, default='temp.hdf5', help='location of file with training data')
    parser.add_argument("--model", type=str, default='alexnet', help='name of model architecture to use')
    parser.add_argument("--batch-size", type=int,default=64, help='batch size to use for training')
    parser.add_argument("--n-epochs", type=int,default=100, help='number of epochs to train for')
    parser.add_argument("--device", type=str,default='cpu', help='device to run the model on [cpu, cuda]')

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("cuda selected, but not available")
        exit()


    train_dataset = ImitationLearningDataset(args.file,test=False)
    test_dataset = ImitationLearningDataset(args.file,test=True)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, args.batch_size)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, args.batch_size)

    n_examples = len(train_dataset)
    n_outputs = 3 + 2
    model = get_model(False, n_outputs)
    model.to(device=args.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(args.n_epochs):

        train_loss = 0
        valid_loss = 0
        for X, Y in tqdm(train_dataloader):
            loss = 0
            optimizer.zero_grad()

            X = X.to(device=args.device)
            Y = Y.to(device=args.device)
            output = model(X)
            loss += criterion(output[...,:2], Y[...,:1].view(-1))
            loss += criterion(output[...,2:], Y[...,1:].view(-1))
            
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()
            #input("Done")
            
        for X,Y in test_dataloader:
            loss = 0
            with torch.no_grad():
                X = X.to(device=args.device)
                Y = Y.to(device=args.device)
                output = model(X)
                loss += criterion(output[...,:2], Y[...,:1].view(-1))
                loss += criterion(output[...,2:], Y[...,1:].view(-1))
            valid_loss+=loss.item()

        print("Epoch {} Train Loss {} Validation Loss {}".format(epoch, train_loss, valid_loss))
    
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