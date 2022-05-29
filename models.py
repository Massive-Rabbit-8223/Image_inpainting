import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

class Autoencoder(nn.Module):
    def __init__(self, device):
        super(Autoencoder, self).__init__()
        self.device = device

        self.encoder = nn.Sequential( # like the Composition layer you built
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        ).to(self.device)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
        ).to(self.device)

    def forward(self, x):
        x = self.encoder(x).to(self.device)
        x = self.decoder(x).to(self.device)
        return x

    def apply_init_weights(self):
        ## apply weight initialization ##
        pass

def train_network(model, optimizer, criterion, epochs, train_dataloader, val_dataloader):
    test_loss1_list = []
    test_acc1_list = []
    epoch_list = []
    for t in range(epochs):
        #print(f"Epoch {t+1}\n-------------------------------")
        
        train_loop(t, train_dataloader, model, criterion, optimizer)
        loss_acc_tuple = test_loop(val_dataloader, model, criterion, threshold=0.01)

        test_loss1_list.append(loss_acc_tuple[0])
        test_acc1_list.append(loss_acc_tuple[1])

        epoch_list.append(t+1)
    print("Done!")

def train_loop(epoch, dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    with tqdm(dataloader, unit="batch") as tepoch:
        for f, t, i, k in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            # Compute prediction and loss
            pred= model(f)
            loss = loss_fn(pred, i.to(model.device))

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tepoch.set_postfix(loss=loss.item())
            #if batch % 100 == 0:
            #    loss, current = loss.item(), batch * len(f)
            #    print(f"\rloss: {loss:>7f}  [{current:>5d}/{size:>5d}]",end='')


def test_loop(dataloader, model, loss_fn, threshold):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    i = 0
    with torch.no_grad():
        for f, t, i, k in dataloader:
            pred = model(f)
            test_loss += loss_fn(pred, i.to(model.device)).item()
            """
            diff = torch.abs(pred-y)

            if torch.where(diff < threshold, 0, 1).sum() == 0:
                correct += 1
            else:
                correct += 0
            """

            i += 1
            
    test_loss /= num_batches
    #correct /= num_batches
    #print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    #print(f"\rTest Error: \n Avg loss: {test_loss:>8f} \n", end='\n')

    return (test_loss, correct)