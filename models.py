from turtle import pos, update
from matplotlib.pyplot import axis
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential( # like the Composition layer you built
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def apply_init_weights(self):
        ## apply weight initialization ##
        pass

def train_network(model, optimizer, criterion, n_epochs, train_dataloader, val_dataloader, device, experiment_id):
    # create tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join("results", experiment_id))
    size_train_loader = len(train_dataloader)

    # get initial evaluation on validation set
    model.eval()
    with torch.no_grad():
        for f, t, i, k in val_dataloader:
            pred = model(f.to(device))
            test_loss = criterion(pred, i.to(device))
    writer.add_scalar(tag="validation/loss", scalar_value=test_loss.cpu(), global_step=0)

    with tqdm(total=len(train_dataloader), unit="batch") as pbar:
        for epoch in range(n_epochs):
            
            counter = 0
            model.train()
        
            for f, t, i, k in train_dataloader:
                model.train()
                update = epoch*size_train_loader + counter
                pbar.set_description(f"Epoch {epoch}")

                # Compute prediction and loss
                pred = model(f.to(device))
                loss = criterion(pred, i.to(device))

                # Backpropagation
                loss.backward()
                optimizer.step()

                if (counter % 10) == 0:
                    # Add losses as scalars to tensorboard
                    writer.add_scalar(tag="training/loss", scalar_value=loss.cpu(), global_step=update)
                    # Add parameters (weights) and gradients as arrays to tensorboard
                    for i, (name, param) in enumerate(model.named_parameters()):
                        writer.add_histogram(tag=f"training/param_{i} ({name})", values=param.cpu(), global_step=update)
                        writer.add_histogram(tag=f"training/gradients_{i} ({name})", values=param.grad.cpu(), global_step=update)

                # reset gradients
                optimizer.zero_grad()
                counter += 1
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)

            model.eval()
            with torch.no_grad():
                for f, t, i, k in val_dataloader:
                    #print("f: ", f.shape)
                    pred = model(f.to(device))
                    test_loss = criterion(pred, i.to(device))
            writer.add_scalar(tag="validation/loss", scalar_value=test_loss.cpu(), global_step=update)
            pbar.refresh()
            pbar.reset()

    print("Done!")
    writer.close()

def create_data_dict(model, criterion, dataloader, device):
    ## create data dicts ##
    input_image_list = []
    pred_image_list = []
    target_image_list = []
    loss_list = []

    model.eval()
    with torch.no_grad():
        for f, t, i, k in dataloader:
            pred = model(f.to(device))
            loss = criterion(pred, i.to(device))

            input_image_list.append(f)
            pred_image_list.append(pred.cpu())
            target_image_list.append(i)
            loss_list.append(loss.cpu())

        data_dict = {
            "input_images": torch.cat(input_image_list, axis=0).cpu().detach().numpy(),
            "pred_images": torch.cat(pred_image_list, axis=0).cpu().detach().numpy(),
            "target_images": torch.cat(target_image_list, axis=0).cpu().detach().numpy(),
            "loss_list": torch.cat(loss_list, axis=0).cpu().detach().numpy().mean(axis=(1,2,3))
        }

    return data_dict

def create_test_predictions(model, device):
    """
    Create predictions from test set and return in correct format, 
    ready for upload to challenge server! 
    """
    pass