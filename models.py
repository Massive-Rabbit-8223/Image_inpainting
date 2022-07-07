from turtle import pos, update
from matplotlib.pyplot import axis
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import numpy as np

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential( # like the Composition layer you built
            nn.Conv2d(6, 16, 3, stride=2, padding=1),
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

class SimpleCNN(torch.nn.Module):
    def __init__(self, n_in_channels: int = 6, n_hidden_layers: int = 3, n_kernels: int = 32, kernel_size: int = 7):
        """Simple CNN with `n_hidden_layers`, `n_kernels`, and `kernel_size` as hyperparameters"""
        super().__init__()
        print("n_hidden_layers: ", n_hidden_layers)
        print("n_kernels: ", n_kernels)
        cnn = []
        for i in range(n_hidden_layers):
            cnn.append(torch.nn.Conv2d(
                in_channels=n_in_channels,
                out_channels=n_kernels,
                kernel_size=kernel_size,
                padding=int(kernel_size / 2)
            ))
            cnn.append(torch.nn.ReLU())
            n_in_channels = n_kernels
        self.hidden_layers = torch.nn.Sequential(*cnn)
        
        self.output_layer = torch.nn.Conv2d(
            in_channels=n_in_channels,
            out_channels=3,
            kernel_size=kernel_size,
            padding=int(kernel_size / 2)
        )
    
    def forward(self, x):
        """Apply CNN to input `x` of shape (N, n_channels, X, Y), where N=n_samples and X, Y are spatial dimensions"""
        cnn_out = self.hidden_layers(x)  # apply hidden layers (N, n_in_channels, X, Y) -> (N, n_kernels, X, Y)
        pred = self.output_layer(cnn_out)  # apply output layer (N, n_kernels, X, Y) -> (N, 1, X, Y)
        return pred

def train_network(model, optimizer, criterion, n_epochs, train_dataloader, val_dataloader, device, experiment_id):
    # create tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join("results", experiment_id))
    size_train_loader = len(train_dataloader)

    # get initial evaluation on validation set
    model.eval()
    val_loss_list = []
    with torch.no_grad():
        for f, i, k, _, _ in val_dataloader:
            pred = model(torch.cat([f, k], axis=1).to(device))
            test_loss = criterion(pred, i.to(device))
            val_loss_list.append(test_loss.cpu().item())
    writer.add_scalar(tag="validation/loss", scalar_value=torch.mean(torch.tensor(val_loss_list)), global_step=0)

    with tqdm(total=len(train_dataloader), unit="batch") as pbar:
        for epoch in range(n_epochs):
            
            counter = 0
            model.train()
        
            for f, i, k, _, _ in train_dataloader:
                model.train()
                update = epoch*size_train_loader + counter
                pbar.set_description(f"Epoch {epoch}")

                # Compute prediction and loss
                pred = model(torch.cat([f, k], axis=1).to(device))
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
            val_loss_list = []
            with torch.no_grad():
                for f, i, k, _, _ in val_dataloader:
                    pred = model(torch.cat([f, k], axis=1).to(device))
                    test_loss = criterion(pred, i.to(device))
                    val_loss_list.append(test_loss.cpu().item())
            writer.add_scalar(tag="validation/loss", scalar_value=torch.mean(torch.tensor(val_loss_list)), global_step=update)
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
        for f, i, k, _, _ in dataloader:
            pred = model(torch.cat([f, k], axis=1).to(device))
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

def create_test_predictions(model, device, test_dataset):
    """
    Create predictions from test set and return in correct format, 
    ready for upload to challenge server! 
    """
    pred_pixels_list = []

    model.eval()
    with torch.no_grad():
        for input_image, known_array, _, _ in tqdm(test_dataset):
            pred_image = model(torch.from_numpy(np.concatenate([input_image, known_array.copy()], axis=0)).float().to(device))
               
            denorm_pred_image = test_dataset.denormalize_image(pred_image).cpu().detach().numpy()
            target_array = denorm_pred_image[known_array == 0].copy()
            pred_pixels_list.append(np.clip(target_array, 0, 255).astype(np.uint8))

    return pred_pixels_list