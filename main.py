import torch
from download_dataset import download_datasets
from config import device, lr, epochs, batch_size, num_workers, high_res, low_res, data_paths
from model import SuperResolutionNet, Discriminator, VGGLoss
from dataset import get_dataloaders
from train import train_super_resolution_model
from test import evaluate_and_create_gif
import torch.nn as nn

def main():
    # Download datasets first
    downloaded_paths = download_datasets()
    # Initialize models
    generator = SuperResolutionNet().to(device)
    discriminator = Discriminator().to(device)
    vgg_loss = VGGLoss().to(device)

    # Optimizers
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.0, 0.9))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.0, 0.9))

    # Loss functions
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    # Get data loaders
    train_loader, test_loaders = get_dataloaders()

    # Train the model
    train_super_resolution_model(
        epochs=epochs,
        train_dataloader=train_loader,
        device=device,
        generator=generator,
        discriminator=discriminator,
        optimizer_generator=optimizer_g,
        optimizer_discriminator=optimizer_d,
        lr=lr,
        mse=mse,
        bce=bce,
        vgg_loss=vgg_loss
    )

    # Evaluate on test sets
    for name, loader in test_loaders.items():
        print(f"\nEvaluating on {name} dataset...")
        # evaluate_model(loader, generator, device)
        evaluate_and_create_gif(loader, generator, device, max_samples=12, gif_path=str(name)+".gif"):

if __name__ == "__main__":
    main()