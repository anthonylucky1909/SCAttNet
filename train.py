from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

def train_super_resolution_model(epochs, train_dataloader, device, generator, discriminator,
                               optimizer_generator, optimizer_discriminator, lr, mse, bce, vgg_loss):
    generator.train()
    discriminator.train()

    for epoch in range(epochs):
        generator.train()
        discriminator.train()

        total_g_loss = 0.0
        total_d_loss = 0.0

        batch_iter = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)

        for batch_idx, (lr_batch, hr_batch) in enumerate(batch_iter):
            LR = lr_batch.to(device)
            HR = hr_batch.to(device)

            # Forward pass through the generator
            fake = generator(LR)
            disc_fake = discriminator(fake.detach())
            disc_real = discriminator(HR)

            disc_loss_real = bce(disc_real, torch.ones_like(disc_real))
            disc_loss_fake = bce(disc_fake, torch.zeros_like(disc_fake))
            disc_loss = disc_loss_fake + disc_loss_real

            optimizer_discriminator.zero_grad()
            disc_loss.backward()
            optimizer_discriminator.step()

            disc_fake = discriminator(fake)
            adversarial_loss = 1e-3 * bce(disc_fake, torch.ones_like(disc_fake))
            loss_for_vgg = 0.006 * vgg_loss(fake, HR)
            gen_loss = loss_for_vgg + adversarial_loss + mse(fake, HR)

            optimizer_generator.zero_grad()
            gen_loss.backward()
            optimizer_generator.step()

            total_g_loss += gen_loss.item()
            total_d_loss += disc_loss.item()
            batch_iter.set_postfix({"G Loss": f"{gen_loss.item():.4f}"})

        print(f"Epoch [{epoch+1}/{epochs}] - Generator Loss: {total_g_loss/len(train_dataloader):.4f}, "
              f"Discriminator Loss: {total_d_loss/len(train_dataloader):.4f}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_generator_state_dict': optimizer_generator.state_dict(),
            'optimizer_discriminator_state_dict': optimizer_discriminator.state_dict(),
        }
        torch.save(checkpoint, f"checkpoint_epoch_{epoch+1}.pth")