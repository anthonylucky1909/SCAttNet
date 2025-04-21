import imageio
import os
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from utils import calculate_psnr, calculate_ssim  

def evaluate_and_create_gif(dataset, generator, device, max_samples=12, gif_path='super_resolution_demo.gif'):
    total_psnr, total_ssim = 0, 0
    count = 0
    frame_paths = []  
    
    generator.eval()
    
    os.makedirs('gif_frames', exist_ok=True)  # Create directory to store frames

    with torch.no_grad():
        for lr_batch, hr_batch in tqdm(dataset):
            try:
                if count >= max_samples:
                    break
                count += 1

                # Process images
                LR = lr_batch.to(device).unsqueeze(0) if lr_batch.dim() == 3 else lr_batch.to(device)
                HR = hr_batch.to(device).unsqueeze(0) if hr_batch.dim() == 3 else hr_batch.to(device)

                fake = generator(LR)
                psnr_value = calculate_psnr(HR, fake)
                ssim_value = calculate_ssim(HR, fake)
                total_psnr += psnr_value
                total_ssim += ssim_value

                # Convert tensors to numpy arrays
                sample_fake = fake[0].cpu().detach().clamp(0, 1).permute(1, 2, 0).float().numpy()
                sample_lr = LR[0].cpu().detach().clamp(0, 1).permute(1, 2, 0).float().numpy()
                sample_hr = HR[0].cpu().detach().clamp(0, 1).permute(1, 2, 0).float().numpy()

                # Create figure to show the progress
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))

                # Display images
                axes[0].imshow(sample_lr)
                axes[0].set_title("Low Res Input")
                axes[0].axis('off')

                axes[1].imshow(sample_fake)
                axes[1].set_title(f"Generated SR\nPSNR: {psnr_value:.2f} dB\nSSIM: {ssim_value:.4f}")
                axes[1].axis('off')

                axes[2].imshow(sample_hr)
                axes[2].set_title("High Res Target")
                axes[2].axis('off')

                plt.tight_layout()
                
                # Save frame as an image
                frame_path = f'gif_frames/frame_{count:03d}.png'
                plt.savefig(frame_path, bbox_inches='tight', dpi=100)
                frame_paths.append(frame_path)
                plt.close()
                
                print(f"Saved frame {count} - PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.4f}")

            except Exception as e:
                print(f"An error occurred while processing the batch: {e}")
                continue

    if frame_paths:
        frames = [imageio.imread(frame_path) for frame_path in frame_paths]
        imageio.mimsave(gif_path, frames, duration=1.0, loop=0)  # 1 second per frame
        print(f"\nGIF saved to {gif_path}")

        for frame_path in frame_paths:
            os.remove(frame_path)
        os.rmdir('gif_frames')
    
    if count > 0:
        average_psnr = total_psnr / count
        average_ssim = total_ssim / count
        print(f'\nAverage PSNR: {average_psnr:.2f} dB')
        print(f'Average SSIM: {average_ssim:.4f}')
    else:
        print("No samples processed, skipping average calculation.")
