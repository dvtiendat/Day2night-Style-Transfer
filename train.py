import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.generator import Generator 
from models.discriminator import Discriminator
from utils.helper import *
from dataset.dataset import FaceCycleGANDataset
from tsboard import *
import argparse
from tqdm import tqdm

writer = SummaryWriter('/kaggle/working/Style-Transfer/runs/day2night')
set_seed()

def get_args():
    parser = argparse.ArgumentParser(description='Train CycleGAN for daylight style transfer')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='batch size for training')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                       help='initial learning rate')
    parser.add_argument('--num_epochs', type=int, default=20,
                       help='number of epochs to train')
    parser.add_argument('--lambda_cycle', type=float, default=10.0,
                       help='weight for cycle consistency loss')
    parser.add_argument('--lambda_identity', type=float, default=0.0,
                       help='weight for identity loss')
    parser.add_argument('--img_size', type=int, default=256,
                       help='size of input images')
    parser.add_argument('--root_day', type=str, required=True,
                       help='directory path containing day images')
    parser.add_argument('--root_night', type=str, required=True,
                       help='directory path containing night images')
    parser.add_argument('--checkpoint_dir', type=str, default='/kaggle/working/Style-Transfer/checkpoints',
                       help='directory path to save model checkpoints')
    parser.add_argument('--load_checkpoint', action='store_true', default=False,
                       help='load model from checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='device to use for training')
    return parser.parse_args()

# A: day <-> B: night # 
def train_loop(D_A, D_B, G_A, G_B, optimizer_d, optimizer_g, d_scaler, g_scaler, mse, L1, dataloader, epoch):
    running_d_loss = 0.0
    running_g_loss = 0.0
    running_cycle_loss = 0.0
    running_identity_loss = 0.0
    
    progress = tqdm(dataloader, leave=True, desc=f'Epoch {epoch}')
    for idx, (day, night) in enumerate(progress):
        step = epoch * len(dataloader) + idx
        day = day.type(torch.float32).to(config['device'])
        night = night.type(torch.float32).to(config['device'])
        
        # Train Discriminator A and B #
        with torch.autocast('cuda'):
            fake_night = G_A(day) # day -> (G_A) -> night
            D_B_real = D_B(night) # discriminate real night
            D_B_fake = D_B(fake_night.detach()) # discriminate fake night
            D_B_real_loss = mse(D_B_real, torch.ones_like(D_B_real))
            D_B_fake_loss = mse(D_B_fake, torch.zeros_like(D_B_fake))
            D_B_loss = D_B_real_loss + D_B_fake_loss

            fake_day = G_B(night)
            D_A_real = D_A(day) # discriminate real day
            D_A_fake = D_A(fake_day.detach()) # discriminate fake day
            D_A_real_loss = mse(D_A_real, torch.ones_like(D_A_real))
            D_A_fake_loss = mse(D_A_fake, torch.zeros_like(D_A_fake))
            D_A_loss = D_A_real_loss + D_A_fake_loss

            D_loss = (D_A_loss + D_B_loss) / 2

        optimizer_d.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(optimizer_d)
        d_scaler.update()

        # Train Generator A and B #
        with torch.amp.autocast('cuda'):
            # Generator loss: try to trick the discriminator into think the fake image is real
            D_B_fake = D_B(fake_night)
            D_A_fake = D_A(fake_day)
            loss_G_B = mse(D_B_fake, torch.ones_like(D_B_fake)) 
            loss_G_A = mse(D_A_fake, torch.ones_like(D_A_fake))

            # Cycle consistency loss: make sure the image after cycle is similar to the original image
            cycle_day = G_B(fake_night) # fake_night -> (G_B) -> day
            cycle_night = G_A(fake_day) # fake_day -> (G_A) -> night
            cycle_day_loss = L1(day, cycle_day)
            cycle_night_loss = L1(night, cycle_night)

            # Identity loss: make sure the generator does not change the input image (gen ảnh từ ảnh gốc thì nó 0 đổi)
            identity_day = G_B(day)
            identity_night = G_A(night)
            identity_day_loss = L1(day, identity_day)
            identity_night_loss = L1(night, identity_night)
            
            g_loss = (loss_G_A + loss_G_B) + \
                config['lambda_cycle'] * (cycle_day_loss + cycle_night_loss) + \
                config['lambda_identity'] * (identity_day_loss + identity_night_loss)

        optimizer_g.zero_grad()
        g_scaler.scale(g_loss).backward()
        g_scaler.step(optimizer_g)
        g_scaler.update()

        log_losses(
            writer,
            D_loss.item(),
            (loss_G_A + loss_G_B).item(),
            (cycle_day_loss + cycle_night_loss).item(),
            (identity_day_loss + identity_night_loss).item(),
            step
        )

        if idx == len(dataloader) - 1: 
            log_images(
            writer,
            day[0].detach().cpu() * 0.5 + 0.5,
            night[0].detach().cpu() * 0.5 + 0.5,
            fake_day[0].detach().cpu() * 0.5 + 0.5,
            fake_night[0].detach().cpu() * 0.5 + 0.5,
            step
            )

        running_d_loss += D_loss.item()
        running_g_loss += (loss_G_A + loss_G_B).item()
        running_cycle_loss += (cycle_day_loss + cycle_night_loss).item()
        running_identity_loss += (identity_day_loss + identity_night_loss).item()

        progress.set_postfix({
            'D_loss': f'{D_loss.item():.4f}',
            'G_loss': f'{g_loss.item():.4f}'
        })
    
    avg_d_loss = running_d_loss / len(dataloader)
    avg_g_loss = running_g_loss / len(dataloader)
    avg_cycle_loss = running_cycle_loss / len(dataloader)
    avg_identity_loss = running_identity_loss / len(dataloader)

    if 'best_cycle_loss' not in globals():
        global best_cycle_loss
        best_cycle_loss = float('inf')

    if avg_cycle_loss < best_cycle_loss:
        best_cycle_loss = avg_cycle_loss
        save_checkpoint(G_A, optimizer_g, epoch, config['checkpoint_dir'], 'G_A_best.pth')
        save_checkpoint(G_B, optimizer_g, epoch, config['checkpoint_dir'], 'G_B_best.pth')
        save_checkpoint(D_A, optimizer_d, epoch, config['checkpoint_dir'], 'D_A_best.pth')
        save_checkpoint(D_B, optimizer_d, epoch, config['checkpoint_dir'], 'D_B_best.pth')

    # Save the last epoch
    if epoch == config['num_epochs'] - 1:
        save_checkpoint(G_A, optimizer_g, epoch, config['checkpoint_dir'], 'G_A_last.pth')
        save_checkpoint(G_B, optimizer_g, epoch, config['checkpoint_dir'], 'G_B_last.pth')
        save_checkpoint(D_A, optimizer_d, epoch, config['checkpoint_dir'], 'D_A_last.pth')
        save_checkpoint(D_B, optimizer_d, epoch, config['checkpoint_dir'], 'D_B_last.pth')

    return avg_d_loss, avg_g_loss, avg_cycle_loss, avg_identity_loss
            
def main():
    D_A = Discriminator(in_channels=3).to(config['device'])
    D_B = Discriminator(in_channels=3).to(config['device'])
    G_A = Generator(in_channels=3, features=64, num_residuals=9).to(config['device'])
    G_B = Generator(in_channels=3, features=64, num_residuals=9).to(config['device'])

    optimizer_d = optim.Adam(list(D_A.parameters()) + list(D_B.parameters()), lr=float(config['learning_rate']), betas=(0.5, 0.999))
    optimizer_g = optim.Adam(list(G_A.parameters()) + list(G_B.parameters()), lr=float(config['learning_rate']), betas=(0.5, 0.999))
    L1 = nn.L1Loss()
    mse = nn.MSELoss()
    transforms = get_transform(img_size=config['img_size'])

    start_epoch = 0
    if config['load_checkpoint']:
        start_epoch = load_checkpoint('/kaggle/input/g_a/pytorch/default/1/G_A_last.pth', G_A, optimizer_g, float(config['learning_rate']))
        load_checkpoint('/kaggle/input/g_b/pytorch/default/1/G_B_last.pth', G_B, optimizer_g, float(config['learning_rate']))
        load_checkpoint('/kaggle/input/d_a/pytorch/default/1/D_A_last.pth', D_A, optimizer_d, float(config['learning_rate']))
        load_checkpoint('/kaggle/input/d_b/pytorch/default/1/D_B_last.pth', D_B, optimizer_d, float(config['learning_rate']))
            
    dataset = FaceCycleGANDataset(root_day=config['root_day'], root_night=config['root_night'], transform=transforms)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2)
    g_scaler = torch.amp.GradScaler('cuda')
    d_scaler = torch.amp.GradScaler('cuda')

    for epoch in range(start_epoch, start_epoch + config['num_epochs']):
        avg_d_loss, avg_g_loss, avg_cycle_loss, avg_identity_loss = train_loop(D_A, D_B, G_A, G_B, optimizer_d, optimizer_g, d_scaler, g_scaler, mse, L1, dataloader, epoch)
        print(f'Epoch {epoch} | D_loss: {avg_d_loss:.4f} | G_loss: {avg_g_loss:.4f} | Cycle loss: {avg_cycle_loss:.4f} | Identity loss: {avg_identity_loss:.4f}')

        if avg_cycle_loss == best_cycle_loss:
            print(f'New best cycle loss: {best_cycle_loss:.4f}, best checkpoint saved')


if __name__ == '__main__':
    args = get_args()
    config = vars(args)
    main()