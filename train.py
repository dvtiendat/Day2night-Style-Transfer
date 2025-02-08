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

writer = SummaryWriter('/kaggle/working/Ukiyo-e-style-transfer/runs/ukyio_face_cyclegan')
set_seed()

def get_args():
    parser = argparse.ArgumentParser(description='Train CycleGAN for face style transfer')
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
    parser.add_argument('--root_face', type=str, required=True,
                       help='directory path containing face images')
    parser.add_argument('--root_ukiyo', type=str, required=True,
                       help='directory path containing style images')
    parser.add_argument('--checkpoint_dir', type=str, default='/kaggle/working/Ukiyo-e-style-transfer/checkpoints',
                       help='directory path to save model checkpoints')
    parser.add_argument('--load_checkpoint', action='store_true', default=False,
                       help='load model from checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='device to use for training')
    return parser.parse_args()

# A: humans face <-> B: ukiyo-e face # 
def train_loop(D_A, D_B, G_A, G_B, optimizer_d, optimizer_g, d_scaler, g_scaler, mse, L1, dataloader, epoch):
    running_d_loss = 0.0
    running_g_loss = 0.0
    running_cycle_loss = 0.0
    running_identity_loss = 0.0
    
    progress = tqdm(dataloader, leave=True, desc=f'Epoch {epoch}')
    for idx, (face, ukiyo) in enumerate(progress):
        step = epoch * len(dataloader) + idx
        face = face.type(torch.float32).to(config['device'])
        ukiyo = ukiyo.type(torch.float32).to(config['device'])
        
        # Train Discriminator A and B #
        with torch.autocast('cuda'):
            fake_ukiyo = G_A(face) # face -> (G_A) -> ukiyo
            D_B_real = D_B(ukiyo) # discriminate real ukiyo
            D_B_fake = D_B(fake_ukiyo.detach()) # discriminate fake ukiyo
            D_B_real_loss = mse(D_B_real, torch.ones_like(D_B_real))
            D_B_fake_loss = mse(D_B_fake, torch.zeros_like(D_B_fake))
            D_B_loss = D_B_real_loss + D_B_fake_loss

            fake_face = G_B(ukiyo)
            D_A_real = D_A(face) # discriminate real face
            D_A_fake = D_A(fake_face.detach()) # discriminate fake face
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
            D_B_fake = D_B(fake_ukiyo)
            D_A_fake = D_A(fake_face)
            loss_G_B = mse(D_B_fake, torch.ones_like(D_B_fake)) 
            loss_G_A = mse(D_A_fake, torch.ones_like(D_A_fake))

            # Cycle consistency loss: make sure the image after cycle is similar to the original image
            cycle_face = G_B(fake_ukiyo) # fake_ukiyo -> (G_B) -> face
            cycle_ukiyo = G_A(fake_face) # fake_face -> (G_A) -> ukiyo
            cycle_face_loss = L1(face, cycle_face)
            cycle_ukiyo_loss = L1(ukiyo, cycle_ukiyo)

            # Identity loss: make sure the generator does not change the input image (gen ảnh từ ảnh gốc thì nó 0 đổi)
            identity_face = G_B(face)
            identity_ukiyo = G_A(ukiyo)
            identity_face_loss = L1(face, identity_face)
            identity_ukiyo_loss = L1(ukiyo, identity_ukiyo)
            
            g_loss = (loss_G_A + loss_G_B) + \
                config['lambda_cycle'] * (cycle_face_loss + cycle_ukiyo_loss) + \
                config['lambda_identity'] * (identity_face_loss + identity_ukiyo_loss)

        optimizer_g.zero_grad()
        g_scaler.scale(g_loss).backward()
        g_scaler.step(optimizer_g)
        g_scaler.update()

        log_losses(
            writer,
            D_loss.item(),
            (loss_G_A + loss_G_B).item(),
            (cycle_face_loss + cycle_ukiyo_loss).item(),
            (identity_face_loss + identity_ukiyo_loss).item(),
            step
        )

        if idx == len(dataloader) - 1: 
            log_images(
            writer,
            face[0].detach().cpu() * 0.5 + 0.5,
            ukiyo[0].detach().cpu() * 0.5 + 0.5,
            fake_face[0].detach().cpu() * 0.5 + 0.5,
            fake_ukiyo[0].detach().cpu() * 0.5 + 0.5,
            step
            )

        running_d_loss += D_loss.item()
        running_g_loss += (loss_G_A + loss_G_B).item()
        running_cycle_loss += (cycle_face_loss + cycle_ukiyo_loss).item()
        running_identity_loss += (identity_face_loss + identity_ukiyo_loss).item()

        progress.set_postfix({
            'D_loss': f'{D_loss.item():.4f}',
            'G_loss': f'{g_loss.item():.4f}'
        })
    
    avg_d_loss = running_d_loss / len(dataloader)
    avg_g_loss = running_g_loss / len(dataloader)
    avg_cycle_loss = running_cycle_loss / len(dataloader)
    avg_identity_loss = running_identity_loss / len(dataloader)

    if epoch % 5 == 0 or epoch == config['num_epochs'] - 1:
        save_checkpoint(G_A, optimizer_g, epoch, config['checkpoint_dir'], f'G_A_epoch_{epoch}.pth')
        save_checkpoint(G_B, optimizer_g, epoch, config['checkpoint_dir'], f'G_B_epoch_{epoch}.pth')
        save_checkpoint(D_A, optimizer_d, epoch, config['checkpoint_dir'], f'D_A_epoch_{epoch}.pth')
        save_checkpoint(D_B, optimizer_d, epoch, config['checkpoint_dir'], f'D_B_epoch_{epoch}.pth')

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
        start_epoch = load_checkpoint('/kaggle/input/g_a/pytorch/default/1/G_A.pth', G_A, optimizer_g, float(config['learning_rate']))
        load_checkpoint('/kaggle/input/g_b/pytorch/default/1/G_B.pth', G_B, optimizer_g, float(config['learning_rate']))
        load_checkpoint('/kaggle/input/d_a/pytorch/default/1/D_A.pth', D_A, optimizer_d, float(config['learning_rate']))
        load_checkpoint('/kaggle/input/d_b/pytorch/default/1/D_B.pth', D_B, optimizer_d, float(config['learning_rate']))
            
    dataset = FaceCycleGANDataset(root_face=config['root_face'], root_ukiyo=config['root_ukiyo'], transform=transforms)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2)
    g_scaler = torch.amp.GradScaler('cuda')
    d_scaler = torch.amp.GradScaler('cuda')

    for epoch in range(start_epoch, config['num_epochs']):
        avg_d_loss, avg_g_loss, avg_cycle_loss, avg_identity_loss = train_loop(D_A, D_B, G_A, G_B, optimizer_d, optimizer_g, d_scaler, g_scaler, mse, L1, dataloader, epoch)
        print(f'Epoch {epoch} | D_loss: {avg_d_loss} | G_loss: {avg_g_loss} | Cycle loss: {avg_cycle_loss} | Identity loss: {avg_identity_loss}')

if __name__ == '__main__':
    args = get_args()
    config = vars(args)
    main()