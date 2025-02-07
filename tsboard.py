import tensorboard
from torch.utils.tensorboard import SummaryWriter

def log_losses(writer, d_loss, g_loss, cycle_loss, identity_loss, step):
    writer.add_scalar('Discriminator Loss', d_loss, global_step=step)
    writer.add_scalar('Generator Loss', g_loss, global_step=step)
    writer.add_scalar('Cycle Loss', cycle_loss, global_step=step)
    writer.add_scalar('Identity Loss', identity_loss, global_step=step)

def log_images(writer, real_face, real_ukiyo, fake_face, fake_ukiyo, step):
    writer.add_image('Real Face', real_face, global_step=step)
    writer.add_image('Real Ukiyo', real_ukiyo, global_step=step)
    writer.add_image('Fake Face', fake_face, global_step=step)
    writer.add_image('Fake Ukiyo', fake_ukiyo, global_step=step)