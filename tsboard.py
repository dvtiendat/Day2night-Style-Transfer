import tensorboard
from torch.utils.tensorboard import SummaryWriter

def log_losses(writer, d_loss, g_loss, cycle_loss, identity_loss, step):
    writer.add_scalar('Discriminator Loss', d_loss, global_step=step)
    writer.add_scalar('Generator Loss', g_loss, global_step=step)
    writer.add_scalar('Cycle Loss', cycle_loss, global_step=step)
    writer.add_scalar('Identity Loss', identity_loss, global_step=step)

def log_images(writer, real_day, real_night, fake_day, fake_night, step):
    writer.add_image('Real day', real_day, global_step=step)
    writer.add_image('Real night', real_night, global_step=step)
    writer.add_image('Fake day', fake_day, global_step=step)
    writer.add_image('Fake night', fake_night, global_step=step)