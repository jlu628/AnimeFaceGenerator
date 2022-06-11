import torch

def train_discriminator(G, D, images, loss_fn, optimizer, lr_decay=0, latent_size=(128,1,1), batch_size=64,  device='cpu'):
    optimizer.zero_grad()
    optimizer.param_groups[0]['lr'] -= lr_decay


    real_labels = torch.ones(batch_size, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)

    real_preds = D(images)
    real_loss = loss_fn(real_preds, real_labels)
    
    latent = torch.randn(batch_size, *latent_size).to(device)
    fake_images = G(latent)
    fake_preds = D(fake_images)
    fake_loss = loss_fn(fake_preds, fake_labels)

    d_loss = real_loss + fake_loss
    d_loss.backward()
    optimizer.step()

    return d_loss.item()


def train_generator(G, D, loss_fn, optimizer, lr_decay=0, latent_size=(128,1,1), batch_size=64,  device='cpu'):
    optimizer.zero_grad()
    optimizer.param_groups[0]['lr'] -= lr_decay

    real_labels = torch.ones(batch_size, 1).to(device)

    latent = torch.randn(batch_size, *latent_size).to(device)
    fake_images = G(latent)
    fake_preds = D(fake_images)
    g_loss = loss_fn(fake_preds, real_labels)
    
    g_loss.backward()
    optimizer.step()
    
    return g_loss.item()