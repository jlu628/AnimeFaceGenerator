import torch

def train_discriminator(G, D, images, loss_fn, optimizer, latent_size=(128,1,1), batch_size=64,  device='cuda'):
    optimizer.zero_grad()

    real_labels = (torch.ones(batch_size, 1)-0.1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)

    real_preds = D(add_noise(images))
    real_loss = loss_fn(real_preds, real_labels)
    
    latent = torch.randn(batch_size, *latent_size).to(device)
    fake_images = G(latent)
    fake_preds = D(add_noise(fake_images))
    fake_loss = loss_fn(fake_preds, fake_labels)

    d_loss = real_loss + fake_loss
    d_loss.backward()
    optimizer.step()

    return d_loss.item()


def train_generator(G, D, loss_fn, optimizer, latent_size=(128,1,1), batch_size=64,  device='cuda'):
    optimizer.zero_grad()

    real_labels = torch.ones(batch_size, 1).to(device)

    latent = torch.randn(batch_size, *latent_size).to(device)
    fake_images = G(latent)
    fake_preds = D(add_noise(fake_images))
    g_loss = loss_fn(fake_preds, real_labels)
    
    g_loss.backward()
    optimizer.step()
    
    return g_loss.item()


def evaluate(G, D, loss_fn, val_loader, latent_size=(128,1,1), batch_size=64,  device='cuda',):
    for batch_images in val_loader:
        with torch.no_grad():
            batch_images.to(device)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            latent = torch.randn(batch_size, *latent_size).to(device)
            fake_images = G(latent)

            real_preds = D(batch_images)
            real_loss = loss_fn(real_preds, real_labels)
            fake_preds = D(fake_images)
            fake_loss = loss_fn(fake_preds, fake_labels)

            d_loss = real_loss + fake_loss
            g_loss = loss_fn(fake_preds, real_labels)

            real_score = torch.mean(real_preds).item()
            fake_score = torch.mean(fake_preds).item()

    return d_loss.item(), g_loss.item(), real_score, fake_score, fake_images * 0.5 + 0.5


def add_noise(images, mean=0, std=0.0025):
    return images + torch.zeros(images.shape).data.normal_(mean, std)