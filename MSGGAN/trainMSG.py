import torch

def train_discriminator(G, D, downsampler, images, loss_fn, optimizer, latent_size=(128,1,1), batch_size=64,  device='cuda'):
    optimizer.zero_grad()

    real_labels = (torch.ones(batch_size, 1)-0.1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)

    downsampled_real = downsampler(images)
    real_preds = D(add_noise(downsampled_real, device=device))
    real_loss = loss_fn(real_preds, real_labels)
    
    latent = torch.randn(batch_size, *latent_size).to(device)
    intermediate_outputs = G(latent)
    fake_preds = D(intermediate_outputs)
    fake_loss = loss_fn(fake_preds, fake_labels)

    d_loss = real_loss + fake_loss
    d_loss.backward()
    optimizer.step()

    return d_loss.item()


def train_generator(G, D, loss_fn, optimizer, latent_size=(128,1,1), batch_size=64, device='cuda'):
    optimizer.zero_grad()

    real_labels = torch.ones(batch_size, 1).to(device)

    latent = torch.randn(batch_size, *latent_size).to(device)
    intermediate_outputs = G(latent)
    fake_preds = D(intermediate_outputs)
    g_loss = loss_fn(fake_preds, real_labels)
    
    g_loss.backward()
    optimizer.step()
    
    return g_loss.item()


def evaluate(G, D, downsampler, loss_fn, val_loader, latent_size=(128,1,1), batch_size=64, device='cuda',):
    for batch_images in val_loader:
        with torch.no_grad():
            batch_images = batch_images.to(device)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            downsampled_real = downsampler(batch_images)
            real_preds = D(add_noise(downsampled_real, device=device))
            real_loss = loss_fn(real_preds, real_labels)

            latent = torch.randn(batch_size, *latent_size).to(device)
            intermediate_outputs = G(latent)
            fake_preds = D(intermediate_outputs)
            fake_loss = loss_fn(fake_preds, fake_labels)
            fake_images = intermediate_outputs[0]

            d_loss = real_loss + fake_loss
            g_loss = loss_fn(fake_preds, real_labels)

            real_score = torch.mean(real_preds).item()
            fake_score = torch.mean(fake_preds).item()

    return d_loss.item(), g_loss.item(), real_score, fake_score, fake_images * 0.5 + 0.5


def add_noise(images, mean=0, std=0.025, device="cpu"):
    noised_images = []
    for image in images:
        noised_images.append(image + torch.zeros(image.shape).data.normal_(mean, std).to(device))
    return noised_images