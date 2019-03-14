import matplotlib.pyplot as plt
import torch

def view_samples(epochs, samples):
    fig, axes = plt.subplots(figsize=(7, 7), nrows=4, ncols=4, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epochs]):
        img = img.detach()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.cpu().reshape((28, 28)), cmap='Greys_r')
    plt.show()

    
def view_one_random(gen):
    noise = torch.randn(1, 100, 1, 1, device=device)
    out = gen(noise)
    out = out.detach().cpu().squeeze(0).transpose(0, 1).transpose(1, 2).numpy()
    out = out * (0.5, 0.5, 0.5)
    out += (0.5, 0.5, 0.5)
    plt.axis('off')
    plt.imshow(out)
    plt.show()

def init_weight(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(0.1, 0.02)
        m.bias.data.constant_(1)

def init_weight(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(0.1, 0.02)
        m.bias.data.fill_(0)
