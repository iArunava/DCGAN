from Loss import *

def train(FLAGS):
    # Define the hyperparameters
    p_every = FLAGS.p_every
    t_every = FLAGS.t_every
    e_every = FLAGS.e_every
    epochs = FLAGS.epochs
    dlr = FLAGS.dlr
    glr = FLAGS.glr
    beta1 = FLAGS.beta1
    beta2 = FLAGS.beta2

    # Optimizers

    d_opt = optim.Adam(D.parameters(), lr=dlr, betas=(beta1, beta2))
    g_opt = optim.Adam(G.parameters(), lr=glr, betas=(beta1, beta2))

    # Train loop
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    p_every = 80
    t_every = 1
    e_every = 1
    epochs = 100

    train_losses = []
    eval_losses = []

    dcgan = dcgan.to(device)

    dcgan.train()

    for e in range(epochs):

        td_loss = 0
        tg_loss = 0

        for batch_i, real_images in trainloader:

            # Scaling image to be between -1 and 1
            real_images = scale_images(real_images)

            real_images = real_images.to(device)

            batch_size = real_images.size(0)

            d_opt.zero_grad()

            z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            z = torch.from_numpy(z).float().to(device)

            d_fake, d_real = dcgan(real_images, z)

            r_loss = real_loss(d_real, smooth=True, device=device)
            f_loss = fake_loss(d_fake)
            d_loss = r_loss + f_loss

            td_loss += d_loss.item()

            d_loss.backward()
            d_opt.step()

            g_opt.zero_grad()

            g_loss = real_loss(d_fake)

            tg_loss += g_loss.item()

            g_loss.backward()
            g_opt.step()

            if batch_i % p_every == 0:
                print ('Epoch [{:5d} / {:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'. \
                        format(e+1, epochs, d_loss))

        train_losses.append([td_loss, tg_loss])

        if e % e_every:
            with torch.no_grad():
                G.eval()
                z = np.random.uniform(-1, 1, size=(batch_size, z_size))
                z = torch.from_numpy(z).float().to(device)
                d_fake = dcgan(None, z)
                e_loss = real_loss(d_fake)
            G.train()

            eval_losses.append(e_loss)

    print ('[INFO] Training Completed successfully!')
