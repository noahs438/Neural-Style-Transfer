import argparse
import time
import torch
from pathlib import Path
from PIL import Image, ImageFile
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from custom_dataset import custom_dataset
from AdaIN_net import AdaIN_net, encoder_decoder
import torch.optim as optim
import matplotlib.pyplot as plt

# FROM AdaIN example GitHub
cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_transform():
    transform_list = [
        transforms.Resize((512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


def train(content_data, style_data, gamma, n_epochs, batch_size, encoder_path, decoder_path, plot_path, cuda):
    # Total time
    total_time = time.time()

    losses_train = []
    content_losses = []
    style_losses = []
    total_loss = 0

    if cuda == 'y' or cuda == 'Y':
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Loading in the dataset
    train_transforms = train_transform()
    content_loader = custom_dataset(content_data, train_transforms)
    style_loader = custom_dataset(style_data, train_transforms)

    # Print out the device being used - cuda or cpu
    print('Using device: ', device)

    n_batches = len(content_loader) / batch_size

    if n_batches % 1 != 0:
        n_batches = int(n_batches) + 1
    else:
        n_batches = int(n_batches)

    content_loader = DataLoader(content_loader, arguments.b, shuffle=True)
    style_loader = DataLoader(style_loader, arguments.b, shuffle=True)

    # Load the model and specify train mode
    encoder = encoder_decoder.encoder
    decoder = encoder_decoder.decoder
    encoder.load_state_dict(torch.load(encoder_path))
    model = AdaIN_net(encoder, decoder).to(device=device)
    model.train()  # keep track of gradient for backtracking

    # Define the optimizer and scheduler (for learning rate)
    optimizer = optim.Adam(decoder.parameters(), lr=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True,
                                                     min_lr=1e-4)

    content_loss = 0
    style_loss = 0
    comb_loss = 0
    # To loop through each dataset simultaneously
    for epoch in range(1, n_epochs + 1):
        # Epoch time
        epoch_time = time.time()

        for batch in range(n_batches):
            content_loss = 0
            style_loss = 0
            comb_loss = 0
            # Iterate through the datasets
            content_images = next(iter(content_loader)).to(device=device)
            style_images = next(iter(style_loader)).to(device=device)

            # Style transfer
            loss_c, loss_s = model(content_images, style_images)

            # Total loss
            loss_t = loss_c + (gamma * loss_s)

            # Add the losses to the total loss
            content_loss += loss_c
            style_loss += loss_s
            comb_loss += loss_t

            # Backpropagation
            optimizer.zero_grad()
            loss_t.backward()
            optimizer.step()

        # Save model after each epoch (just to show what version of the model is saved)
        torch.save(decoder.state_dict(), str(decoder_path + '_' + str(epoch) + '.pth'))
        # Save the model under its expected name
        torch.save(decoder.state_dict(), decoder_path)

        # Epoch time
        epoch_time = time.time() - epoch_time
        print('Epoch: {}/{} | Time: {:.2f}'.format(epoch, n_epochs, epoch_time))

        # Append the losses to the list
        content_losses.append(content_loss.item() / n_batches)
        style_losses.append(style_loss.item() / n_batches)
        losses_train.append(comb_loss.item() / n_batches)

        # Print out the losses for each epoch
        print('Content Loss: {:.4f} | Style Loss: {:.4f} | Total Loss: {:.4f}'.format(content_loss, style_loss,
                                                                                      comb_loss))

        # Scheduler step
        scheduler.step(comb_loss)

        total_loss = comb_loss / n_batches

        # Plotting the content, style and total training loss (content + style)
        plt.plot(content_losses, label='Content Loss')
        plt.plot(style_losses, label='Style Loss')
        plt.plot(losses_train, label='Content+Style')
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(plot_path)
        plt.close()

    # Total time
    total_time = time.time() - total_time
    print('Total time: {:.4f}'.format(total_time))

    return losses_train


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-content_dir', '--content_dir', type=str, required=True)
    parser.add_argument('-style_dir', '--style_dir', type=str, required=True)
    parser.add_argument('-gamma', type=float, default=1.0)
    parser.add_argument('-e', type=int, required=True, help='Number of epochs')
    parser.add_argument('-b', type=int, required=True, help='Batch size')
    parser.add_argument('-l', '--l', type=str, required=True, help='Encoder path')
    parser.add_argument('-s', '--s', type=str, required=True, help='Decoder path')
    parser.add_argument('-p', '--p', type=str, required=True, help='Plot path')
    parser.add_argument('-cuda', type=str, help='[Y/N]')

    arguments = parser.parse_args()

    train(arguments.content_dir, arguments.content_dir, arguments.gamma, arguments.e, arguments.b, arguments.l,
          arguments.s, arguments.p, arguments.cuda)
