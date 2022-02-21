# References for ML workflow
# Pytorch: Quickstart: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
# PyTorch Lightning: https://www.youtube.com/watch?v=OMDn66kM9Qc&ab_channel=PyTorchLightning
# Krishna Ramesh: https://www.youtube.com/watch?v=ijaT8HuCtIY&ab_channel=KrishnaRamesh 

# References to setup CNN
# https://www.youtube.com/watch?v=wnK3uWv_WkU&list=RDCMUCkzW5JSFwvKRjXABI-UTAkQ&start_radio=1&rv=wnK3uWv_WkU&t=601&ab_channel=AladdinPersson
# https://shap.readthedocs.io/en/latest/example_notebooks/image_examples/image_classification/PyTorch%20Deep%20Explainer%20MNIST%20example.html
# https://towardsdatascience.com/mnist-handwritten-digits-classification-using-a-convolutional-neural-network-cnn-af5fafbc35e9
# https://stackoverflow.com/questions/63754645/convert-conv2d-to-linear

# Key concepts
# Conv layer needs matrices as inputs while linear layers need flattened vectors.
# Kernel size can't be greater than actual input size

import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision.transforms import ToTensor
from torchvision import datasets
from torch import optim
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

def get_data(train_cut=0.7, val_cut=0.1, test_cut=0.2, concat=False):
    """Get the MNIST dataset via pytorch.
    Then slice the MNIST dataset into a training, validation and test set.

    Args:
        train_cut (float, optional): Fraction of total MNIST data allocated for training. Defaults to 0.7.
        val_cut (float, optional): Fraction of total MNIST data. Defaults to 0.1.
        test_cut (float, optional): Fraction of total MNIST data. Defaults to 0.2.
        concat (boolean, optional): True to concatenat the MNIST training and data set before slicing.
    """
    train_data = datasets.MNIST(
        root = 'data',
        train = True,                         
        transform = ToTensor(), 
        download = True,            
    )

    test_data = datasets.MNIST(
        root = 'data', 
        train = False, 
        transform = ToTensor()
    )

    if concat:
        all_data = ConcatDataset([train_data, test_data])
    else:
        all_data = train_data


    idx_train = int(len(all_data)*train_cut)
    idx_validation = idx_train + int(len(all_data)*val_cut)

    train_indices = list(range(0, idx_train))
    train_data_resliced = Subset(all_data, train_indices)

    validation_indices = list(range(idx_train, idx_validation))
    val_data = Subset(all_data, validation_indices)

    test_indices = list(range(idx_validation, len(all_data)))
    test_data_resliced = Subset(all_data, test_indices)

    print_banner('Getting the MNIST dataset...')
    print(f'Contenate MNIST: {concat}.')
    print(f'There is {len(all_data)} data instances in the set before reslicing.')

    print_banner('Reslicing the dataset...')
    print(f'Len training is {len(train_data_resliced)} ({train_cut} of the initial set).')
    print(f'Len validation set {len(val_data)} ({val_cut} of the initial set).')
    print(f'Len test set is {len(test_data_resliced)} ({test_cut} of the initial set).')

    return(train_data_resliced, val_data, test_data_resliced)

# Define models
class NN_linear(nn.Module):
    def __init__(self):
        super(NN_linear, self).__init__()
        self.flatten = nn.Flatten() # converts 2D arrays into one contiguous array.
        self.linear_relu_stack = nn.Sequential( 
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class NN_CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(NN_CNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1), # Input channel is 1 because MNIST is an image with 1 grey scale channel. Output channels is arbitrary.
            nn.MaxPool2d(kernel_size=2, stride=2), # maxpool with kernel size 2 halves the 28x28 image into 14x14
            nn.ReLU(),

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),  # maxpool with kernel size 2 halves the 14x14 into 7x7
            nn.ReLU(),

        )
        self.fc_layers = nn.Sequential(
            nn.Linear(16*7*7, num_classes), # Do not add an activation function after this. This is the final output layer!
        )

    def forward(self, x):
        # CNNs are designed for images, so we can input the image directly
        x = self.conv_layers(x) 
        
        # Linear layers need a flat array
        x = x.reshape(x.shape[0], -1)
        x = self.fc_layers(x)
        return x


def train_model(dataloader, model, loss_function, optimiser):
    num_images_in_dataset = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to("cpu"), y.to("cpu")

        # Forward
        prediction = model(X)
        loss = loss_function(prediction, y)

        # Backwards
        model.zero_grad() # Equivalent to optimser.zero_grad() or params.grad._zero()
        loss.backward() # Accumulate the partial derivatives of J wrt params. Equivalent to params.grad._sum(dJ/dparams)
        optimiser.step() # Step in the opposite direction of the gradient

        if batch % 100 == 0:
            loss, count_images_processed = loss.item(), batch * len(X)
            print(f"Batch: {batch}, Loss: {loss:>0.4f}%, Images processed: {count_images_processed}/{num_images_in_dataset}")


def eval_model(dataloader, model, loss_function, data_group):
    size = len(dataloader.dataset)
    num_images_in_dataset = len(dataloader)
    model.eval()

    losses = 0
    correct_prediction= 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to("cpu"), y.to("cpu")

            # Forward
            prediction = model(X)
            losses += loss_function(prediction, y).item()
            correct_prediction += (prediction.argmax(1) == y).type(torch.float).sum().item()

    av_loss = losses / num_images_in_dataset
    correct_prediction /= size
    print(f"{data_group} Result: \n Accuracy: {(100*correct_prediction):>0.1f}%, Average loss: {av_loss:>5f}")

    return 100*correct_prediction, losses


def plot_curve(num_epochs, train_result, val_result, test_result, result_type=''):

    if len(train_result)!= num_epochs or len(val_result) != num_epochs or len(test_result) != num_epochs:
        print('Lengths don\'t match.')
        return

    epochs = list(range(1, num_epochs+1))

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(epochs, train_result, 'rs--', label='Training data')
    ax.plot(epochs, val_result, 'bs--', label='Validation data')
    ax.plot(epochs, test_result, 'g^--', label='Test data')
    ax.legend()
    plt.ylabel(result_type)
    plt.xlabel('Epoch No.')
    plt.title(f'{result_type} vs. data set')
    plt.xticks(epochs, epochs)
    plt.show()

def plot_image_and_label(data_label_pair: list):
    plt.imshow(np.squeeze(data_label_pair[0]))
    plt.title(f'Label: {data_label_pair[1]}')

def print_banner(message):
    print('\n**************************************************')
    print(message)
    print('**************************************************')

if __name__ == '__main__':
    # Set hyperparams
    epochs = 5

    # Get data
    train_data, val_data, test_data = get_data()

    # Load data into Pytorch loader
    train_loader = DataLoader(train_data, batch_size=32) 
    val_loader = DataLoader(val_data, batch_size=32)
    test_loader = DataLoader(test_data, batch_size=32)

    # Setup model
    # model = NN_linear()
    model = NN_CNN()
    optimiser = optim.SGD(model.parameters(), lr=1e-2)
    loss_function = nn.CrossEntropyLoss()

    # Train model over number of epochs
    acc_train = []
    loss_train = []
    acc_val = []
    loss_val = []
    acc_test = []
    loss_test = []

    for i in range(epochs):
        print_banner('Running Epoch '+ str(i+1) + ' of ' + str(epochs) +'...')
        train_model(train_loader, model, loss_function, optimiser)

        _acc_train, _loss_train = eval_model(train_loader, model, loss_function, 'Training')
        _acc_val, _loss_val = eval_model(val_loader, model, loss_function, 'Validation')
        _acc_test, _loss_test = eval_model(test_loader, model, loss_function, 'Test')

        acc_train.append(_acc_train) 
        acc_val.append(_acc_val)
        acc_test.append(_acc_test)

        loss_train.append(_loss_train)
        loss_val.append(_loss_val)
        loss_test.append(_loss_test)

    print_banner('Training complete! Phew that was exhausting.')

    # Plot results
    print_banner('Plotting...remember to close each plot to resume the script.')
    plot_curve(epochs, acc_train, acc_val, acc_test, 'accuracy')
    plot_curve(epochs, loss_train, loss_val, loss_test, 'loss')