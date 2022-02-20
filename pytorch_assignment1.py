# References for ML workflow
# Pytorch: Quickstart: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
# PyTorch Lightning: https://www.youtube.com/watch?v=OMDn66kM9Qc&ab_channel=PyTorchLightning
# Krishna Ramesh: https://www.youtube.com/watch?v=ijaT8HuCtIY&ab_channel=KrishnaRamesh 

# References to setup CNN
# https://shap.readthedocs.io/en/latest/example_notebooks/image_examples/image_classification/PyTorch%20Deep%20Explainer%20MNIST%20example.html
# https://towardsdatascience.com/mnist-handwritten-digits-classification-using-a-convolutional-neural-network-cnn-af5fafbc35e9
# https://stackoverflow.com/questions/63754645/convert-conv2d-to-linear

# Key concepts
# Conv layer needs matrices as inputs while linear layers need flattened vectors.
# Kernel size can't be greater than actual input size

# from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision import datasets
from torch import optim
from torch import nn
import matplotlib.pyplot as plt

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

# all_data = torch.utils.data.ConcatDataset([train_data, test_data])
all_data = train_data
print(f'There is now {len(all_data)} data instances in all_data.')

# Recut all_data: 70% training, 20% test, 10% validation.
idx_train = int(len(all_data)*0.7)
idx_test = idx_train + int(len(all_data)*0.2)
idx_validation = len(all_data)

train_indices = list(range(0, idx_train))
train_data_resliced = torch.utils.data.Subset(all_data, train_indices)
print(f'There is now {len(train_data_resliced)} data instances in train_data_resliced (70% of all_data).')

test_indices = list(range(idx_train, idx_test))
test_data_resliced = torch.utils.data.Subset(all_data, test_indices)
print(f'There is now {len(test_data_resliced)} data instances in test_data_resliced (20% of all_data).')

validation_indices = list(range(idx_test, idx_validation))
val_data = torch.utils.data.Subset(all_data, validation_indices)
print(f'There is now {len(val_data)} data instances in validation_data (10% of all_data).')


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define models
class NN_linear(nn.Module):
    def __init__(self):
        super(NN_linear, self).__init__()
        self.flatten = nn.Flatten() # converts 2D arrays into one contiguous array.
        self.linear_relu_stack = nn.Sequential( 
            nn.Linear(28*28, 64), # args for CNN: input channels, output channels, kernel_size
            nn.ReLU(), # an activation layer that introduces non-linearities to the linear model
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class NN_CNN(nn.Module):
    def __init__(self):
        super(NN_CNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5), # Args: channel in, channel out, kernel size
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout(),
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Conv2d(20, 40, kernel_size=3),
            nn.Dropout(),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(40, 50), #2nd layer creats 20x4x4 outputs. 3rd layer creates 40 outputs x 1 x 1 as f(kernel size, padding, output of layer 2)
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        # Since you need matrices for conv. layers and for vectors linear layers, you have to take the matrix an flatten it, for example a matrix of shape (m, n) would become a vector (m*n, 1).
        x = x.view(-1, x.size()[1] * x.size()[2] * x.size()[3]) # torch.Size([32, 20, 4, 4]). Batch size 32x of 20 output channels x 4 x 4
        x = self.fc_layers(x)
        return x


def train(dataloader, model, loss_function, optimiser):
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute predication error
        logits = model(X) # Predict the result given this images
        J = loss_function(logits, y)

        # Backpropagration
        model.zero_grad() # Equivalent to optimser.zero_grad() or params.grad._zero()
        J.backward() # Accumulate the partial derivatives of J wrt params. Equivalent to params.grad._sum(dJ/dparams)
        optimiser.step() # Step in the opposite direction of the gradient

        if batch % 100 == 0:
            loss, current = J.item(), batch * len(X)
            print(f"loss: {loss:>7f}, [{current:>5d}/{size:>5d}]")


def eval_model(dataloader, model, loss_fn, data_group):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"{data_group} Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")

    return 100*correct, test_loss


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


# Load data into Pytorch loader
train_loader = DataLoader(train_data_resliced, batch_size=32)
val_loader = DataLoader(val_data, batch_size=32)
test_loader = DataLoader(test_data_resliced, batch_size=32)

# Setup model inputs
model = NN_linear()
# model = NN_CNN()
optimiser = optim.SGD(model.parameters(), lr=1e-2)
loss_function = nn.CrossEntropyLoss() # This defines how the optimiser changes the NN parameters

# Train model
epochs = 5
acc_train = []
loss_train = []
acc_val = []
loss_val = []
acc_test = []
loss_test = []

for i in range(epochs):
    print(f"\n Epoch {i+1}\n-------------------------------")
    train(train_loader, model, loss_function, optimiser)

    _acc_train, _loss_train = eval_model(train_loader, model, loss_function, 'Training')
    _acc_val, _loss_val = eval_model(val_loader, model, loss_function, 'Validation')
    _acc_test, _loss_test = eval_model(test_loader, model, loss_function, 'Test')

    acc_train.append(_acc_train) 
    acc_val.append(_acc_val)
    acc_test.append(_acc_test)

    loss_train.append(_loss_train)
    loss_val.append(_loss_val)
    loss_test.append(_loss_test)

print("Done!")

plot_curve(epochs, acc_train, acc_val, acc_test, 'accuracy')
plot_curve(epochs, loss_train, loss_val, loss_test, 'loss')