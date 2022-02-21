import matplotlib.pyplot as plt

epoch = [1, 2, 3, 4, 5]
data1 = [0.1, 0.2, 0.3, 0.4, 0.7]
data2 = [0.2, 0.4, 0.45, 0.6, 0.65]
data3 = [0.3, 0.32, 0.4, 0.7, 0.72]


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
    plt.ylabel('Loss')
    plt.xlabel('Epoch No.')
    plt.title(f'{result_type} vs. data set')
    plt.xticks(epochs, epochs)
    plt.show()

plot_curve(5, data1, data2, data3, 'loss')
plot_curve(5, data1, data2, data3, 'accuracy')