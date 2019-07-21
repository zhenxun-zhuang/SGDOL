if __name__ == "__main__":
    import argparse

    import torch
    import torch.nn as nn
    import torch.optim as optim

    from mnistconvnet import MNISTConvNet
    from mnist_data_loader import mnist_data_loader

    from sgdol import SGDOL

    # Parse input arguments.
    parser = argparse.ArgumentParser(description='MNIST CNN SGDOL')

    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='allow the use of CUDA (default: False)')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 0)')
    parser.add_argument('--train-epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--train-batchsize', type=int, default=100,
                        help='batchsize in training (default: 100)')        
    parser.add_argument('--dataroot', type=str, default='./data',
                        help='location to save the dataset')
    parser.add_argument('--optim-method', type=str, default='SGDOL',
                        choices=['SGDOL', 'Adam', 'SGD', 'Adagrad'],
                        help='the optimizer to be employed (default: SGDOL)')
    parser.add_argument('--smoothness', type=float, default=10.0, metavar='M',
                        help='to be used in SGDOL (default: 10)')
    parser.add_argument('--alpha', type=float, default=10.0,
                        help='to be used in SGDOL (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate of the chosen optimizer (default: 0.001)')

    args = parser.parse_args()

    # Check the availability of GPU.
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

	# Set the random seed for reproducibility.
    torch.manual_seed(args.seed)

	# Load data.
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    dataset_info = mnist_data_loader(root_dir=args.dataroot,
                                     batch_size=args.train_batchsize,
                                     valid_ratio=0,
                                     **kwargs)
    train_loader = dataset_info[0]
    test_loader = dataset_info[4]

	# Initialize the neural network model and move it to GPU if needed.
    net = MNISTConvNet()
    net.to(device)

	# Define the loss function.
    criterion = nn.CrossEntropyLoss()    

	# Select optimizer.
    optim_method = args.optim_method
    if optim_method == 'SGDOL':
        optimizer = SGDOL(net.parameters(),
						  smoothness=args.smoothness,
						  alpha=args.alpha)
    elif optim_method == 'SGD':
        optimizer = optim.SGD(net.parameters(),
							  lr=args.lr)
    elif optim_method == 'Adagrad':
        optimizer = optim.Adagrad(net.parameters(),
								  lr=args.lr)
    elif optim_method == 'Adam':
        optimizer = optim.Adam(net.parameters(),
							   lr=args.lr)
    else:
        raise ValueError("Invalid optimization method: {}".format(
			optim_method))

	# Train the model.
    all_train_losses = []
    for epoch in range(args.train_epochs):
        # Train the model for one epoch.
        net.train()

        for data in train_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

		# Evaluate the trained model over all training samples.
        net.eval()

        running_loss = 0.0
        with torch.no_grad():
            for data in train_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        all_train_losses.append(avg_train_loss)
        print('Epoch %d: Training Loss: %.4f' % (epoch + 1, avg_train_loss))

	# Evaluate the test error of the final model.
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    test_accu = 1.0 * correct / total
    print('Final Test Accuracy: %.4f\n' % (test_accu))

    # Write log files.
    if optim_method == 'SGDOL':
        opt_para = args.smoothness
    else:
        opt_para = args.lr
    
    train_loss_fname = ''.join(['logs/',
                                '{0}'.format(optim_method),
                                '_training_loss.txt'])
    with open(train_loss_fname, 'a') as f:
        f.write('{0}, {1}\n'.format(opt_para, all_train_losses))
    
    test_error_fname = ''.join(['logs/',
                                '{0}'.format(optim_method),
                                '_test_error.txt'])
    with open(test_error_fname, 'a') as f:
        f.write('{0}, {1}\n'.format(opt_para, test_accu))
