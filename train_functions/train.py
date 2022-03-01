import torch
import logging
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from constants import SAVE_FILE, device 
from networks.Network import StartingNetwork


def starting_train(train_dataset, val_dataset, model: StartingNetwork, hyperparameters, n_eval):
    """
    Trains and evaluates a model.

    Args:
        train_dataset:   PyTorch dataset containing training data.
        val_dataset:     PyTorch dataset containing validation data.
        model:           PyTorch model to be trained.
        hyperparameters: Dictionary containing hyperparameters.
        n_eval:          Interval at which we evaluate our model.
    """

    # Get keyword arguments
    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]

    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )

    model.to(device)
    model.resnet.to(device)

    # Initalize optimizer (for gradient descent) and loss function
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    step, correct, total_loss = 0, 0, 0
    count = 0
    for epoch in range(epochs):
        logging.info(f'Epoch {epoch + 1} of {epochs}\n')

        # Loop over each batch in the dataset
        for batch in tqdm(train_loader):
            images, labels = batch

            # Transfer to GPU
            # Does not work?
            images = images.to(device)
            labels = labels.to(device)
            
            
            outputs = model(images) # forward prop

            loss = loss_fn(outputs, labels).mean() # compute loss

            total_loss += loss.item()

            correct += (torch.argmax(outputs, dim = 1) == labels).sum().item()
            count += len(labels)

            logging.info(f'Training accuracy: {round(correct / count * 100, 2)}%')

            loss.backward() # backprop
            optimizer.step() # update the parameters using backprop    
            optimizer.zero_grad() # clear the gradients of the optimizer

            
            # Periodically evaluate our model + log to Tensorboard
            if step % n_eval == 0 and step != 0:
                model.eval()

                evaluate(val_loader, model, loss_fn)

                model.train()

            step += 1

        logging.info()

    torch.save(model.state_dict(), SAVE_FILE)


def evaluate(val_loader, model, loss_fn):
    correct = 0
    total = 0
    total_loss = 0

    logging.info('evaluating...\n')

    with torch.no_grad():
        for images, labels in val_loader:
            batch_images, batch_labels = images.to(device), labels.to(device)

            outputs = model(batch_images)
            total_loss += loss_fn(outputs, batch_labels).mean().item()
            predictions = torch.argmax(outputs, dim = 1)

            correct += (predictions == batch_labels).sum().item()

            total += len(predictions)

        logging.info(f'Total loss: {round(total_loss, 2)}')
        logging.info(f'Accuracy: {round(correct / total * 100, 2)}%')

        logging.info('end evaluation...\n')
