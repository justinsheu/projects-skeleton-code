import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from constants import device 


def starting_train(train_dataset, val_dataset, model, hyperparameters, n_eval):
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

    # Initalize optimizer (for gradient descent) and loss function
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    step = 0
    for epoch in range(epochs):
        count = 0
        model.train() # change to train mode
        print(f"Epoch {epoch + 1} of {epochs}")

        total_loss = []

        # Loop over each batch in the dataset
        for batch in tqdm(train_loader):
            count += 1
            # TODO: Backpropagation and gradient descent
            images, labels = batch

            # Transfer to GPU
            # Does not work?
            images = images.to(device)
            labels = labels.to(device)
            
            # print(images.size())
            
            outputs = model(images) # forward prop

            loss = loss_fn(outputs.squeeze(), labels) # compute loss           
            
            loss.backward() # backprop

            total_loss.append(loss) # append this loss to the loss list

            optimizer.step() # update the parameters using backprop    
            optimizer.zero_grad() # clear the gradients of the optimizer

            
            # Periodically evaluate our model + log to Tensorboard
            if step % n_eval == 0:
                average_loss = sum(total_loss) / count
                # TODO:
                # Compute training loss and accuracy.
                # Log the results to Tensorboard.
                model.eval()

                

                
        

                # TODO:
                # Compute validation loss and accuracy.
                # Log the results to Tensorboard.
                # Don't forget to turn off gradient calculations!

                evaluate(val_loader, model, loss_fn)

            step += 1

        print()


def compute_accuracy(outputs, labels):
    """
    Computes the accuracy of a model's predictions.

    Example input:
        outputs: [0.7, 0.9, 0.3, 0.2]
        labels:  [1, 1, 0, 1]

    Example output:
        0.75
    """

    n_correct = (torch.round(outputs) == labels).sum().item()
    n_total = len(outputs)
    return n_correct / n_total


def evaluate(val_loader, model, loss_fn):
    """correct = 0
    
    for images, labels in val_loader:
        batch_images, batch_labels = images.to(device), labels.to(device)
        outputs = torch.argmax(model(batch_images), dim = 1)
        correct += (outputs == batch_labels).sum().item()
    print(f'{100 * correct / len(val_loader)}% accuracy')"""
        
