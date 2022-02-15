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

    step, correct, total_loss = 0, 0, 0
    count = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        # Loop over each batch in the dataset
        for batch in tqdm(train_loader):
            images, labels = batch

            # Transfer to GPU
            # Does not work?
            images = images.to(device)
            labels = labels.to(device)
            
            # print(images.size())
            
            outputs = model(images) # forward prop

            loss = loss_fn(outputs, labels).mean() # compute loss

            total_loss += loss.item()
            correct += (torch.argmax(outputs, dim = 1) == labels).sum().item()
            count += len(labels)

            loss.backward() # backprop
            optimizer.step() # update the parameters using backprop    
            optimizer.zero_grad() # clear the gradients of the optimizer

            
            # Periodically evaluate our model + log to Tensorboard
            if step % n_eval == 0:
                accuracy = correct / count
                total_loss, correct, count = 0, 0, 0

                eval_loss, eval_accuracy = evaluate(val_loader, model, loss_fn)

                model.train()

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
    model.eval() # evaluation mode instead of training

    total_loss, correct, count = 0, 0, 0

    for batch in val_loader:
        images, labels = batch

        images = images.to(device)
        labels = labels.to(device)

        output = model(images)
        total_loss += loss_fn(output, labels).mean().item()
        correct += (torch.argmax(output, dim = 1) == labels).sum().item()
        count += len(labels)

    return total_loss, correct / count

        
