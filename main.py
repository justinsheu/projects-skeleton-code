import os
import torch
import constants
import logging
from data.Datasets import Dataset
from networks.Network import StartingNetwork
from train_functions.train import starting_train

def main():
    hyperparameters = {"epochs": constants.EPOCHS, "batch_size": constants.BATCH_SIZE}

    print("Epochs:", constants.EPOCHS)
    print("Batch size:", constants.BATCH_SIZE)

    logging.basicConfig(filename = constants.LOG_FILE, level = logging.DEBUG, filemode = 'w')

    # Initalize dataset and model. Then train the model!
    csv_root = os.path.join(os.getcwd(), 'data', 'csv')
    train_dataset = Dataset(os.path.join(csv_root, 'train.csv'))
    val_dataset = Dataset(os.path.join(csv_root, 'test.csv'))
    model = StartingNetwork()

    if torch.cuda.is_available():
        model.cuda()

    starting_train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        hyperparameters=hyperparameters,
        n_eval=constants.N_EVAL,
    )

if __name__ == "__main__":
    main()
