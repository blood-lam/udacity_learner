# Import necessary dependencies
import logging
import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models.resnet import ResNet50_Weights

from smdebug import modes
from smdebug.pytorch import get_hook

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

import argparse

def test(model, test_loader, hook):
    '''
    Function to evaluate the model on the test dataset and log the accuracy and loss.
    '''
    model.eval()
    hook.set_mode(modes.EVAL)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += nn.functional.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    logger.info(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, hook):
    '''
    Function to train the model using the training dataset and validate using the validation dataset.
    Implements early stopping to prevent overfitting.
    '''
    model.train()
    hook.set_mode(modes.TRAIN)
    train_losses = []
    val_losses = []
    losses_dict = {'train_losses': [], 'val_losses': []}
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            if batch_idx % 100 == 0:
                logger.info(f'Train Epoch: {epoch+1}/{epochs} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        # Calculate average training loss for this epoch
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        hook.set_mode(modes.EVAL)
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                val_loss += criterion(output, target).item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        logger.info(f'Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

        losses_dict['train_losses'].append(avg_train_loss)
        losses_dict['val_losses'].append(avg_val_loss)

        # Save losses to a file
        with open(os.path.join('/opt/ml/output', 'losses.json'), 'w') as f:
            json.dump(losses_dict, f)
        
        # Check for overfitting
        if epoch > 0 and val_losses[-1] > val_losses[-2]:
            logger.info("Validation loss increased. Possible overfitting.")
            if epoch > 2 and all(val_losses[-i] > val_losses[-i-1] for i in range(1, 4)):
                logger.info("Validation loss increased for 3 consecutive epochs. Stopping training.")
                break

    
    return model, train_losses, val_losses

def net(num_classes):
    '''
    Function to initialize the model using a pretrained ResNet50.
    Freezes the parameters and modifies the final layer to match the number of classes.
    '''
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model

def create_data_loaders(data, batch_size):
    '''
    Function to create data loaders for the dataset.
    Applies necessary transformations to the images.
    '''
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = torchvision.datasets.ImageFolder(root=data, transform=transform)
    return utils.DataLoader(dataset, batch_size=batch_size, shuffle=True)

def main(args):
    '''
    Main function to initialize the model, create loss and optimizer, and start the training and testing process.
    '''
    model = net(args.num_classes)
    
    # Define the loss criterion
    loss_criterion = nn.CrossEntropyLoss()
    
    # Define the optimizer with weight decay
    optimizer = optim.SGD(model.fc.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=args.momentum) # type: ignore
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)

    # Initialize and register the debugging hook
    hook = get_hook(create_if_not_exists=True)
    hook.register_hook(model)
    hook.register_loss(loss_criterion)
    
    # Create data loaders for training and validation datasets
    try:
        train_data_dir = os.environ['SM_CHANNEL_TRAIN']
        valid_data_dir = os.environ['SM_CHANNEL_VALID']
        train_loader = create_data_loaders(train_data_dir, args.batch_size)
        valid_loader = create_data_loaders(valid_data_dir, args.batch_size)
    except KeyError as e:
        print(f"Environment variable {e} not set.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
    
    # Train the model
    model, train_losses, val_losses = train(model, train_loader, valid_loader, loss_criterion, optimizer, scheduler, args.epochs, hook)
    
    # Create data loader for the test dataset
    test_data_dir = os.environ['SM_CHANNEL_TEST']
    test_loader = create_data_loaders(test_data_dir, args.batch_size)
    
    # Test the model
    test(model, test_loader, hook)
    
    # Save the trained model
    torch.save(model.state_dict(), args.model_dir + '/model.pth') # type: ignore

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    '''
    Specify all the hyperparameters needed to train the model.
    '''
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--learning-rate', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--num-classes', type=int, default=5, metavar='N',
                        help='number of classes (default: 5)')
    parser.add_argument('--weight-decay', type=float, default=0.0001, metavar='WD',
                        help='weight decay (default: 0.0001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')
    
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    
    args = parser.parse_args()
    
    main(args)
