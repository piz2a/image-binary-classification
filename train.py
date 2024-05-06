import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, SubsetRandomSampler
from binaryclassifier import BinaryClassifier


image_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_tag, dim=1)
    correct_results_sum = (y_pred_tags == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc


def train(model, device, train_loader, val_loader, criterion, optimizer, epoch):
    accuracy_stats = {'train': [], 'val': []}
    loss_stats = {'train': [], 'val': []}

    for e in tqdm(range(1, epoch+1)):
        # training
        train_epoch_loss = 0
        train_epoch_acc = 0

        model.train()
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()

            y_train_pred = model(X_train_batch).squeeze()

            train_loss = criterion(y_train_pred, y_train_batch)
            train_acc = binary_acc(y_train_pred, y_train_batch)

            train_loss.backward()
            optimizer.step()

            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()

        # validation
        with torch.no_grad():
            model.eval()
            val_epoch_loss = 0
            val_epoch_acc = 0
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                y_val_pred = model(X_val_batch).squeeze()
                y_val_pred = torch.unsqueeze(y_val_pred, 0)
                val_loss = criterion(y_val_pred, y_val_batch)
                val_acc = binary_acc(y_val_pred, y_val_batch)
                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()
        loss_stats['train'].append(train_epoch_loss/len(train_loader))
        loss_stats['val'].append(val_epoch_loss/len(val_loader))
        accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
        accuracy_stats['val'].append(val_epoch_acc/len(val_loader))
        print(f'Epoch {e}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(val_loader):.3f}')

    return accuracy_stats, loss_stats


def visualize(accuracy_stats, loss_stats, savefig_name: str = None):
    # Loss and Accuracy Visualization
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(loss_stats['train'], label='Train Loss')
    plt.plot(loss_stats['val'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(accuracy_stats['train'], label='Train Accuracy')
    plt.plot(accuracy_stats['val'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    if savefig_name is not None:
        plt.savefig(savefig_name, bbox_inches='tight')


if __name__ == '__main__':
    name = input('Enter dataset name: ')
    root = input('Enter root directory: ')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)
    new_dataset = datasets.ImageFolder(root=root, transform=image_transforms)

    new_dataset_size = len(new_dataset)
    new_dataset_size_indices = list(range(new_dataset_size))
    np.random.shuffle(new_dataset_size_indices)
    val_split_index = int(np.floor(0.2 * new_dataset_size))  # 80% train, 20% validation
    train_idx, val_idx = new_dataset_size_indices[val_split_index:], new_dataset_size_indices[:val_split_index]
    train_loader = DataLoader(dataset=new_dataset, shuffle=False, batch_size=8, sampler=SubsetRandomSampler(train_idx))
    val_loader = DataLoader(dataset=new_dataset, shuffle=False, batch_size=1, sampler=SubsetRandomSampler(val_idx))

    model = BinaryClassifier()
    model.to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    accuracy_stats, loss_stats = train(model, device, train_loader, val_loader, criterion, optimizer, epoch=20)

    if not os.path.isdir('output'):
        os.mkdir('output')
    if not os.path.isdir('output/model'):
        os.mkdir('output/model')
    if not os.path.isdir('output/train-stats'):
        os.mkdir('output/train-stats')
    visualize(accuracy_stats, loss_stats, savefig_name=f'output/train-stats/{name}-train-stats.png')
    torch.save(model, f'output/model/{name}.pt')

#%%
