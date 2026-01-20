import torch
from tqdm import tqdm

def train_one_epoch(model, loader, criterion, optimizer, DEVICE):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(loader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()  #Clear old gradients
        outputs = model(inputs)  #Forward pass
        loss = criterion(outputs, labels) #Compute loss
        loss.backward()  #Backpropagate
        optimizer.step()  #Update weights

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss/ len(loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def evaluate(model, loader, criterion, DEVICE):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(inputs)  #Forward pass
            loss = criterion(outputs, labels) #Compute loss

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    avg_loss = running_loss/ len(loader)
    accuracy = 100. * correct / total
    print(f"Validation Acc: {accuracy:.2f}%   Loss: {avg_loss:.4f}")

    return avg_loss, accuracy