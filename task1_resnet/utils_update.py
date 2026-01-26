import torch
from tqdm import tqdm

def train_one_epoch(model, loader, criterion, optimizer, DEVICE, accumulation_steps=4):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0
    
    # Initialize gradients once
    optimizer.zero_grad(set_to_none=True)

    for i, (inputs, labels) in enumerate(tqdm(loader, desc="Training")):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        # Forward pass with autocast for mixed precision
        with torch.autocast(device_type="mps", dtype=torch.float16):
            outputs = model(inputs)
            # Normalize loss because we sum gradients over accumulation_steps
            loss = criterion(outputs, labels) / accumulation_steps

        # Backprop
        loss.backward()

        # Update weights after accumulation_steps
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Metrics (Multiply loss back for reporting)
        running_loss += (loss.item() * accumulation_steps)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    # Handle any remaining gradients
    if (i + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    avg_loss = running_loss / len(loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def evaluate(model, loader, criterion, DEVICE):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        with torch.autocast(device_type="mps", dtype=torch.float16):
            for inputs, labels in tqdm(loader, desc="Validating"):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
    avg_loss = running_loss / len(loader)
    accuracy = 100. * correct / total
    print(f"Validation Acc: {accuracy:.2f}%   Loss: {avg_loss:.4f}")

    return avg_loss, accuracy