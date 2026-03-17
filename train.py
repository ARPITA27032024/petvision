import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, WeightedRandomSampler  
from torchvision import datasets, transforms
from model import build_model


def get_dataloaders(data_dir, batch_size=32):

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(f"{data_dir}/train", train_transforms)
    val_data   = datasets.ImageFolder(f"{data_dir}/valid",   val_transforms)

    # ─────────────────────────────────────────────
    # ✅ WEIGHTED SAMPLER GOES HERE — right after
    #    train_data is created, before DataLoader
    # ─────────────────────────────────────────────
    print("Class mapping:", train_data.class_to_idx)  # always verify this

    class_counts   = [len([x for x in train_data.targets if x == i])
                      for i in range(len(train_data.classes))]
    print("Samples per class:", class_counts)         # see your imbalance

    weights        = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = weights[train_data.targets]
    sampler        = WeightedRandomSampler(sample_weights, len(sample_weights))
    # ─────────────────────────────────────────────

    # ✅ Use sampler= here, NOT shuffle=True
    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler)

    # Val loader stays exactly the same — no sampler needed
    val_loader   = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_data.classes


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")

    train_loader, val_loader, classes = get_dataloaders(r"C:\Users\arpit\OneDrive\Desktop\AI TRAINING\vision\dogy_dataset", batch_size=32)
    model     = build_model(num_classes=len(classes), device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_acc = 0.0

    for epoch in range(30):

        # ── Training ──
        model.train()
        train_loss, correct, total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += labels.size(0)

        # ── Validation ──
        model.eval()
        val_correct, val_total = 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs     = model(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total   += labels.size(0)

        val_acc = val_correct / val_total

        # ─────────────────────────────────────────────
        # ✅ THIS IS THE LOSS MONITORING OUTPUT
        #    Watch these numbers every epoch
        #    Loss should go DOWN, Acc should go UP
        # ─────────────────────────────────────────────
        print(f"Epoch {epoch+1:02d} | "
              f"Loss: {train_loss/len(train_loader):.3f} | "
              f"Train Acc: {correct/total:.2%} | "
              f"Val Acc: {val_acc:.2%}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  ✓ Saved new best model ({val_acc:.2%})")

        scheduler.step()

    print(f"\nTraining complete. Best Val Accuracy: {best_val_acc:.2%}")


if __name__ == "__main__":
    train()


