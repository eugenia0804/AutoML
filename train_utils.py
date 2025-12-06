import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from model import SmallCNN

from utils import make_loaders


def evaluate(model, loader, device):
    model.to(device)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            preds.extend(out.argmax(dim=1).cpu().numpy())
            trues.extend(yb.cpu().numpy())
    return f1_score(trues, preds, average="macro")


def train_and_eval(train_loader, valid_loader, activation, num_classes, device,
                   epochs=15, lr=1e-2, weight_decay=1e-4, dropout=0.25):
    model = SmallCNN(num_classes=num_classes, activation=activation, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.5)

    train_f1_history = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_f1 = evaluate(model, train_loader, device)
        val_f1 = evaluate(model, valid_loader, device)

        scheduler.step()

        # training F1
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                preds.extend(out.argmax(dim=1).cpu().numpy())
                trues.extend(yb.cpu().numpy())
        train_f1 = f1_score(trues, preds, average="macro")
        train_f1_history.append(train_f1)

    val_f1 = evaluate(model, valid_loader, device)
    return val_f1, model, train_f1_history


def evaluate_fitness(chrom, train_x, train_y, valid_x, valid_y, num_classes, device):
    batch = chrom["batch"]
    act = chrom["activation"]
    lr = chrom["lr"]
    epochs = chrom["epochs"]

    train_loader, valid_loader = make_loaders(train_x, train_y, valid_x, valid_y, batch)
    val_f1, _, _ = train_and_eval(train_loader, valid_loader, activation=act, num_classes=num_classes, device=device, epochs=epochs, lr=lr)
    return val_f1


# Implement caching mechanism for fitness evaluations
def chrom_to_key(chrom):
    return tuple((k, chrom[k]) for k in ["batch", "activation", "lr", "epochs"])

def evaluate_fitness_cached(chrom, train_x, train_y, valid_x, valid_y, num_classes, fitness_cache, device):
    key = chrom_to_key(chrom)
    if key in fitness_cache:
        return fitness_cache[key]
    val_f1 = evaluate_fitness(chrom, train_x, train_y, valid_x, valid_y, num_classes, device)
    fitness_cache[key] = val_f1
    return val_f1