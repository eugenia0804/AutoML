import numpy as np
import torch
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from utils import load_data, make_loaders, make_test_loader
from train_utils import train_and_eval, evaluate


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


ACTIVATIONS = ["relu", "sigmoid", "tanh"]

def objective(batch_size, activation_idx):
    
    # Define hyperparameters
    batch_size = int(batch_size)
    activation_idx = int(round(activation_idx))
    activation = ACTIVATIONS[activation_idx]

    train_loader, val_loader = make_loaders(
        X_train, y_train, X_val, y_val,
        batch_size,  
        1024  
    )

    val_f1, _, _ = train_and_eval(
        train_loader,
        val_loader,
        activation=activation,
        num_classes=10,
        epochs=8,    
        lr=0.02,
        device=device
    )

    return val_f1


if __name__ == "__main__":

    WARMUP_ITER = 10
    OPTIMIZATION_ITER = 20
    FINAL_EPOCHS = 20

    X, y, X_test, y_test = load_data()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y
    )

    print("\nRunning Bayesian Optimization...")

    # Bounds for continuous optimizer
    pbounds = {
        "batch_size": (16, 1024),
        "activation_idx": (0, 2),
    }

    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        verbose=2,
        random_state=42,
        allow_duplicate_points=True
    )

    # BO settings
    optimizer.maximize(
        init_points=WARMUP_ITER,   # Random exploration iterations
        n_iter=OPTIMIZATION_ITER   # BO iterations
    )

    print("\nBO Optimization Complete:")
    print("Best result:")
    print(optimizer.max)

    # Extract best hyperparameters
    best_params = optimizer.max["params"]
    best_batch = int(best_params["batch_size"])
    best_activation = ACTIVATIONS[int(round(best_params["activation_idx"]))]

    print("\nBest hyperparams found by BO:")
    print(f"Batch size = {best_batch}")
    print(f"Activation = {best_activation}")

    # Final Training on Train+Val
    X_combined = np.concatenate([X_train, X_val])
    y_combined = np.concatenate([y_train, y_val])

    train_loader_final, dummy_valid_loader = make_loaders(
        X_combined,
        y_combined,
        X_combined,
        y_combined,
        train_batch_size=best_batch,
        valid_batch_size=1024
    )

    val_f1_final, final_model, f1_history = train_and_eval(
        train_loader_final,
        dummy_valid_loader,
        activation=best_activation,
        num_classes=10,
        epochs=FINAL_EPOCHS,
        lr=0.01,
        device=device
    )

    # Plot training F1 vs epochs
    plt.figure()
    plt.plot(range(1, len(f1_history)+1), f1_history, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training F1 (macro)")
    plt.title("Final Training (Bayesian Optimized Hyperparameters)")
    plt.title("Params: batch={}, act={}".format(best_batch, best_activation))
    plt.grid(True)
    plt.savefig("bo_optimal_f1.png")

    # Test evaluation
    test_loader = make_test_loader(X_test, y_test, batch_size=1024)
    test_f1 = evaluate(final_model, test_loader, device=device)

    print(f"\nFinal Test macro-F1 (BO): {test_f1:.4f}")
