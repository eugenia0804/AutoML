import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from utils import load_data, make_loaders, make_test_loader
from train_utils import train_and_eval, evaluate, evaluate_fitness_cached
from param_utils import HP_CONFIG, sample_hp


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def random_chrom():
    chrom = {k: sample_hp(v) for k, v in HP_CONFIG.items()}
    chrom["age"] = 0
    return chrom

def is_new_chrom(old_chrom, new_chrom):
    for key in HP_CONFIG.keys():
        if old_chrom[key] != new_chrom[key]:
            return True

def roulette(pop, fitness):
    total = sum(fitness)
    if total == 0:
        return random.choice(pop)
    pick = random.uniform(0, total)
    curr = 0.0
    for ind, fit in zip(pop, fitness):
        curr += fit
        if curr >= pick:
            return ind
    return pop[-1]

def crossover(a, b):
    keys = ["batch", "activation", "lr", "epochs"]
    ga, gb = a.copy(), b.copy()
    point = random.randint(1, len(keys) - 1)
    for k in keys[point:]:
        ga[k], gb[k] = gb[k], ga[k]

    if is_new_chrom(a, ga):
        ga["age"] = 0
    else:
        ga["age"] = a["age"]
    if is_new_chrom(b, gb): 
        gb["age"] = 0
    else:
        gb["age"] = b["age"]
    return ga, gb

def mutate(chrom, p=0.25):
    new_chrom = chrom.copy()
    for k, cfg in HP_CONFIG.items():
        if random.random() < p and cfg["type"] != "fixed":
            new_chrom[k] = sample_hp(cfg)

    if is_new_chrom(chrom, new_chrom):
        new_chrom["age"] = 0
    else:
        new_chrom["age"] = chrom["age"]
    return new_chrom

def update_population(population, fitness, pop_size,
                      train_x, train_y, valid_x, valid_y, n_classes,
                      roulette, crossover, mutate, evaluate_fitness):

    # Define number of offspring
    n_offspring = int(pop_size * 0.75)

    # Precompute fitness for roulette
    adj_fitness = [max(f, 0.0) for f in fitness]

    # Generate offspring
    offspring = []
    while len(offspring) < n_offspring:
        # Select parents using roulette rule
        p1 = roulette(population, adj_fitness)
        p2 = roulette(population, adj_fitness)
        # If same parent, reselect
        while p1 == p2:
            p2 = roulette(population, adj_fitness)

        # Crossover and mutate
        c1, c2 = crossover(p1, p2)
        c1 = mutate(c1)
        c2 = mutate(c2)

        # Add to offspring
        offspring.append(c1)
        if len(offspring) < n_offspring:
            offspring.append(c2)

    # Combine old population and offspring
    combined = population + offspring

    # Evaluate fitness of combined population
    combined_fitness = [
        evaluate_fitness_cached(ind, train_x, train_y, valid_x, valid_y, n_classes, fitness_cache, device)
        for ind in combined
    ]

    # Sort by age (oldest first), then by fitness (lowest first)
    combined_sorted_idx = sorted(
        range(len(combined)),
        key=lambda i: (combined[i]["age"], -combined_fitness[i]) 
    )
    # Select top individuals to form new population
    survivors_idx = combined_sorted_idx[:pop_size]
    new_population = [combined[i] for i in survivors_idx]

    # Age all chromosomes in the new population
    for ind in new_population:
        ind["age"] += 1

    return new_population


def genetic_algorithm(train_x, train_y, valid_x, valid_y, fitness_cache,
                      pop_size, generations):
    
    # Initialize population
    population = [random_chrom() for _ in range(pop_size)]
    best_history, avg_history = [], []
    n_classes = len(np.unique(train_y))


    for g in range(generations):

        print(f"\nAll ages at start of Gen {g+1}: {[ind['age'] for ind in population]}")

        fitness = []
        for ind in tqdm(population, desc=f"Evaluating Gen {g+1}"):
            f = evaluate_fitness_cached(ind, train_x, train_y, valid_x, valid_y, num_classes=n_classes, 
                                        fitness_cache=fitness_cache, device=device)
            fitness.append(f)
        best = max(fitness)
        avg = float(np.mean(fitness))
        best_history.append(best)
        avg_history.append(avg)
        print(f"Gen {g+1} | best F1={best:.4f} | avg F1={avg:.4f}")

        # Get top 5 individuals
        top_indices = np.argsort(fitness)[-5:][::-1]
        print("Top 5 individuals this generation:")
        for rank, idx in enumerate(top_indices, start=1):
            chrom = population[idx]
            print(f"{rank}: batch={chrom['batch']}, act={chrom['activation']}, "
                f"lr={chrom['lr']:.4f}, epochs={chrom['epochs']}, age={chrom['age']}, "
                f"F1={fitness[idx]:.4f}")
        
        # Update population for the new generation
        population = update_population(population, fitness, pop_size,
            train_x, train_y, valid_x, valid_y, n_classes,
            roulette, crossover, mutate, evaluate_fitness_cached
        )

    # Final evaluation to get the best individual
    fitness = [evaluate_fitness_cached(ind, train_x, train_y, valid_x, valid_y, num_classes=n_classes, 
                                       fitness_cache=fitness_cache, device=device)
            for ind in population]
    best_idx = int(np.argmax(fitness))
    best = population[best_idx]
    return best, best_history, avg_history



if __name__ == "__main__":

    fitness_cache = {}

    POPULATION_SIZE = 15
    GENERATIONS = 30
    FINAL_EPOCHS = 20

    X, y, X_test, y_test = load_data()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

    print("Running GA...")
    print(f"Population size: {POPULATION_SIZE}, Generations: {GENERATIONS}")

    best, best_hist, avg_hist = genetic_algorithm(
        X_train, y_train, X_val, y_val,
        pop_size=POPULATION_SIZE,
        generations=GENERATIONS,
        fitness_cache=fitness_cache
    )

    print("\nBest hyperparameters found by GA:")
    print(best)

    # Final Training on Train+Val
    X_combined = np.concatenate([X_train, X_val], axis=0)
    y_combined = np.concatenate([y_train, y_val], axis=0)
    batch_final = int(best["batch"])
    train_loader_final, valid_loader_dummy = make_loaders(
        X_combined, y_combined, X_combined, y_combined, train_batch_size=batch_final, valid_batch_size=1024)

    val_f1_final, final_model, train_f1_hist = train_and_eval(
        train_loader_final, valid_loader_dummy,
        activation=best["activation"],
        num_classes=10,
        epochs=FINAL_EPOCHS,
        lr=best["lr"],
        device=device
    )

    print(f"\nFinal training completed.")

    # Plot GA progress
    plt.figure()
    plt.plot(best_hist, label="Best")
    plt.plot(avg_hist, label="Average")
    plt.xlabel("Generation")
    plt.ylabel("Validation F1")
    plt.title("GA progress")
    plt.legend()
    plt.savefig("ga_progress.png")

    # Plot training F1 vs epochs
    plt.figure()
    plt.plot(range(1, len(train_f1_hist)+1), train_f1_hist, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Training F1 (macro)")
    plt.title("Training F1 vs Epochs (final training)")
    plt.title("Params: batch={}, act={}".format(best['batch'], best["activation"]))
    plt.grid(True)
    plt.savefig("ga_optimal_f1.png")

    # Test evaluation
    test_loader = make_test_loader(X_test, y_test, batch_size=1024)
    test_f1 = evaluate(final_model, test_loader, device=device)
    print(f"Final Test macro-F1: {test_f1:.4f}")