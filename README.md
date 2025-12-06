# AutoML

## Project Overview
This project explores and compares two distinct Automated Machine Learning (AutoML) techniques for hyperparameter tuning: **Genetic Algorithms (GA)** and **Bayesian Optimization (BO)**.

The goal was to optimize a small CNN model's hyperparameters (training Batch Size and Activation Function) to maximize the F1 Score on a testing EMNIST dataset. 

## Repository Structure
* `run_genetic.py`: Implementation of the Genetic Algorithm for hyperparameter tuning.
* `run_bayesian.py`: Implementation of Bayesian Optimization using `bayesian-optimization` package.
* `output/`: Contains training logs and resulting plots.

## Experiment Setup
Both algorithms were tested on a training dataset of 4,000 samples with the following fixed settings:
* **Validation Set:** 1,000 samples
* **Test Set:** 1,000 samples
* **Default Learning Rate:** 0.02

### 1. Genetic Algorithm (GA) Configuration
The GA mimics natural selection to evolve better hyperparameters over generations.
* **Population Size:** 15
* **Generations:** 30
* **Offspring Percentage:** 75%
* **Mutation Probability:** 25%
* **Training Epochs (during search):** 8

### 2. Bayesian Optimization (BO) Configuration
The BO builds a probabilistic model to select the next most promising hyperparameters.
* **Random Exploration Iterations:** 10
* **Optimization Iterations:** 20
* **Training Epochs (during search):** 8

## How to Run
1.  **Clone the repository and install dependencies:**
    ```bash
    git clone https://github.com/eugenia0804/AutoML.git
    cd AutoML
    pip install -r requirements.txt
    ```

2.  **Run the Genetic Algorithm Experiment:**
    Execute the GA script to evolve the hyperparameters over 30 generations.
    ```bash
    python run_genetic.py
    ```

3.  **Run the Bayesian Optimization Experiment:**
    Execute the BO script to perform 30 total iterations of search.
    ```bash
    python run_bayesian.py
    ```

## Experiment Results

Both methods successfully converged to high-performing models with **F1 scores > 0.97**.

| Feature | Genetic Algorithm (GA) | Bayesian Optimization (BO) |
| :--- | :--- | :--- |
| **Best Batch Size** | 34 | 36 (rounded) |
| **Best Activation** | ReLU | Tanh |
| **Test F1 Score** | **0.9718** | **0.9734** |

### Summarization
* **Similarity:** Both methods found nearly identical optimal batch sizes (34 vs 36), proving that both strategies are effective for this search space.
* **Activation Functions:** GA preferred `relu` while BO settled on `tanh`, though GA often selected `tanh` in earlier generations as well.
