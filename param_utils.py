import random

HP_CONFIG = {
    "batch": {"type": "int", "min": 16, "max": 1024},
    "activation": {"type": "choice", "values": ["relu", "sigmoid", "tanh"]},

    # fixed hyperparameters (not optimized)
    "lr": {"type": "fixed", "value": 0.02},
    "epochs": {"type": "fixed", "value": 8}
}

# Util function to sample a hyperparameter based on its configuration
def sample_hp(param_config):
    ptype = param_config["type"]

    if ptype == "fixed":
        return param_config["value"]

    elif ptype == "int":
        return random.randint(param_config["min"], param_config["max"])

    elif ptype == "choice":
        return random.choice(param_config["values"])

    else:
        raise ValueError(f"Unknown hyperparameter type: {ptype}")