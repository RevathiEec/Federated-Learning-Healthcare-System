import json
import numpy as np
import torch
import torch.nn as nn

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from privacy import add_differential_privacy


class HeartDiseaseModel(nn.Module):

    def __init__(self, input_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


def create_hospital_data(seed, samples=300):

    X, y = make_classification(
        n_samples=samples,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        random_state=seed
    )

    X = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed
    )

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
    )


def train_local_model(model, X_train, y_train, epochs=10):

    criterion = nn.BCELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    model.train()

    for _ in range(epochs):

        predictions = model(X_train)

        loss = criterion(
            predictions,
            y_train
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def federated_average(models):

    global_state = {}

    for key in models[0].state_dict():

        global_state[key] = torch.stack(
            [
                model.state_dict()[key].float()
                for model in models
            ]
        ).mean(dim=0)

    return global_state


def evaluate(model, X_test, y_test):

    model.eval()

    with torch.no_grad():

        predictions = model(X_test)

        predicted_classes = (
            predictions >= 0.5
        ).float()

        accuracy = (
            predicted_classes.eq(y_test)
            .sum()
            .item()
            / len(y_test)
        )

    return accuracy


# -----------------------------
# Create hospitals
# -----------------------------

hospital_data = [
    create_hospital_data(1),
    create_hospital_data(2),
    create_hospital_data(3)
]


# -----------------------------
# Federated Learning
# -----------------------------

input_size = 10

global_model = HeartDiseaseModel(input_size)

round_accuracies = []

number_of_rounds = 10


for round_number in range(1, number_of_rounds + 1):

    print(
        f"\n========== Federated Round "
        f"{round_number} =========="
    )

    local_models = []

    for hospital_number, data in enumerate(
        hospital_data,
        start=1
    ):

        X_train, y_train, X_test, y_test = data

        local_model = HeartDiseaseModel(input_size)

        local_model.load_state_dict(
            global_model.state_dict()
        )

        train_local_model(
            local_model,
            X_train,
            y_train
        )

        add_differential_privacy(
            local_model,
            clip_value=1.0,
            noise_scale=0.01
        )

        local_models.append(local_model)

        print(
            f"Hospital {hospital_number} "
            f"completed local training "
            f"with privacy protection."
        )

    # Federated aggregation

    global_state = federated_average(
        local_models
    )

    global_model.load_state_dict(
        global_state
    )

    # Evaluate

    accuracies = []

    for hospital_number, data in enumerate(
        hospital_data,
        start=1
    ):

        X_train, y_train, X_test, y_test = data

        accuracy = evaluate(
            global_model,
            X_test,
            y_test
        )

        accuracies.append(accuracy)

        print(
            f"Hospital {hospital_number} "
            f"accuracy: "
            f"{accuracy * 100:.2f}%"
        )

    average_accuracy = np.mean(
        accuracies
    )

    round_accuracies.append(
        float(average_accuracy * 100)
    )

    print(
        f"Round {round_number} "
        f"Average Accuracy: "
        f"{average_accuracy * 100:.2f}%"
    )


# -----------------------------
# Save results for Flask
# -----------------------------

results = {
    "hospitals": 3,
    "rounds": number_of_rounds,
    "privacy": "Enabled",
    "accuracies": round_accuracies,
    "final_accuracy": round_accuracies[-1]
}


with open(
    "results.json",
    "w"
) as file:

    json.dump(
        results,
        file,
        indent=4
    )


print("\n==============================")
print("Federated Learning Completed")
print("==============================")

print(
    f"Final Average Accuracy: "
    f"{round_accuracies[-1]:.2f}%"
)

print(
    "\nResults saved to results.json"
)