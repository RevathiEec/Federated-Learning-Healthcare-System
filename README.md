# Federated Learning Healthcare System

A privacy-aware healthcare machine learning system that simulates federated learning across multiple hospitals using Python, PyTorch, Differential Privacy, and Flask.

## 📌 Overview

This project demonstrates how multiple hospitals can collaboratively train a machine learning model without directly sharing their local training data.

Instead of sending patient data to a central server, each hospital trains a local model and shares only its model parameters. The server then performs federated averaging to create an updated global model.

The project also applies a lightweight differential privacy mechanism to local model parameters before aggregation.

A Flask-based dashboard is provided to monitor the federated learning process and display model performance.

> **Note:** The current implementation uses a synthetic classification dataset for demonstration and educational purposes. It does not use real patient data.

---

## 🎯 Objectives

- Simulate collaborative machine learning across multiple hospitals.
- Keep hospital training data local.
- Implement federated averaging for global model creation.
- Apply noise to local model parameters as a basic privacy mechanism.
- Track model accuracy across multiple federated rounds.
- Provide a web-based monitoring dashboard using Flask.

---

## 🚀 Key Features

### 🏥 Multi-Hospital Simulation

The system simulates three independent hospitals.

Each hospital has its own local dataset and performs model training independently.

```text
Hospital 1 ──┐
Hospital 2 ──┼──> Local Training
Hospital 3 ──┘


### The project uses a federated learning workflow:

Global Model
     ↓
Local Training
     ↓
Privacy Protection
     ↓
Model Updates
     ↓
Federated Averaging
     ↓
Updated Global Model
     ↓
Next Round
