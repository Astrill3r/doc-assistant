import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from src.vectorize import get_prepared_data
from src.model import TextClassifier

import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import json

def train_and_log(lr, epochs, batch_size=32):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("classic-doc-classifier")

    with mlflow.start_run():
        print(f"\n=== Запуск эксперимента: lr={lr}, epochs={epochs} ===")

        X, y = get_prepared_data()
        in_features = X.shape[1]
        num_classes = int(y.max()) + 1

        X_tr, X_te, y_tr, y_te = train_test_split(X.numpy(), y.numpy(), test_size=0.2, random_state=42)

        X_train_tensor = torch.tensor(X_tr, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_tr, dtype=torch.long)
        X_test_tensor = torch.tensor(X_te, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_te, dtype=torch.long)

        train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

        model = TextClassifier(in_features=in_features, num_classes=num_classes)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        history_train_loss = []
        history_test_loss = []

        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("epochs", epochs)
        mlflow.set_tag("model_type", "TF-IDF + PyTorch NN")
        mlflow.set_tag("author", "Astrill3r")

        for epoch in range(epochs):
            model.train()
            total_train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                loss = loss_fn(model(batch_X), batch_y)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            history_train_loss.append(avg_train_loss)

            # Валидация (проверяем точность и ошибку на тесте)
            model.eval()
            total_test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    logits = model(batch_X)
                    loss = loss_fn(logits, batch_y)
                    total_test_loss += loss.item()
                    _, predicted = torch.max(logits, dim=1)
                    correct += (predicted == batch_y).sum().item()
                    total += batch_y.size(0)
            
            avg_test_loss = total_test_loss / len(test_loader)
            history_test_loss.append(avg_test_loss)
            acc = correct / total
            
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("test_loss", avg_test_loss, step=epoch)
            mlflow.log_metric("test_accuracy", acc, step=epoch)
            
            print(f"Эпоха {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Acc: {acc:.4f}")

        # Сохраняем веса
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), 'models/best_text_classifier.pth')
        
        # 1. Сохраняем график (как в твоем изначальном коде)
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, epochs + 1), history_train_loss, label='Train Loss')
        plt.plot(range(1, epochs + 1), history_test_loss, label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (CrossEntropy)')
        plt.title(f'Loss curve (lr={lr})')
        plt.legend()
        fig = plt.gcf()
        mlflow.log_figure(fig, "artifacts/loss_curve.png")
        plt.close()

        # 2. Сохраняем config
        config = {"lr": lr, "epochs": epochs, "optimizer": "Adam"}
        with open("config.json", "w") as f:
            json.dump(config, f)
        mlflow.log_artifact("config.json")
        os.remove("config.json")

        # 3. Сохраняем модель
        mlflow.pytorch.log_model(model, "model", serialization_format="pickle")
        print("Модель обучена и сохранена!")

if __name__ == "__main__":
    # Запускаем два эксперимента для сравнения
    train_and_log(lr=0.01, epochs=20)
    train_and_log(lr=0.001, epochs=20)