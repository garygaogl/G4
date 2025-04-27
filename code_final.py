
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load dataset from the path
df = pd.read_csv('C:\\Users\\HP\\Desktop\\Dataset_ATS_v2.csv')

# 2. Data preprocessing
target_column = 'Churn'  
if df[target_column].dtype == 'object':
    df[target_column] = LabelEncoder().fit_transform(df[target_column])

X = df.drop(columns=[target_column])
y = df[target_column]

# Encoding categorical variables
X = pd.get_dummies(X)

# Feature Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dataset Partitioning
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# Define MLPClassifier model
clf = MLPClassifier(solver='adam', hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)

# Print Results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Accuracy: 0.7565649396735273
# Classification Report:
#                precision    recall  f1-score   support

#            0       0.81      0.88      0.84      1042
#            1       0.54      0.41      0.47       367

#     accuracy                           0.76      1409
#    macro avg       0.68      0.65      0.66      1409
# weighted avg       0.74      0.76      0.75      1409

# Confusion Matrix:
#  [[914 128]
#  [215 152]]

import joblib  # Import joblib to save the model
# joblib.dump(clf, r'C:\Users\HP\Desktop\result\models\trained_model.pkl')  # Save the trained model

import matplotlib.pyplot as plt
import seaborn as sns
# Plotting the training loss curve
plt.figure(figsize=(10, 6))
plt.plot(clf.loss_curve_)
plt.title('Training Loss Curve')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

# Plotting the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


#%%


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt

# 1. Load dataset from the path
df = pd.read_csv('C:\\Users\\HP\\Desktop\\Dataset_ATS_v2.csv')

# 2. Data preprocessing
target_column = 'Churn'
if df[target_column].dtype == 'object':
    df[target_column] = LabelEncoder().fit_transform(df[target_column])

X = df.drop(columns=[target_column])
y = df[target_column]

# Encoding categorical variables
X = pd.get_dummies(X)

# Feature Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dataset Partitioning
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# Convert to PyTorch Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# # Encapsulated as DataLoader
# train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# Create training and validation sets
full_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)


# 3. Defining the ANN model
class ANNModel(nn.Module):
    def __init__(self, input_dim):
        super(ANNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

#%%
# Initialize the model
model = ANNModel(X_train.shape[1])

# 4. Defining loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 5. Train the model
epochs = 200
train_loss_list = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_loss_list.append(avg_loss)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

# Epoch [1/200], Loss: 0.4664
# Epoch [2/200], Loss: 0.4422
# Epoch [3/200], Loss: 0.4404
# Epoch [4/200], Loss: 0.4376
# Epoch [5/200], Loss: 0.4354
# Epoch [6/200], Loss: 0.4322
# Epoch [7/200], Loss: 0.4291
# Epoch [8/200], Loss: 0.4306
# Epoch [9/200], Loss: 0.4270
# Epoch [10/200], Loss: 0.4282
# ...
# Epoch [190/200], Loss: 0.3366
# Epoch [191/200], Loss: 0.3432
# Epoch [192/200], Loss: 0.3342
# Epoch [193/200], Loss: 0.3390
# Epoch [194/200], Loss: 0.3461
# Epoch [195/200], Loss: 0.3390
# Epoch [196/200], Loss: 0.3420
# Epoch [197/200], Loss: 0.3371
# Epoch [198/200], Loss: 0.3412
# Epoch [199/200], Loss: 0.3343
# Epoch [200/200], Loss: 0.3369

#%%
# 6. Model Evaluation
model.eval()
with torch.no_grad():
    y_pred_probs = model(X_test_tensor)
    y_pred = (y_pred_probs > 0.5).float()

# Convert to numpy calculate metrics
y_pred_np = y_pred.numpy()
y_test_np = y_test_tensor.numpy()

print("\n=== Accuracy Score ===")
print(accuracy_score(y_test_np, y_pred_np))
print("\n=== Classification Report ===")
print(classification_report(y_test_np, y_pred_np))
print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test_np, y_pred_np))

# === Accuracy Score ===
# 0.759403832505323

# === Classification Report ===
#               precision    recall  f1-score   support

#          0.0       0.81      0.89      0.85      1042
#          1.0       0.55      0.39      0.46       367

#     accuracy                           0.76      1409
#    macro avg       0.68      0.64      0.65      1409
# weighted avg       0.74      0.76      0.74      1409


# === Confusion Matrix ===
# [[926 116]
#  [223 144]]


# 7. Visualizing the loss curve
plt.plot(train_loss_list)
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# 8. Save the model
# torch.save(model.state_dict(), "C:\\Users\\HP\\Desktop\\ANN_file\\Predictive_Modeling\\models\\ann_churn_model1.pth")


#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt

# ========== Load dataset ==========
df = pd.read_csv('C:\\Users\\HP\\Desktop\\Dataset_ATS_v2.csv')

target_column = 'Churn'
if df[target_column].dtype == 'object':
    df[target_column] = LabelEncoder().fit_transform(df[target_column])

X = pd.get_dummies(df.drop(columns=[target_column]))
y = df[target_column]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Create training and validation sets
full_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# ========== Define the Model ==========
class ANNModel(nn.Module):
    def __init__(self, input_dim):
        super(ANNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

model = ANNModel(X_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Dynamic learning rate regulator (lower lr when validation set loss does not improve)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                  patience=5, verbose=True, min_lr=1e-5)

# ========== Train the model ==========
epochs = 200
train_loss_list = []
val_loss_list = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    train_loss_list.append(train_loss)

    # Verification Phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            val_outputs = model(val_inputs)
            val_loss += criterion(val_outputs, val_labels).item()

    val_loss /= len(val_loader)
    val_loss_list.append(val_loss)

    scheduler.step(val_loss)  # Adjusting the learning rate

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - LR: {current_lr:.6f}")

# Epoch [1/200] - Train Loss: 0.4671 - Val Loss: 0.4658 - LR: 0.010000
# Epoch [2/200] - Train Loss: 0.4520 - Val Loss: 0.4524 - LR: 0.010000
# Epoch [3/200] - Train Loss: 0.4488 - Val Loss: 0.4641 - LR: 0.010000
# Epoch [4/200] - Train Loss: 0.4480 - Val Loss: 0.4470 - LR: 0.010000
# Epoch [5/200] - Train Loss: 0.4415 - Val Loss: 0.4592 - LR: 0.010000
# Epoch [6/200] - Train Loss: 0.4437 - Val Loss: 0.4395 - LR: 0.010000
# Epoch [7/200] - Train Loss: 0.4392 - Val Loss: 0.4487 - LR: 0.010000
# Epoch [8/200] - Train Loss: 0.4395 - Val Loss: 0.4570 - LR: 0.010000
# Epoch [9/200] - Train Loss: 0.4381 - Val Loss: 0.4520 - LR: 0.010000
# Epoch [10/200] - Train Loss: 0.4413 - Val Loss: 0.4415 - LR: 0.010000
# ...
# Epoch [190/200] - Train Loss: 0.3779 - Val Loss: 0.4547 - LR: 0.000010
# Epoch [191/200] - Train Loss: 0.3776 - Val Loss: 0.4546 - LR: 0.000010
# Epoch [192/200] - Train Loss: 0.3776 - Val Loss: 0.4547 - LR: 0.000010
# Epoch [193/200] - Train Loss: 0.3777 - Val Loss: 0.4547 - LR: 0.000010
# Epoch [194/200] - Train Loss: 0.3775 - Val Loss: 0.4547 - LR: 0.000010
# Epoch [195/200] - Train Loss: 0.3775 - Val Loss: 0.4547 - LR: 0.000010
# Epoch [196/200] - Train Loss: 0.3774 - Val Loss: 0.4547 - LR: 0.000010
# Epoch [197/200] - Train Loss: 0.3775 - Val Loss: 0.4547 - LR: 0.000010
# Epoch [198/200] - Train Loss: 0.3776 - Val Loss: 0.4547 - LR: 0.000010
# Epoch [199/200] - Train Loss: 0.3776 - Val Loss: 0.4547 - LR: 0.000010
# Epoch [200/200] - Train Loss: 0.3774 - Val Loss: 0.4547 - LR: 0.000010


#%%
# ========== Evaluate the Model ==========
model.eval()
with torch.no_grad():
    y_pred_probs = model(X_test_tensor)
    y_pred = (y_pred_probs > 0.5).float()

y_pred_np = y_pred.numpy()
y_test_np = y_test_tensor.numpy()

print("\n=== Accuracy Score ===")
print(accuracy_score(y_test_np, y_pred_np))
print("\n=== Classification Report ===")
print(classification_report(y_test_np, y_pred_np))
print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test_np, y_pred_np))

# === Accuracy Score ===
# 0.7608232789212207

# === Classification Report ===
#               precision    recall  f1-score   support

#          0.0       0.81      0.89      0.85      1042
#          1.0       0.56      0.39      0.46       367

#     accuracy                           0.76      1409
#    macro avg       0.68      0.64      0.65      1409
# weighted avg       0.74      0.76      0.75      1409


# === Confusion Matrix ===
# [[928 114]
#  [223 144]]


# ========== Visualization ==========
plt.plot(train_loss_list, label='Train Loss')
plt.plot(val_loss_list, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.grid()
plt.show()

# ========== Save the model ==========
# torch.save(model.state_dict(), "C:\\Users\\HP\\Desktop\\result\\models\\ann_churn_model_optimized.pth")




#%%
# Linear ➜ ReLU ➜ Linear ➜ ReLU ➜ Sigmoid
# Linear ➜ BatchNorm ➜ ReLU ➜ Dropout ➜ ...

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
import os

# ========== Load dataset ==========
df = pd.read_csv('C:\\Users\\HP\\Desktop\\Dataset_ATS_v2.csv')

target_column = 'Churn'
if df[target_column].dtype == 'object':
    df[target_column] = LabelEncoder().fit_transform(df[target_column])

X = pd.get_dummies(df.drop(columns=[target_column]))
y = df[target_column]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Split validation set
full_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# ========== Define the optimized model ==========
class ANNModel(nn.Module):
    def __init__(self, input_dim):
        super(ANNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(64, 1)

        self.activation = nn.ReLU()
        self.output = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.drop1(self.activation(self.bn1(self.fc1(x))))
        x = self.drop2(self.activation(self.bn2(self.fc2(x))))
        x = self.output(self.fc3(x))
        return x

model = ANNModel(X_train.shape[1])

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True, factor=0.5, min_lr=1e-5)

# ========== Early Stopping ==========
best_val_loss = float('inf')
early_stop_patience = 15
patience_counter = 0

# ========== Training ==========
train_loss_list = []
val_loss_list = []
epochs = 200

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)
    train_loss_list.append(train_loss)

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for xb, yb in val_loader:
            preds = model(xb)
            loss = criterion(preds, yb)
            val_loss += loss.item()

        val_loss /= len(val_loader)
        val_loss_list.append(val_loss)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # torch.save(model.state_dict(), "‪C:\\Users\\HP\\Desktop\\result\\models\\best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print("Early stopping triggered.")
                break
# Epoch 1/200 | Train Loss: 0.5054 | Val Loss: 0.4421 | LR: 0.010000
# Epoch 2/200 | Train Loss: 0.4745 | Val Loss: 0.4382 | LR: 0.010000
# Epoch 3/200 | Train Loss: 0.4649 | Val Loss: 0.4370 | LR: 0.010000
# Epoch 4/200 | Train Loss: 0.4647 | Val Loss: 0.4330 | LR: 0.010000
# Epoch 5/200 | Train Loss: 0.4637 | Val Loss: 0.4318 | LR: 0.010000
# Epoch 6/200 | Train Loss: 0.4591 | Val Loss: 0.4342 | LR: 0.010000
# Epoch 7/200 | Train Loss: 0.4650 | Val Loss: 0.4384 | LR: 0.010000
# Epoch 8/200 | Train Loss: 0.4567 | Val Loss: 0.4382 | LR: 0.010000
# Epoch 9/200 | Train Loss: 0.4554 | Val Loss: 0.4315 | LR: 0.010000
# Epoch 10/200 | Train Loss: 0.4595 | Val Loss: 0.4341 | LR: 0.010000
# Epoch 11/200 | Train Loss: 0.4570 | Val Loss: 0.4339 | LR: 0.010000
# Epoch 12/200 | Train Loss: 0.4561 | Val Loss: 0.4282 | LR: 0.010000
# Epoch 13/200 | Train Loss: 0.4581 | Val Loss: 0.4331 | LR: 0.010000
# Epoch 14/200 | Train Loss: 0.4580 | Val Loss: 0.4336 | LR: 0.010000
# Epoch 15/200 | Train Loss: 0.4516 | Val Loss: 0.4300 | LR: 0.010000
# Epoch 16/200 | Train Loss: 0.4573 | Val Loss: 0.4305 | LR: 0.010000
# Epoch 17/200 | Train Loss: 0.4558 | Val Loss: 0.4370 | LR: 0.010000
# Epoch 18/200 | Train Loss: 0.4546 | Val Loss: 0.4309 | LR: 0.005000
# Epoch 19/200 | Train Loss: 0.4513 | Val Loss: 0.4277 | LR: 0.005000
# Epoch 20/200 | Train Loss: 0.4492 | Val Loss: 0.4400 | LR: 0.005000
# Epoch 21/200 | Train Loss: 0.4488 | Val Loss: 0.4296 | LR: 0.005000
# Epoch 22/200 | Train Loss: 0.4475 | Val Loss: 0.4298 | LR: 0.005000
# Epoch 23/200 | Train Loss: 0.4463 | Val Loss: 0.4307 | LR: 0.005000
# Epoch 24/200 | Train Loss: 0.4514 | Val Loss: 0.4326 | LR: 0.005000
# Epoch 25/200 | Train Loss: 0.4483 | Val Loss: 0.4302 | LR: 0.002500
# Epoch 26/200 | Train Loss: 0.4467 | Val Loss: 0.4321 | LR: 0.002500
# Epoch 27/200 | Train Loss: 0.4464 | Val Loss: 0.4323 | LR: 0.002500
# Epoch 28/200 | Train Loss: 0.4473 | Val Loss: 0.4279 | LR: 0.002500
# Epoch 29/200 | Train Loss: 0.4440 | Val Loss: 0.4289 | LR: 0.002500
# Epoch 30/200 | Train Loss: 0.4423 | Val Loss: 0.4293 | LR: 0.002500
# Epoch 31/200 | Train Loss: 0.4450 | Val Loss: 0.4297 | LR: 0.001250
# Epoch 32/200 | Train Loss: 0.4392 | Val Loss: 0.4291 | LR: 0.001250
# Epoch 33/200 | Train Loss: 0.4393 | Val Loss: 0.4290 | LR: 0.001250
# Epoch 34/200 | Train Loss: 0.4408 | Val Loss: 0.4299 | LR: 0.001250
# Early stopping triggered.

#%%
# ========== Evaluation on test set ==========
model.load_state_dict(torch.load("C:\\Users\\HP\\Desktop\\result\\models\\best_model.pth"))
model.eval()
with torch.no_grad():
    test_probs = model(X_test_tensor).numpy()
    test_preds = (test_probs > 0.5).astype(int)

print("\n Accuracy:", accuracy_score(y_test, test_preds))
print("AUC-ROC:", roc_auc_score(y_test, test_probs))
print("Classification Report:\n", classification_report(y_test, test_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, test_preds))
# Accuracy: 0.7665010645848119
# AUC-ROC: 0.7942962339244902
# Classification Report:
#                precision    recall  f1-score   support

#            0       0.81      0.90      0.85      1042
#            1       0.58      0.38      0.46       367

#     accuracy                           0.77      1409
#    macro avg       0.69      0.64      0.66      1409
# weighted avg       0.75      0.77      0.75      1409

# Confusion Matrix:
#  [[939 103]
#  [226 141]]

# ========== ROC curve ==========
fpr, tpr, _ = roc_curve(y_test, test_probs)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, test_probs):.3f}")
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()



#%%

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import os

# ========== 1. Data loading and preprocessing ==========
df = pd.read_csv('C:\\Users\\HP\\Desktop\\Dataset_ATS_v2.csv')
target_column = 'Churn'

# Encoded Label
if df[target_column].dtype == 'object':
    df[target_column] = LabelEncoder().fit_transform(df[target_column])

X = pd.get_dummies(df.drop(columns=[target_column]))
y = df[target_column]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Split validation set
full_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# ========== 2. Define the model ==========
class ANN(nn.Module):
    def __init__(self, input_dim):
        super(ANN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# ========== 3. Training function ==========
def train_model(model, criterion, optimizer, scheduler, n_epochs=200, patience=10):
    best_loss = np.inf
    trigger_times = 0
    train_losses, val_losses, lrs = [], [], []

    for epoch in range(1, n_epochs+1):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                output = model(xb)
                loss = criterion(output, yb)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        lrs.append(optimizer.param_groups[0]['lr'])

        print(f"Epoch {epoch}/{n_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            trigger_times = 0
            # torch.save(model.state_dict(), '‪C:\\Users\\HP\\Desktop\\result\\models\\best_model1.pt')
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping triggered.")
                break

    return train_losses, val_losses, lrs

# ========== 4. Initialization ==========
model = ANN(X_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

train_losses, val_losses, lrs = train_model(model, criterion, optimizer, scheduler)

# Epoch 1/200 | Train Loss: 0.4943 | Val Loss: 0.4530 | LR: 0.010000
# Epoch 2/200 | Train Loss: 0.4636 | Val Loss: 0.4529 | LR: 0.010000
# Epoch 3/200 | Train Loss: 0.4661 | Val Loss: 0.4460 | LR: 0.010000
# Epoch 4/200 | Train Loss: 0.4593 | Val Loss: 0.4552 | LR: 0.010000
# Epoch 5/200 | Train Loss: 0.4550 | Val Loss: 0.4441 | LR: 0.010000
# Epoch 6/200 | Train Loss: 0.4570 | Val Loss: 0.4446 | LR: 0.010000
# Epoch 7/200 | Train Loss: 0.4530 | Val Loss: 0.4619 | LR: 0.010000
# Epoch 8/200 | Train Loss: 0.4620 | Val Loss: 0.4407 | LR: 0.010000
# Epoch 9/200 | Train Loss: 0.4486 | Val Loss: 0.4519 | LR: 0.010000
# Epoch 10/200 | Train Loss: 0.4515 | Val Loss: 0.4717 | LR: 0.010000
# Epoch 11/200 | Train Loss: 0.4588 | Val Loss: 0.4429 | LR: 0.010000
# Epoch 12/200 | Train Loss: 0.4525 | Val Loss: 0.4483 | LR: 0.010000
# Epoch 13/200 | Train Loss: 0.4403 | Val Loss: 0.4543 | LR: 0.005000
# Epoch 14/200 | Train Loss: 0.4369 | Val Loss: 0.4424 | LR: 0.005000
# Epoch 15/200 | Train Loss: 0.4322 | Val Loss: 0.4466 | LR: 0.005000
# Epoch 16/200 | Train Loss: 0.4389 | Val Loss: 0.4436 | LR: 0.005000
# Epoch 17/200 | Train Loss: 0.4314 | Val Loss: 0.4455 | LR: 0.002500
# Epoch 18/200 | Train Loss: 0.4303 | Val Loss: 0.4438 | LR: 0.002500
# Early stopping triggered.

# ========== 5. Model Evaluation ==========
model.load_state_dict(torch.load('C:\\Users\\HP\\Desktop\\result\\models\\best_model1.pt'))
model.eval()
with torch.no_grad():
    y_pred_prob = model(X_test_tensor).numpy()
    y_pred = (y_pred_prob > 0.5).astype(int)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_prob)}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# Accuracy: 0.7686302342086586
# AUC-ROC: 0.7926187325777821
# Classification Report:
#               precision    recall  f1-score   support

#            0       0.80      0.91      0.85      1042
#            1       0.59      0.36      0.45       367

#     accuracy                           0.77      1409
#    macro avg       0.70      0.64      0.65      1409
# weighted avg       0.75      0.77      0.75      1409

# Confusion Matrix:
# [[951  91]
#  [235 132]]
# ========== 6. Visualization ==========
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over Epochs')
plt.show()

plt.plot(lrs)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.show()

# %%

# CrossEntropyLoss + LongTensor label: required for classification tasks
# Early Stopping + ReduceLROnPlateau： prevent overfitting
# ROC curve + AUC + confusion matrix + report: complete evaluation
# Normalization + OneHot encoding + LabelEncoder: good data preparation process

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, roc_curve

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt

# ========== 1. Data preprocessing ==========
df = pd.read_csv('C:\\Users\\HP\\Desktop\\Dataset_ATS_v2.csv')
target_column = 'Churn'

if df[target_column].dtype == 'object':
    df[target_column] = LabelEncoder().fit_transform(df[target_column])

X = pd.get_dummies(df.drop(columns=[target_column]))
y = df[target_column]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

full_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=64)

# ========== 2. Define the model ==========
class ChurnClassifier(nn.Module):
    def __init__(self, input_dim):
        super(ChurnClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.model(x)

# ========== 3. Training function ==========
def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, epochs=100, patience=10):
    best_loss = np.inf
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss = evaluate_loss(model, val_loader, criterion)
        print(f"Epoch {epoch}/{epochs} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | LR: {scheduler.optimizer.param_groups[0]['lr']:.6f}")
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            # torch.save(model.state_dict(), '‪C:\\Users\\HP\\Desktop\\result\\models\\best_model2.pt')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

def evaluate_loss(model, loader, criterion):
    model.eval()
    loss_total = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss_total += loss.item()
    return loss_total / len(loader)

# ========== 4. Evaluation Function ==========
def evaluate_model(model, test_loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    print(f"Accuracy: {accuracy_score(all_labels, all_preds)}")
    print(f"AUC-ROC: {roc_auc_score(all_labels, all_probs)}")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    # Plotting ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure()
    plt.plot(fpr, tpr, label='AUC = %.4f' % roc_auc_score(all_labels, all_probs))
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.show()

# ========== 5. Start training ==========
input_dim = X_train.shape[1]
model = ChurnClassifier(input_dim)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, epochs=100, patience=10)

# Epoch 1/100 | Train Loss: 0.5122 | Val Loss: 0.4496 | LR: 0.010000
# Epoch 2/100 | Train Loss: 0.4634 | Val Loss: 0.4397 | LR: 0.010000
# Epoch 3/100 | Train Loss: 0.4579 | Val Loss: 0.4471 | LR: 0.010000
# Epoch 4/100 | Train Loss: 0.4591 | Val Loss: 0.4451 | LR: 0.010000
# Epoch 5/100 | Train Loss: 0.4578 | Val Loss: 0.4445 | LR: 0.010000
# Epoch 6/100 | Train Loss: 0.4550 | Val Loss: 0.4432 | LR: 0.010000
# Epoch 7/100 | Train Loss: 0.4465 | Val Loss: 0.4428 | LR: 0.005000
# Epoch 8/100 | Train Loss: 0.4470 | Val Loss: 0.4451 | LR: 0.005000
# Epoch 9/100 | Train Loss: 0.4443 | Val Loss: 0.4427 | LR: 0.005000
# Epoch 10/100 | Train Loss: 0.4456 | Val Loss: 0.4406 | LR: 0.005000
# Epoch 11/100 | Train Loss: 0.4402 | Val Loss: 0.4402 | LR: 0.002500
# Epoch 12/100 | Train Loss: 0.4378 | Val Loss: 0.4385 | LR: 0.002500
# Epoch 13/100 | Train Loss: 0.4427 | Val Loss: 0.4398 | LR: 0.002500
# Epoch 14/100 | Train Loss: 0.4461 | Val Loss: 0.4386 | LR: 0.002500
# Epoch 15/100 | Train Loss: 0.4336 | Val Loss: 0.4382 | LR: 0.002500
# Epoch 16/100 | Train Loss: 0.4402 | Val Loss: 0.4394 | LR: 0.002500
# Epoch 17/100 | Train Loss: 0.4437 | Val Loss: 0.4380 | LR: 0.002500
# Epoch 18/100 | Train Loss: 0.4394 | Val Loss: 0.4392 | LR: 0.002500
# Epoch 19/100 | Train Loss: 0.4343 | Val Loss: 0.4407 | LR: 0.002500
# Epoch 20/100 | Train Loss: 0.4356 | Val Loss: 0.4377 | LR: 0.002500
# Epoch 21/100 | Train Loss: 0.4388 | Val Loss: 0.4415 | LR: 0.002500
# Epoch 22/100 | Train Loss: 0.4368 | Val Loss: 0.4398 | LR: 0.002500
# Epoch 23/100 | Train Loss: 0.4360 | Val Loss: 0.4409 | LR: 0.002500
# Epoch 24/100 | Train Loss: 0.4330 | Val Loss: 0.4375 | LR: 0.002500
# Epoch 25/100 | Train Loss: 0.4344 | Val Loss: 0.4396 | LR: 0.002500
# Epoch 26/100 | Train Loss: 0.4407 | Val Loss: 0.4382 | LR: 0.002500
# Epoch 27/100 | Train Loss: 0.4329 | Val Loss: 0.4391 | LR: 0.002500
# Epoch 28/100 | Train Loss: 0.4374 | Val Loss: 0.4418 | LR: 0.002500
# Epoch 29/100 | Train Loss: 0.4323 | Val Loss: 0.4377 | LR: 0.001250
# Epoch 30/100 | Train Loss: 0.4316 | Val Loss: 0.4380 | LR: 0.001250
# Epoch 31/100 | Train Loss: 0.4327 | Val Loss: 0.4387 | LR: 0.001250
# Epoch 32/100 | Train Loss: 0.4312 | Val Loss: 0.4388 | LR: 0.001250
# Epoch 33/100 | Train Loss: 0.4306 | Val Loss: 0.4380 | LR: 0.000625
# Epoch 34/100 | Train Loss: 0.4256 | Val Loss: 0.4383 | LR: 0.000625
# Early stopping triggered.

# ========== 6. Load the best model and evaluate ==========
model.load_state_dict(torch.load('‪C:\\Users\\HP\\Desktop\\result\\models\\best_model2.pt'))
evaluate_model(model, test_loader)
# Accuracy: 0.7835344215755855
# AUC-ROC: 0.8057893513136479
# Classification Report:
#               precision    recall  f1-score   support

#            0       0.83      0.89      0.86      1035
#            1       0.62      0.48      0.54       374

#     accuracy                           0.78      1409
#    macro avg       0.72      0.69      0.70      1409
# weighted avg       0.77      0.78      0.77      1409

# Confusion Matrix:
# [[925 110]
#  [195 179]]






































