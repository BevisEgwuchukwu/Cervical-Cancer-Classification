import random
import copy
import shutil
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets
from torchvision import transforms, models
from torch.utils.data import Dataset, random_split, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import seaborn as sns

raw_data = datasets.ImageFolder(
    "/Users/macintosh/Desktop/Cervical Cancer Classification/archive (4)"
)

image, label = raw_data[0]
plt.imshow(image)
plt.title(raw_data.classes[label])
plt.show()


def extract_bmp(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".bmp"):
            file_path = os.path.join(input_dir, filename)

            shutil.move(file_path, output_dir)


extract_bmp(
    "/Users/macintosh/Desktop/Cervical Cancer Classification/archive (4)/im_Dyskeratotic",
    "/Users/macintosh/Desktop/Cervical Cancer Classification/main_im_Dyskeratotic",
)
extract_bmp(
    "/Users/macintosh/Desktop/Cervical Cancer Classification/archive (4)/im_Koilocytotic",
    "/Users/macintosh/Desktop/Cervical Cancer Classification/main_im_Koilocytotic",
)
extract_bmp(
    "/Users/macintosh/Desktop/Cervical Cancer Classification/archive (4)/im_Metaplastic",
    "/Users/macintosh/Desktop/Cervical Cancer Classification/main_im_Metaplastic",
)
extract_bmp(
    "/Users/macintosh/Desktop/Cervical Cancer Classification/archive (4)/im_Parabasal",
    "/Users/macintosh/Desktop/Cervical Cancer Classification/main_im_Parabasal",
)
extract_bmp(
    "/Users/macintosh/Desktop/Cervical Cancer Classification/archive (4)/im_Superficial-Intermediate",
    "/Users/macintosh/Desktop/Cervical Cancer Classification/main_im_Superficial-Intermediate",
)

# defining parts for normal and abnormal catgories
normal_dirs = [
    "/Users/macintosh/Desktop/Cervical Cancer Classification/normal_cells/main_im_Superficial-Intermediate",
    "/Users/macintosh/Desktop/Cervical Cancer Classification/normal_cells/main_im_Metaplastic",
]
abnormal_dirs = [
    "/Users/macintosh/Desktop/Cervical Cancer Classification/abnormal_cells/main_im_Dyskeratotic",
    "/Users/macintosh/Desktop/Cervical Cancer Classification/abnormal_cells/main_im_Koilocytotic",
    "/Users/macintosh/Desktop/Cervical Cancer Classification/abnormal_cells/main_im_Parabasal",
]


# Function to label the dataset as normal and abnormal
def label_dataset(base_dirs, label):
    labeled_images = []
    for dir in base_dirs:
        for image in os.listdir(dir):
            image_path = os.path.join(dir, image)
            if image.endswith(".bmp"):
                labeled_images.append((image_path, label))
    return labeled_images


# Labelling datasets(normal = 0, abnormal = 1)
normal_images = label_dataset(normal_dirs, 0)
abnormal_images = label_dataset(abnormal_dirs, 1)

# Combining both normal and abnormal datasets
all_images = normal_images + abnormal_images

# Shuffling the dataset
random.shuffle(all_images)


class imageClass(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

dataset = imageClass(all_images, transform=transform)

# Splitting the dataset into training and testing sets
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

# For neural network, data loader is used to load the data
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

print(len(train_set))
len(test_set)


# Model Setup (Simple CNN)
class simpleCNN(nn.Module):
    def __init__(self):
        super(simpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.ln1 = nn.Linear(32 * 56 * 56, 128)
        self.ln2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = torch.relu(self.ln1(x))
        x = self.ln2(x)

        return x


model_cnn = simpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_cnn.parameters(), lr=0.001)


# Training the model
def train(model_cnn, data_loader, criterion, optimizer, epochs=5):
    model_cnn.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in data_loader:
            optimizer.zero_grad()
            outputs = model_cnn(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(data_loader):.4f}")


# Evaluating the model
def evaluate(model_cnn, data_loader):
    model_cnn.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model_cnn(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy of the model on the images: {100 * correct / total:.2f}%")


train(model_cnn, train_loader, criterion, optimizer)

print(evaluate(model_cnn, train_loader))
print(evaluate(model_cnn, test_loader))

# Model Setup (RESNET_50)
model_resnet = models.resnet50(weights="DEFAULT")
num_ftrs = model_resnet.fc.in_features
model_resnet.fc = nn.Linear(num_ftrs, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_resnet = model_resnet.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_resnet.parameters(), lr=0.001)


# Early Stopping Class
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model_resnet):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model_resnet.state_dict())
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model_resnet.state_dict())
            self.counter = 0


# 3. Training Function
def train_model(
    model_resnet,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    early_stopping,
):

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    for epoch in range(num_epochs):
        model_resnet.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model_resnet(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # Validation Phase
        model_resnet.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model_resnet(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Validation Loss: {avg_val_loss:.4f}"
        )

        scheduler.step()

        # Check for early stopping
        early_stopping(avg_val_loss, model_resnet)
        if early_stopping.early_stop:
            print("Stopping early due to no improvement in validation loss.")
            # Load the best model state found so far
            model_resnet.load_state_dict(early_stopping.best_model_state)
            break


# Evaluation Function
def evaluate_model(model_resnet, data_loader):
    model_resnet.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model_resnet(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def get_predictions(model_resnet, dataloader, device="cpu"):
    model_resnet.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model_resnet(images)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    return all_labels, all_preds


# Execution
if __name__ == "__main__":
    early_stopping = EarlyStopping(patience=5, min_delta=0.01)

    train_model(
        model_resnet,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        num_epochs=30,
        early_stopping=early_stopping,
    )
    train_accuracy = evaluate_model(model_resnet, train_loader)
    test_accuracy = evaluate_model(model_resnet, test_loader)
    print(f"Training Accuracy: {train_accuracy:.2f}%")
    print(f"Testing Accuracy: {test_accuracy:.2f}%")

    model_path = "/Users/macintosh/Desktop/Cervical Cancer Classification/Models/model_resnet.pth"
    torch.save(model_resnet, model_path)

    all_labels, all_preds = get_predictions(model_resnet, test_loader)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    cm = confusion_matrix(all_labels, all_preds)
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Abnormal"],
        yticklabels=["Normal", "Abnormal"],
    )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.show()

# Model Setup (ALEXNET)
model_alex = models.alexnet(weights="AlexNet_Weights.IMAGENET1K_V1")
num_ftrs = model_alex.classifier[6].in_features
model_alex.classifier[6] = nn.Linear(num_ftrs, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_alex = model_alex.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_alex.parameters(), lr=0.001)


# Early Stopping Class
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model_alex):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model_alex.state_dict())
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model_alex.state_dict())
            self.counter = 0


# 3. Training Function
def train_model(
    model_alex,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    early_stopping,
):

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    for epoch in range(num_epochs):
        model_alex.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model_alex(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # Validation Phase
        model_alex.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model_alex(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Validation Loss: {avg_val_loss:.4f}"
        )

        scheduler.step()

        # Check for early stopping
        early_stopping(avg_val_loss, model_alex)
        if early_stopping.early_stop:
            print("Stopping early due to no improvement in validation loss.")
            # Load the best model state found so far
            model_alex.load_state_dict(early_stopping.best_model_state)
            break


# Evaluation Function
def evaluate_model(model_alex, data_loader):
    model_alex.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model_alex(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def get_predictions(model_alex, dataloader, device="cpu"):
    model_alex.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model_alex(images)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    return all_labels, all_preds


# Execution
if __name__ == "__main__":
    early_stopping = EarlyStopping(patience=5, min_delta=0.01)

    train_model(
        model_alex,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        num_epochs=30,
        early_stopping=early_stopping,
    )
    train_accuracy = evaluate_model(model_alex, train_loader)
    test_accuracy = evaluate_model(model_alex, test_loader)
    print(f"Training Accuracy: {train_accuracy:.2f}%")
    print(f"Testing Accuracy: {test_accuracy:.2f}%")

    model_path = (
        "/Users/macintosh/Desktop/Cervical Cancer Classification/Models/model_alex.pth"
    )
    torch.save(model_alex, model_path)

    all_labels, all_preds = get_predictions(model_alex, test_loader)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    cm = confusion_matrix(all_labels, all_preds)
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Abnormal"],
        yticklabels=["Normal", "Abnormal"],
    )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.show()

# Model Setup (EFFICIENTNET_B3)
model_eff = models.efficientnet_b3(weights="DEFAULT")
num_ftrs = model_eff.classifier[1].in_features
model_eff.classifier[1] = nn.Linear(num_ftrs, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_eff = model_eff.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_eff.parameters(), lr=0.001)


# Early Stopping Class
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model_eff):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model_eff.state_dict())
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model_eff.state_dict())
            self.counter = 0


# 3. Training Function
def train_model(
    model_eff,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    early_stopping,
):

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    for epoch in range(num_epochs):
        model_eff.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model_eff(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # Validation Phase
        model_eff.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model_eff(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Validation Loss: {avg_val_loss:.4f}"
        )

        scheduler.step()

        # Check for early stopping
        early_stopping(avg_val_loss, model_eff)
        if early_stopping.early_stop:
            print("Stopping early due to no improvement in validation loss.")
            # Load the best model state found so far
            model_eff.load_state_dict(early_stopping.best_model_state)
            break


# Evaluation Function
def evaluate_model(model_eff, data_loader):
    model_eff.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model_eff(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def get_predictions(model_eff, dataloader, device="cpu"):
    model_eff.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model_eff(images)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    return all_labels, all_preds


# Execution
if __name__ == "__main__":
    early_stopping = EarlyStopping(patience=5, min_delta=0.01)

    train_model(
        model_eff,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        num_epochs=30,
        early_stopping=early_stopping,
    )
    train_accuracy = evaluate_model(model_eff, train_loader)
    test_accuracy = evaluate_model(model_eff, test_loader)
    print(f"Training Accuracy: {train_accuracy:.2f}%")
    print(f"Testing Accuracy: {test_accuracy:.2f}%")

    model_path = (
        "/Users/macintosh/Desktop/Cervical Cancer Classification/Models/model_eff.pth"
    )
    torch.save(model_eff, model_path)

    all_labels, all_preds = get_predictions(model_eff, test_loader)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    cm = confusion_matrix(all_labels, all_preds)
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Abnormal"],
        yticklabels=["Normal", "Abnormal"],
    )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.show()

# Model Setup (WIDE_RESNET50_2)
model_wide_res = models.wide_resnet50_2(weights="DEFAULT")
num_ftrs = model_wide_res.fc.in_features
model_wide_res.fc = nn.Linear(num_ftrs, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_wide_res = model_wide_res.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_wide_res.parameters(), lr=0.001)


# Early Stopping Class
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model_wide_res):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model_wide_res.state_dict())
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model_wide_res.state_dict())
            self.counter = 0


# 3. Training Function
def train_model(
    model_wide_res,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    early_stopping,
):

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    for epoch in range(num_epochs):
        model_wide_res.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model_wide_res(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # Validation Phase
        model_wide_res.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model_wide_res(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Validation Loss: {avg_val_loss:.4f}"
        )

        scheduler.step()

        # Check for early stopping
        early_stopping(avg_val_loss, model_wide_res)
        if early_stopping.early_stop:
            print("Stopping early due to no improvement in validation loss.")
            # Load the best model state found so far
            model_wide_res.load_state_dict(early_stopping.best_model_state)
            break


# Evaluation Function
def evaluate_model(model_wide_res, data_loader):
    model_wide_res.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model_wide_res(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def get_predictions(model_wide_res, dataloader, device="cpu"):
    model_wide_res.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model_wide_res(images)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    return all_labels, all_preds


# Execution
if __name__ == "__main__":
    early_stopping = EarlyStopping(patience=5, min_delta=0.01)

    train_model(
        model_wide_res,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        num_epochs=30,
        early_stopping=early_stopping,
    )
    train_accuracy = evaluate_model(model_wide_res, train_loader)
    test_accuracy = evaluate_model(model_wide_res, test_loader)
    print(f"Training Accuracy: {train_accuracy:.2f}%")
    print(f"Testing Accuracy: {test_accuracy:.2f}%")

    model_path = "/Users/macintosh/Desktop/Cervical Cancer Classification/Models/model_wide_res.pth"
    torch.save(model_wide_res, model_path)

    all_labels, all_preds = get_predictions(model_wide_res, test_loader)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    cm = confusion_matrix(all_labels, all_preds)
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Abnormal"],
        yticklabels=["Normal", "Abnormal"],
    )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.show()
