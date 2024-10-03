import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import accuracy_score, recall_score, precision_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

# Determine if GPU is available and use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [transforms.Resize((100, 100)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

dataset = datasets.ImageFolder('./A', transform=transform)

test_size = int(len(dataset) * 0.2)
train_size = len(dataset) - test_size
test_set, train_set = torch.utils.data.random_split(dataset, [test_size, train_size])

# Model hyperparameters
learning_rate = 0.0001
batch_size = 128
epoch_size = 5

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)


# Model design
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=128 * 12 * 12, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=2)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = x.view(-1, 128 * 12 * 12)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.softmax(self.fc2(x), dim=1)
        return x


# Instantiate the CNN model
cnn = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

# Training loop
cnn.train()
for epoch in range(epoch_size):
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = cnn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}/{epoch_size} completed')

# Saving the model
model_save_path = "./cnn_model.pth"
torch.save(cnn.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')

# Evaluate the model on the test set
cnn.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = cnn(inputs)
        _, predicted = torch.max(outputs.data, 1)
        y_true.extend(labels.tolist())
        y_pred.extend(predicted.tolist())

# Calculate evaluation metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

# Define the label mapping: 0 -> "car", 1 -> "no car"
label_mapping = {0: 'Occupied', 1: 'Vacant'}

# Save images with predictions
save_dir = './output_images'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

cnn.eval()
with torch.no_grad():
    for i, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.to(device)
        outputs = cnn(inputs)
        _, predicted = torch.max(outputs.data, 1)

        # Visualize and save the images
        for j in range(inputs.size(0)):
            img = inputs[j].cpu().permute(1, 2, 0).numpy()  # Convert to (H, W, C) format
            img = (img * 0.5) + 0.5  # De-normalize the image

            predicted_label = label_mapping[predicted[j].item()]
            actual_label = label_mapping[labels[j].item()]

            plt.imshow(img)
            plt.title(f'Predicted: {predicted_label}, Actual: {actual_label}')
            image_path = os.path.join(save_dir, f'image_{i * batch_size + j}.png')
            plt.savefig(image_path)
            plt.close()

print(f'Prediction images saved to {save_dir}')
