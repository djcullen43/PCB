# import torch and other necessary modules from torch
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

transform = transforms.Compose(
    [transforms.Resize((100, 100)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = datasets.ImageFolder('./A', transform=transform)
# Print the image and label of the first element in the dataset
image, label = dataset[0]
print(image.size())
print(label)

test_size = int(len(dataset) * 0.2)
train_size = len(dataset) - test_size
test_set, train_set = torch.utils.data.random_split(dataset, [test_size, train_size])

# model hyperparameter
learning_rate = 0.0001
batch_size = 128
epoch_size = 5

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
# get a batch of data
# iterate through the data loader

for i, data in enumerate(train_loader, 0):
    inputs, labels = data
    print(labels)
for images, labels in train_loader:
    # print the shape of the image tensor
    print(images.shape)
    # print the labels of the images
    print(labels)
    # visualize a few images from the batch (optional)
    # make sure to import matplotlib.pyplot first
    import matplotlib.pyplot as plt

    plt.imshow(images[0].permute(1, 2, 0))
    plt.show()


# model design goes here
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


device = 'cuda' if torch.cuda.is_available() else 'cpu'  # whether your device has GPU
print(f"Using {device} device")
cnn = CNN().to(device)  # move the model to GPU
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.0001)

# Instantiate the CNN model
cnn = CNN()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

cnn.train()
# Loop through the training data for the desired number of epochs
n_total_steps = len(train_loader)
for epoch in range(epoch_size):
    # Loop through the batches of training data
    # print("Epoch")
    loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        # Clear the gradients
        optimizer.zero_grad()
        # Forward pass the data through the model
        outputs = cnn(inputs)
        # Calculate the loss
        loss = criterion(outputs, labels)
        # Backpropagate the gradients
        loss.backward()
        # print(loss)
        # Update the weights
        optimizer.step()
        # print some statistics
        loss += loss.item() # add loss for current batch
        if i % 100 == 99:  # print out average loss every 100 batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss / 100:.3f}')
        loss = 0.0
print('Finished Training')
# evaluate the model on the test set
ground_truth = []
prediction = []
cnn.eval()  # turn on evaluation mode
with torch.no_grad():  # turn off gradient calculation for evaluation
    y_true = []
    y_pred = []
    for data in test_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = cnn(inputs)
        _, predicted = torch.max(outputs.data, 1)
        y_true.extend(labels.tolist())
        y_pred.extend(predicted.tolist())

# calculate the evaluation metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)

# print the evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Recall: {recall}")
print(f"Precision: {precision}")
