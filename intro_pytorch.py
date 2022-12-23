import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms



def get_data_loader(training):
    
    custom_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    train_set=datasets.FashionMNIST('./data',train=True,
            download=True,transform=custom_transform)

    test_set=datasets.FashionMNIST('./data', train=False,
            transform=custom_transform)
    
    if training is True:
        loader = torch.utils.data.DataLoader(train_set, batch_size = 64)
        return loader
    else:
        loader = torch.utils.data.DataLoader(test_set, batch_size = 64)
        return loader



def build_model():
    model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
            )
    return model
    




def train_model(model, train_loader, criterion, T):
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) #uses stoicheatric gradient descent to optimize algorithm
    model.train()
    for epoch in range(T):
        running_loss = 0.0
        correct = 0
        total = 0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            opt.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            loss.backward()
            opt.step()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()*train_loader.batch_size
        
        average = round(100 * (correct/total),2)
        running_loss = round(running_loss/len(train_loader.dataset) ,3)
        print("Train Epoch: " + str(epoch) + "\t" + "Accuracy: " + str(correct) + "/" + str(total) +"("+str(average)+"%"+")" + "\t" + "Loss: " + str(running_loss))   
    


def evaluate_model(model, test_loader, criterion, show_loss):
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        correct = 0
        total = 0

        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item() * test_loader.batch_size

        average = round(100 * (correct/total),2)
        running_loss = round(running_loss/len(test_loader.dataset) ,4)
        
        if show_loss is True:
            print("Average Loss: " + str(running_loss))
            print("Accuracy: " + str(average) + "%")
        else:
            print("Accuracy: " + str(average) + "%")
        

    
    


def predict_label(model, test_images, index):
    
    #feed the images to my network 
    logits = model(test_images[index])
    #output is a list of propabilties from that image
    prob = F.softmax(logits, dim=1)
    
    values,tens = torch.topk(prob, 3, dim=1, largest=True, sorted=True, out=None)
    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"] 
    tens = torch.squeeze(tens)
    values = torch.squeeze(values)
    indices = tens.tolist()
    percent = values.tolist()
    for i,j in zip(indices,percent):
        p = round(j*100,2)
        print(str(class_names[i]) + ": " + str(p) + "%")



if __name__ == '__main__':
    train_loader = get_data_loader(True)
    test_loader = get_data_loader(False)
    model = build_model()
    criterion = nn.CrossEntropyLoss()
    train_model(model, train_loader, criterion, T = 5)
    evaluate_model(model, test_loader, criterion, show_loss = True)
    

    pred_set, _ = next(iter(test_loader))
    
    predict_label(model, pred_set, 1)
