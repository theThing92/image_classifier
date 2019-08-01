import torch
import torchvision
from collections import OrderedDict
import json
import matplotlib.pyplot as plt
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Script to train a neuronal network model for image classification from scratch.')

    parser.add_argument(
        'directory',
        type=str,
        default='images/flowers',
        help='Root path to the training, validation and testing data.')

    parser.add_argument(
        '--save_dir',
        required=False,
        type=str,
        help='Optional path to save the trained model to.')

    parser.add_argument(
        '--arch',
        required=False,
        type=str,
        default="resnet152",
        help='Base model architecture for training.')

    parser.add_argument(
        '--learning_rate',
        type=float,
        required=False,
        default=0.001,
        help='Learning rate for training.')

    parser.add_argument(
        '--hidden_units',
        type=int,
        required=False,
        default=1024,
        help='Number of hidden units in layer.')

    parser.add_argument(
        '--epochs',
        type=int,
        required=False,
        default=20,
        help='Number of epochs for training.')

    parser.add_argument(
        '--gpu',
        type=str,
        required=False,
        default="gpu",
        help='Enable gpu training, if available.')


    args = parser.parse_args()



    data_dir = args.directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'


    # TODO: Define your transforms for the training, validation, and testing sets

    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])


    train_transforms = torchvision.transforms.Compose([torchvision.transforms.RandomRotation(30),
                                                       torchvision.transforms.RandomResizedCrop(224),
                                                       torchvision.transforms.RandomHorizontalFlip(),
                                                       torchvision.transforms.RandomVerticalFlip(),
                                                       torchvision.transforms.ToTensor(),
                                                       normalize])

    val_transforms = torchvision.transforms.Compose([torchvision.transforms.Resize(255),
                                                      torchvision.transforms.CenterCrop(224),
                                                      torchvision.transforms.ToTensor(),
                                                      normalize])


    test_transforms = torchvision.transforms.Compose([torchvision.transforms.Resize(255),
                                                      torchvision.transforms.CenterCrop(224),
                                                      torchvision.transforms.ToTensor(),
                                                      normalize])


    # load the datasets with ImageFolder
    train_data = torchvision.datasets.ImageFolder(train_dir, transform=train_transforms)
    val_data = torchvision.datasets.ImageFolder(valid_dir, transform=val_transforms)
    test_data = torchvision.datasets.ImageFolder(test_dir, transform=test_transforms)

    # using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=256, shuffle=True)
    val_loader =  torch.utils.data.DataLoader(val_data, batch_size=256, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=256, shuffle=True)

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)



    # TODO: Build and train your network

    if args.arch == "resnet152":
        model = torchvision.models.resnet152(pretrained=True)
        input_size = model.fc.in_features

    elif args.arch == "densenet201":
        model = torchvision.models.densenet201(pretrained=True)
        input_size = model.classifier.in_features


    # freezing the parameters for the feature extraction part of the model
    for param in model.parameters():
        param.requires_grad = False

    hidden_layer_size = args.hidden_units

    # define new classifier layer
    classifier_layer_new = torch.nn.Sequential(
        OrderedDict([('fc1', torch.nn.Linear(input_size, hidden_layer_size)),
                     ('relu', torch.nn.ReLU()),
                     ('dropout', torch.nn.Dropout(0.2)),
                     ('fc2', torch.nn.Linear(hidden_layer_size, len(cat_to_name))),
                     ('output', torch.nn.LogSoftmax(dim=1))]))

    if args.arch == "resnet152":
        model.fc = classifier_layer_new

    elif args.arch == "densenet201":
        model.classifier = classifier_layer_new


    # use available gpu ressources
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu == "gpu" else "cpu")
    model.to(device)



    # only train fully-connected classifer part of model
    criterion = torch.nn.NLLLoss()
    learning_rate = args.learning_rate

    if args.arch == "resnet152":
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=learning_rate)

    elif args.arch == "densenet201":
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)

    epochs = args.epochs
    print_every = 50
    steps = 0

    train_losses = []
    val_losses = []

    val_accuracies = []


    # code adapted from:
    # https://github.com/udacity/deep-learning-v2-pytorch/blob/master/intro-to-pytorch/Part%208%20-%20Transfer%20Learning%20(Solution).ipynb
    def training(model, criterion, steps):
        model.train()
        running_loss = 0

        for inputs, labels in train_loader:
            model.train()

            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # forward pass through network
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        return running_loss


    # code adapted from:
    # https://github.com/udacity/deep-learning-v2-pytorch/blob/master/intro-to-pytorch/Part%208%20-%20Transfer%20Learning%20(Solution).ipynb
    def validation(model, criterion):
        model.eval()

        with torch.no_grad():
            val_loss = 0
            val_accuracy = 0

            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model.forward(inputs)
                batch_loss = criterion(logps, labels)

                val_loss += batch_loss.item()

                # Calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                val_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            return val_loss, val_accuracy


    # code adapted from:
    # https://github.com/udacity/deep-learning-v2-pytorch/blob/master/intro-to-pytorch/Part%208%20-%20Transfer%20Learning%20(Solution).ipynb
    for epoch in range(epochs):
        running_loss = 0

        running_loss = training(model, criterion, steps)

        if steps % print_every == 0:
            val_loss, val_accuracy = validation(model, criterion)

            training_loss = running_loss / print_every
            validation_loss = val_loss / len(val_loader)
            validation_accuracy = val_accuracy / len(val_loader)

            print("Epoch: {}/{} |||".format(epoch + 1, epochs),
                  "Training Loss: {} |||".format(training_loss),
                  "Validation Loss: {} |||".format(validation_loss),
                  "Validation Accuracy: {} |||".format(validation_accuracy)
                  )

            train_losses.append(training_loss)
            val_losses.append(validation_loss)
            val_accuracies.append(validation_accuracy)


    # plot losses
    index_epochs = list(range(1,epochs+1))

    plt.plot(index_epochs, train_losses)
    plt.plot(index_epochs, val_losses)
    plt.title("Training vs. Validation Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["train","val"])
    plt.savefig("losses.png")
    plt.show()
    plt.close()


    # plot validation accuracy
    plt.plot(index_epochs, val_accuracies)
    plt.title("Validation Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.savefig("accuracy.png")
    plt.show()
    plt.close()


    # code adapted from:
    # https://github.com/udacity/deep-learning-v2-pytorch/blob/master/intro-to-pytorch/Part%208%20-%20Transfer%20Learning%20(Solution).ipynb
    def test(model, criterion):
        model.eval()

        with torch.no_grad():
            test_loss = 0
            test_accuracy = 0

            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model.forward(inputs)
                batch_loss = criterion(logps, labels)

                test_loss += batch_loss.item()

                # Calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            test_loss = test_loss / len(test_loader)
            test_accuracy = test_accuracy / len(test_loader)
            return test_loss, test_accuracy

    # TODO: Do validation on the test set
    test_loss, test_accuracy = test(model, criterion)
    print("Test Loss: {}".format(test_loss))
    print("Test Accuracy: {}".format(test_accuracy))


    if args.save_dir is not None:
        # TODO: Save the checkpoint
        model.class_to_idx = train_data.class_to_idx
        model_path = args.save_dir
        torch.save(model, model_path)



