import argparse
import torch
from collections import OrderedDict
from torch import nn, optim
from torchvision import datasets, transforms, models
from workspace_utils import active_session
import time


def get_arguments():
    
    parser = argparse.ArgumentParser(description="Train.py")
    parser.add_argument("data_dir", action="store", default="./flowers/", help="data directory of the training files", type=str)
    parser.add_argument('--gpu', default="False", action="store_true", dest="gpu")
    parser.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="./")
    parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001, type = float)
    parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)
    parser.add_argument('--epochs', dest="epochs", action="store", type=int, default = 6)
    parser.add_argument("--max_steps", default=None, type=int)
    parser.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
    parser.add_argument("--print_every", default=20, 
                    help="number of batches between validations and print updates", type=int)
    
    return parser.parse_args()

def prep_data(data_dir):
    
    mean_norm = [0.485, 0.456, 0.406]
    std_norm = [0.229, 0.224, 0.225]
    batch_size = 64

    transform_normalize = transforms.Normalize(mean_norm, std_norm)
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    
    data_transforms = { 
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(size=224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transform_normalize
        ]),
        'valid': transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transform_normalize
        ])
    }

    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
    }

    train_loader = torch.utils.data.DataLoader(image_datasets['train'], batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(image_datasets['valid'], batch_size, shuffle = True)
    
    return train_loader, validation_loader, image_datasets['train'].class_to_idx


def setup_model(arch = 'vgg16', input_size = 25088, gpu = False, hidden_layer1=120, dropout = 0.5, lr = 0.001):

    if arch == 'vgg13':
        pretrained_model = models.vgg13(pretrained=True)
    elif arch == 'vgg16':
        pretrained_model = models.vgg16(pretrained=True)
    elif arch == 'alexnet':
        pretrained_model = models.alexnet(pretrained=True)
    elif arch == 'resnet18':
        pretrained_model = models.resnet18(pretrained=True)
    elif arch == 'densenet121':
        pretrained_model = models.densenet121(pretrained=True)
    else:
        print('Unknown Model')
        pretrained_model = None
    
    if  torch.cuda.is_available() and gpu == True:
        device = torch.device("cuda")
        print("GPU: Enabled")
    else:
        device = torch.device("cpu")
        print("GPU: Disabled or Not Available")
    
    model = pretrained_model

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                ('inputs', nn.Linear(input_size, hidden_layer1)),
                ('relu1', nn.ReLU()),
                ('dropout',nn.Dropout(dropout)),
                ('hidden_layer1', nn.Linear(hidden_layer1, 90)),
                ('relu2',nn.ReLU()),
                ('hidden_layer2',nn.Linear(90,70)),
                ('relu3',nn.ReLU()),
                ('hidden_layer3',nn.Linear(70,102)),
                ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    model.to(device)
    
    return criterion, optimizer, model, device

def validation(model, device, dataloader, criterion):
    loss, accuracy = 0, 0
    model.eval()

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            loss += batch_loss.item()
            
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    loss = loss/len(dataloader)
    accuracy = accuracy/len(dataloader)
    return loss, accuracy

def train(model, device, train_loader, valid_loader, criterion, optimizer, 
          epochs=12, print_every=5, max_steps=None):
    steps = 0
    running_loss = 0
    running_accuracy = 0

    model.to(device)
    
    print("\nTraining Started.\n")

    for epoch in range(epochs):
        
        start = time.time()
        
        if steps == max_steps:
            break
        for i, (inputs, labels) in enumerate(train_loader):
            steps += 1
            
            if steps == max_steps:
                break

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            ps = torch.exp(outputs)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            running_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            if steps % print_every == 0:
                
                validation_lost, accuracy = validation(model, device, valid_loader, criterion)
                train_loss = running_loss / print_every
                train_accuracy = running_accuracy / print_every


                print("Step: {} ||".format(steps),
                    "Epoch: {}/{} ||".format(epoch+1, epochs),
                      "Loss: {:.4f} ||".format(running_loss/print_every),
                      "Train Accuracy: {:.4f} ||".format(train_accuracy),
                      "Validation Lost {:.4f} ||".format(validation_lost),
                       "Validation Accuracy: {:.4f}".format(accuracy))
                


                running_loss = 0
                running_accuracy =0
                
        end = time.time()
        print_duration(start, end, f"Epoch {epoch + 1} duration:")
    
    print("**Training Complete**")
    
    return model
    
def check_accuracy(model, validation_loader, device):    
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for data in validation_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy on test images is: %d%%' % (100 * correct / total))
    
def save_checkpoint(model, train_data_class_to_idx, path, 
                    arch ='vgg16', hidden_layer1=120, dropout=0.5, 
                    lr=0.001, epochs = 6):
    model.class_to_idx = train_data_class_to_idx
    model.cpu
    path = path + "checkpoint.pth"
    torch.save({'structure' : arch,
                'hidden_layer1': hidden_layer1,
                'dropout': dropout,
                'lr': lr,
                'epoch': epochs,
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx},
                path)
    
def print_duration(start, end, message):
    end = time.time()
    tot_time = end - start
    print(f"\n** {message}",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)), "**\n")

def main():
    args = get_arguments()
    
    start = time.time()
    
    train_loader, validation_loader, train_data_class_to_idx = prep_data(args.data_dir)
    
    
    criterion, optimizer, model, device = setup_model(arch = args.arch,
                                                      gpu = args.gpu, 
                                                      hidden_layer1 = args.hidden_units,
                                                     dropout = args.dropout, 
                                                      lr = args.learning_rate)

    
    with active_session():
        print('Starting Training...')
        train(model, 
              device, 
              train_loader, 
              validation_loader,
              criterion, 
              optimizer,
              epochs = args.epochs, 
              print_every = args.print_every,
              max_steps = args.max_steps)
        
    save_checkpoint(model, train_data_class_to_idx, args.save_dir, 
                    args.arch , args.hidden_units, 
                    args.dropout, args.learning_rate, args.epochs)
    
    
    end = time.time()
    print_duration(start, end, "Training duration:")
    
    print("\nChecking Accuracy...")
    check_accuracy(model, validation_loader, device)
    
    end = time.time()
    print_duration(start, end, "Total Runtime:")
    
    
if __name__ == '__main__': main()