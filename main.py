import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import os
import wandb
from utils.data_processing.get_dataset import get_train_data, get_validation_data
from utils.models import *
from utils.utilities.utils import progress_bar

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

def define_model(resume: bool):
    torch.manual_seed(1274)
    global start_epoch, best_acc
    net = ResNet18()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    wandb.init(project='optas_project_21', name='CIFAR10_RESNET18_2SUBSETS', config=args)
    wandb.watch(net)

    if resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    return net



def train(epoch,net,trainloader,criterion,optimizer):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets, _) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc=correct/total
    wandb.log({"train_loss":1-acc},step=epoch)

def test(epoch,net,testloader,criterion):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

    acc=acc/100
    wandb.log({"classification_loss":1-acc},step=epoch)

def construct_and_train(args: dict):
    global start_epoch
    trainset1,trainset2 = get_train_data()
    testset = get_validation_data()
    trainloader = torch.utils.data.DataLoader(trainset1,batch_size=128, shuffle= True,num_workers=2)
    testloader = torch.utils.data.DataLoader(testset,batch_size=100, shuffle= False,num_workers=1)

    net = define_model(args['resume'])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args['lr'],
                      momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['epochs'])

    for epoch in range(start_epoch, start_epoch + args['epochs']):
        train(epoch,net,trainloader,criterion,optimizer)
        test(epoch,net,testloader,criterion)
        scheduler.step()

def construct_and_test():
    trainset1,trainset2 = get_train_data()
    testset = get_validation_data()
    testloader1 = torch.utils.data.DataLoader(trainset1,batch_size=100, shuffle= False,num_workers=1)
    testloader2 = torch.utils.data.DataLoader(trainset2,batch_size=100, shuffle= False,num_workers=1)
    testloader = torch.utils.data.DataLoader(testset,batch_size=100, shuffle= False,num_workers=1)

    net = define_model(True)

    criterion = nn.CrossEntropyLoss()

    print("Performance of saved model on Subset 1 is:")
    test(0,net,testloader1,criterion)
    print("Performance of saved model on Subset 2 is:")
    test(1,net,testloader2,criterion)
    print("Performance of saved model on Test set is:")
    test(2,net,testloader,criterion)

def construct_and_train_second_classifier():
    global start_epoch, best_acc
    construct_and_test()
    trainset1,trainset2 = get_train_data()
    trainset= trainset1.dataset
    net = define_model(True)
    previous_labels=trainset.targets[range(25000,50000)]
    with torch.no_grad():
        for batch_idx, (inputs, _, indices) in enumerate(torch.utils.data.DataLoader(trainset2,batch_size=100, shuffle= False, num_workers=1)):
            inputs = inputs.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            trainset.targets[indices]=predicted.cpu().detach().numpy()
    print("For verification in trainset2 percentage of common labels compared to previously is")
    print(np.sum(previous_labels == trainset.targets[range(25000,50000)])/25000 )
    trainset.targets[range(25000)] = (trainset.targets[range(25000)] + 1) % 10

    testset = get_validation_data()
    testloader = torch.utils.data.DataLoader(testset,batch_size=100, shuffle= False,num_workers=1)
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=128, shuffle= True,num_workers=2)


    net = define_model(False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args['lr'],
                      momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['epochs'])
    start_epoch, best_acc = 0, 0

    for epoch in range(args['epochs']):
        train(epoch,net,trainloader,criterion,optimizer)
        test(epoch,net,testloader,criterion)
        scheduler.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch experiments")
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument("--epochs",default=200,type=int, choices=range(0,401), help="number of epochs")
    parser.add_argument("--mode", type=str, default="TrainFirst", choices={"TrainFirst","TrainSecond","TestFirst"}, help=' switch mode')
    args = vars(parser.parse_args())
    if args['mode'] == "TrainFirst":
        construct_and_train(args)
    elif args['mode'] == "TrainSecond":
        construct_and_train_second_classifier()
    else:
        construct_and_test()
