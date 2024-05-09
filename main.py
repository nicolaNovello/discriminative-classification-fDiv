import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from NetArchitectures import *
from utils import *
from classes import *


# The code implementation modifies the code in https://github.com/kuangliu/pytorch-cifar

# Training
def train(net, epoch, model_name, cost_func_v):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        current_batch_size = inputs.shape[0]
        optimizer.zero_grad()
        outputs = net(inputs)
        data_y = get_random_batch(trainset, batch_size=current_batch_size, num_classes=num_classes).to(device)
        unif_outputs = net(data_y)
        loss = compute_loss_divergence(cost_func_v, outputs, unif_outputs, targets, num_classes, 128, 0.8, device)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        R_all = obtain_posterior_from_net_out(outputs, cost_func_v)
        _, predicted = R_all.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print("Epoch: %d | Model: %s | Div: %d | Loss: %.3f | Acc: %.3f%% (%d/%d): " % (
    epoch, model_name, cost_func_v, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    return net


def test(net, epoch, best_acc, model_name, cost_func_v, dataset_name):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            R_all = obtain_posterior_from_net_out(outputs, cost_func_v)
            _, predicted = R_all.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print("Loss: %.3f | Acc: %.3f%% (%d/%d): " % (
        test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        save_dict_lists_csv(
            "checkpoint/dataset_{}_model{}_v{}_seed{}.csv".format(dataset_name, model_name, cost_func_v, random_seed),
            {'Epoch': [epoch], 'Accuracy': [acc]})
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_dataset_{}_model{}_v{}_seed{}.pth'.format(dataset_name, model_name, cost_func_v, random_seed))
        best_acc = acc
    return best_acc, 100. * correct / total


def choose_architecture(net_arc, num_classes):
    print('==> Building model..')
    if net_arc == "VGG":
        net = VGG_custom('VGG19', num_classes=num_classes)
    elif net_arc == "ResNet18":
        net = ResNet18(num_classes=num_classes)
    elif net_arc == "SimpleDLA":
        net = SimpleDLA(num_classes=num_classes)
    elif net_arc == "PreActResNet18":
        net = PreActResNet18(num_classes=num_classes)
    elif net_arc == "DenseNet121":
        net = DenseNet121(num_classes=num_classes)
    elif net_arc == "MobileNetV2":
        net = MobileNetV2(num_classes=num_classes)
    return net


parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

# 2: GAN; 3: KL with softmax; 5: SL; 7: KL with softplus; 9: RKL; 10: HD; 12: P.
list_cost_func_v = [5]  
random_seeds = [0]
net_architectures = ["ResNet18"]  # ["DenseNet121","PreActResNet18","MobileNetV2", "VGG", "SimpleDLA"]
dataset_type = "cifar10"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

for random_seed in random_seeds:
    for net_arc in net_architectures:
        for cost_func_v in list_cost_func_v:
            torch.manual_seed(random_seed)
            random.seed(random_seed)
            np.random.seed(random_seed)

            best_acc = 0  # best test accuracy
            start_epoch = 0  # start from epoch 0 or last checkpoint epoch

            # Data
            print('==> Preparing data..')
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            if dataset_type == "cifar10":
                trainset = torchvision.datasets.CIFAR10(
                    root='./data', train=True, download=True, transform=transform_train)
                trainloader = torch.utils.data.DataLoader(
                    trainset, batch_size=128, shuffle=True, num_workers=2)

                testset = torchvision.datasets.CIFAR10(
                    root='./data', train=False, download=True, transform=transform_test)
                testloader = torch.utils.data.DataLoader(
                    testset, batch_size=100, shuffle=False, num_workers=2)

                classes = ('plane', 'car', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck')
                num_classes = 10

            elif dataset_type == "cifar100":
                trainset = torchvision.datasets.CIFAR100(
                    root='./data', train=True, download=True, transform=transform_train)
                trainloader = torch.utils.data.DataLoader(
                    trainset, batch_size=128, shuffle=True, num_workers=2)

                testset = torchvision.datasets.CIFAR100(
                    root='./data', train=False, download=True, transform=transform_test)
                testloader = torch.utils.data.DataLoader(
                    testset, batch_size=100, shuffle=False, num_workers=2)
                num_classes = 100

            # Model
            net = choose_architecture(net_arc, num_classes=num_classes)
            net = CombinedArchitectureSingle(net, cost_function_v=cost_func_v)

            net = net.to(device)
            if device == 'cuda':
                net = torch.nn.DataParallel(net)
                cudnn.benchmark = True

            if args.resume:
                # Load checkpoint.
                print('==> Resuming from checkpoint..')
                assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
                checkpoint = torch.load('./checkpoint/ckpt_dataset_{}_model{}_v{}_seed{}.pth'.format(dataset_type, net_arc, cost_func_v, random_seed))
                net.load_state_dict(checkpoint['net'])
                best_acc = checkpoint['acc']
                start_epoch = checkpoint['epoch']

            optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
            best_acc = 0
            accuracies_test_set = []
            for epoch in range(start_epoch, start_epoch+200):
                net = train(net, epoch, net_arc, cost_func_v)
                best_acc, acc_epoch = test(net, epoch, best_acc, net_arc, cost_func_v, dataset_name=dataset_type)
                accuracies_test_set.append(acc_epoch)
                scheduler.step()
            save_dict_lists_csv(
                "checkpoint/Accuracies_dataset_{}_model{}_v{}_seed{}.csv".format(dataset_type, net_arc, cost_func_v, random_seed),
                {'Epoch': range(200), 'Accuracy': accuracies_test_set})
