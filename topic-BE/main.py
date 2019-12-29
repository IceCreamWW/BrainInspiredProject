import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet

if len(sys.argv) < 2:
    exit(-1)
elif len(sys.argv) < 3:
    sys.argv.append(0)

torch.cuda.set_device(int(sys.argv[2]))
turn = sys.argv[3]

model_name = "resnet{}".format(sys.argv[1])

# 准备数据集并预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  #先四周填充0，再把图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train) #训练数据集
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)   #生成一个个batch进行批训练，组成batch的时候顺序打乱取

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = getattr(resnet, model_name)(num_classes = 10).cuda()

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

best_acc = 0

root_path = 'checkpoints/{}/{}'.format(model_name, turn)

os.makedirs(root_path, exist_ok=True)

with open(root_path + "/acc.txt", "w") as f:
    with open(root_path + "/log.txt", "w")as f2:
        for epoch in range(0, 200):
            print('\nEpoch: %d' % (epoch + 1))
            net.train()
            sum_loss = 0.0
            correct = 0.0
            total = 0.0
            for i, data in enumerate(trainloader, 0):
                # 准备数据
                length = len(trainloader)
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()
                optimizer.zero_grad()

                # forward + backward
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # 每训练1个batch打印一次loss和准确率
                sum_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += predicted.eq(labels.data).cpu().sum()
                if i % 20 == 0:
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                      % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                      % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('\n')
                    f2.flush()

            # 每训练完一个epoch测试一下准确率
            print("Waiting Test!")
            with torch.no_grad():
                correct = 0
                total = 0
                for data in testloader:
                    net.eval()
                    images, labels = data
                    images, labels = images.cuda(), labels.cuda()
                    outputs = net(images)
                    # 取得分最高的那个类 (outputs.data的索引号)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum()
                print('测试分类准确率为：%.3f%%' % (100 * correct / total))
                acc = 100. * correct / total
                # 将每次测试结果实时写入acc.txt文件中
                print('Saving model......')
                torch.save(net.state_dict(), '%s/net_%03d.pth' % (root_path, epoch + 1))
                f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, acc))
                f.write('\n')
                f.flush()
                # 记录最佳测试分类准确率并写入best_acc.txt文件中
                if acc > best_acc:
                    f3 = open("best_acc.txt", "w")
                    f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                    f3.close()
                    best_acc = acc
        print("Training Finished, TotalEPOCH=%d" % 200)