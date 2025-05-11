import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from model import AlexNet


def main():
    # 选择设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 图片预处理
    data_transform = {
        "transform": transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    # 训练集、验证集导入
    batch_size = 30
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_dataset = datasets.ImageFolder(root="data/process/train", transform=data_transform["transform"])
    train_num = len(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)
    train_steps = len(train_loader)
    val_dataset = datasets.ImageFolder(root="data/process/val", transform=data_transform["transform"])
    val_num = len(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=nw)

    # 字典{"Fail":0, "Success":1}
    result_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in result_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open("class_indices.json", "w") as json_file:
        json_file.write(json_str)

    # 神经网络
    print("using {} images for training, {} images for validation.".format(train_num, val_num))
    net = AlexNet(num_classes=2, init_weights=True)
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.0002)
    save_path = "AlexNet.pth"
    loss_function = nn.CrossEntropyLoss()
    best_acc = 0.0
    epochs = 20
    # 训练循环
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        # 批次遍历
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)
        net.eval()
        acc = 0.0
        # 验证
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
        val_accurate = acc / val_num
        print("[epoch %d] train_loss: %.3f  val_accuracy: %.3f" % (epoch + 1, running_loss / train_steps, val_accurate))
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
    print("Finished Training")


if __name__ == "__main__":
    main()
