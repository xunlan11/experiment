import os
import glob
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from model import AlexNet
from PIL import Image
'''
请先正确放置测试集，再运行process，最后运行predict
'''


def main():
    # 选择设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 图片预处理
    data_transform = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 加载训练好的模型
    model = AlexNet(num_classes=2, init_weights=True).to(device)
    weights_path = "AlexNet.pth"
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    # 测试集
    folder_path = "test_process"
    jpg_files = glob.glob(os.path.join(folder_path, "*.jpg"))
    jpg_count = len(jpg_files)
    for i in range(1, jpg_count + 1):
        # 打开图片并预处理
        img_path = os.path.join(folder_path, "%s.jpg" % i)
        img = Image.open(img_path)
        plt.imshow(img)
        img = data_transform(img)
        img = torch.unsqueeze(img, dim=0)
        # 进行测试
        with torch.no_grad():
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
        print(predict_cla)


if __name__ == "__main__":
    main()
