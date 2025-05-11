import os
from PIL import Image, ImageChops


# 统计文件夹内文件夹个数
def count_folder(path):
    count = 0
    for item in os.listdir(path):
        if os.path.isdir(os.path.join(path, item)):
            count += 1
    return count


# 从输入地址获取图片进行差分，输出到输出地址
def diff(in_path, out_path):
    counts = count_folder(in_path)
    for i in range(1, counts + 1):
        img1_path = os.path.join(in_path, "MyVideo_%s" % i, "1.jpg")
        img2_path = os.path.join(in_path, "MyVideo_%s" % i, "8.jpg")
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        difference = ImageChops.subtract(img1, img2)
        out_full_path = os.path.join(out_path, "%s.jpg" % i)
        difference.save(out_full_path)


# 主函数
'''
diff("data/train/Fail", "data/process/train/Fail")
diff("data/train/Success", "data/process/train/Success")
diff("data/val/Fail", "data/process/val/Fail")
diff("data/val/Success", "data/process/val/Success")
'''
diff("test_data", "test_process")
