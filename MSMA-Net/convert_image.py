from PIL import Image
from os import listdir

folder_dir = "D:\OneDrive - Hanoi University of Science and Technology\Desktop\Đồ án thiết kế\My Project\MSMA-Net\dataset\TrainDataset\images"
png_path = "D:\OneDrive - Hanoi University of Science and Technology\Desktop\Đồ án thiết kế\My Project\MSMA-Net\dataset\TrainDataset\images\png/"

for images in listdir(folder_dir):
    img_png = Image.open(folder_dir + '/' + images)
    img_png.save(png_path + str(i) + '.png')
    i += 1

# img_jpg = Image.open("D:\OneDrive - Hanoi University of Science and Technology\Desktop\Đồ án thiết kế\My Project\MSMA-Net\dataset\TestDataset\ETIS/test_polyps_ (100).jpg")

# img_jpg.save(png_path + str(i) + '.png')

# list = listdir(folder_dir)
# # print('test_polyps_ (95).jpg' in list)

# for image in list:
#     img_png = Image.open()