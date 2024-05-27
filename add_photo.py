import fitz
import numpy as np
from PIL import Image
import cv2
class Photo:
    # 类属性，默认为a3模板的属性
    type = ''
    color = (0, 0, 0)
    size = 7
    size_no = 7
    size_tuos = 5
    def __init__(self, type,name):
        self.type = type
        self.name=name
        if type == 'a4':
            # 打开a4模板
            self.size = 8
            self.size_no = 12
            self.size_tuos = 6
            self.pdf_document = fitz.open('a4.pdf')
            self.page = self.pdf_document.load_page(0)
        if type == 'a3':
            # 打开a3模板
            self.size = 10
            self.size_no = 13
            self.size_tuos = 7
            self.pdf_document = fitz.open('a3.pdf')
            self.page = self.pdf_document.load_page(0)


    def add_image(self, image_path, ratio_w,ratio_h, position=(0, 0)):
        page_rect = self.page.rect
        page_width = page_rect.width  # 页面宽度
        page_height = page_rect.height  # 页面高度
        img = cv2.imread(image_path)

        # 获取图片的宽度和高度
        height, width, channels = img.shape
        if self.type == 'a3':
            position = (167, 65)
            size = (page_height * ratio_h, page_width * ratio_w*0.97)
            #宽度由于模板差异要乘以调整系数
        if self.type == 'a4':
            position = (65, 45)
            size = (page_width * ratio_w, page_height * ratio_h)
            print(size)
        image_rect = fitz.Rect(*position, *(position[0] + size[0], position[1] + size[1]))
        self.page.insert_image(image_rect, filename=image_path)
    # 保存修改后的PDF文件,在调用完所有填写函数后调用
    def save(self):
        self.pdf_document.save(self.name)
        self.pdf_document.close()

# #部分测试
# exm = Photo('a4','123')



# # 加载.npy文件
# image_array = np.load('subject2.npy')
# ratio_wid = 0.675
# ratio_height = 0.665
# # 创建PIL Image对象
# image = Image.fromarray(image_array).convert('RGB')
# if exm.type == 'a3':
#     image = image.rotate(270, expand=True)
# # 保存为JPEG或PNG格式的图片
# image.save('example_image.jpg')
# exm.add_image('example_image.jpg', ratio_wid,ratio_height)
# exm.save()


# # 加载.npy文件
# image_array = np.load('image.npy')
# ratio = 0.5
# # 创建PIL Image对象
# image = Image.fromarray(image_array).convert('RGB')
# if exm.type == 'a3':
#     image = image.rotate(270)
# # 保存为JPEG或PNG格式的图片
# image.save('example_image.jpg')
# exm.add_image('example_image.jpg', ratio)
# exm.save()