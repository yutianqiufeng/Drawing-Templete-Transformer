
from PyQt5.uic.properties import QtWidgets
from model import unet
from NullWriter import NullWriter
from template_transformer import Ui_MainWindow, ImageViewer, convert_result, selectModel, Toast, AboutDialog  # 确保这个路径正确
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QPixmap
import fitz
from PyQt5.QtGui import QImage
from PyQt5.QtCore import Qt
from datetime import datetime

import os
import numpy as np
import torch
from paddleocr import PaddleOCR

import cv2
from PIL import Image
from add_text import Text
from add_photo import Photo
class MyMainForm(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMainForm, self).__init__(parent)
        self.setupUi(self)
        # 覆盖stdout和stderr，以便在没有控制台的环境中避免错误
        if sys.stdout is None:
            sys.stdout = NullWriter()
        if sys.stderr is None:
            sys.stderr = NullWriter()

        ######100mb的模型，效果好，但是必须使用gpu，不然效率太低
        self.ocr = PaddleOCR(rec_model_dir='orcv4_paras/ch_PP-OCRv4_rec_server_infer',
                             det_model_dir='orcv4_paras/ch_PP-OCRv4_det_server_infer',
                             use_angle_cls=True,
                             use_gpu=True)
        ##############

        ######5mb的模型,效果没有前一个好，但是使用cpu可以马上得到结果
        # self.ocr = PaddleOCR(rec_model_dir='orcv4_paras/ch_PP-OCRv4_rec_infer',
        #                 det_model_dir='orcv4_paras/ch_PP-OCRv4_det_infer',
        #                 cls_model_dir='orcv4_paras/ch_ppocr_mobile_v2.0_cls_infer',
        #                 use_angle_cls=True,
        #                 use_gpu=False)
        ##############
        self.file_path_path = None  # 用于存储PDF文件路径
        self.preview_image = None  # 用于存储预览的图像
        self.outPath = os.path.expanduser("~/Documents")  # 系统默认输出地址为文档
        self.result = "转换成功！"
        self.exactPath = None # 实际输出的地址:文件地址+文件名
        self.input_pdf = 'output.pdf' #模板和图片拼接好后的未带基本信息的输出作为push_into_sample_pdf的输入
        self.pdfmodel=None
        self.modelSelected = None  #选择的模板地址
        self.flag = 1  # 设置标志位表示是否提取信息 1否 0是
        self.sizedict={'a3':(1131,1600),'a4':(1600,1131)}
        self.subject=None
        self.actionaddress.triggered.connect(self.set_default_address)  # 连接 triggered 信号与槽函数
        self.about.triggered.connect(self.about_us)  # 连接 triggered 信号与槽函数
        self.actionClose.triggered.connect(self.close)        # 点击上传图片选择文件
        self.label_upload.clicked.connect(self.upload_file)
        self.encoder=torch.load('model_para/encoder.bin')
        # 点击上传图片读取图片并设置到预览标签以缩略图形式体现
        self.label_upload.clicked.connect(self.on_upload_clicked)
        # 点击预览按钮弹出预览窗口
        self.pushButton_view.clicked.connect(self.on_preview_clicked)
        # 点击提取信息按钮将识别的文字填入编辑框
        self.pushButton_extract.clicked.connect(self.pushButton_extract_clicked)
        # 点击确认提交，将信息写入pdf模板中：
        self.pushButton_ok.clicked.connect(self.push_into_sample_pdf)



    '''
    设置->设置默认地址
    '''
    def set_default_address(self):
        folder_selected = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory")
        if folder_selected:
            print("Default address set to:", folder_selected)
            self.outPath = folder_selected


    '''
    帮助->关于
    '''
    def about_us(self):
        self.about = AboutDialog()
        self.about.show()

    '''
    读取图片并设置到预览标签
    '''

    def on_upload_clicked(self):

        image = self.preview_image

        if image is not None:
            # 将 numpy 数组转换为 QImage 对象
            height, width = image.shape
            bytesPerLine = width
            qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_Grayscale8)

            # 将 QImage 转换为 QPixmap，以便在 QLabel 中显示
            pixmap = QPixmap.fromImage(qImg)
            # 显示图片，保持纵横比缩放
            self.label_preview.setPixmap(pixmap.scaled(self.verticalLayoutWidget.size(), Qt.KeepAspectRatio))

    def showToast(self, message):
        toast = Toast(message)
        toast.move(QApplication.desktop().screenGeometry().center() - toast.rect().center())
        toast.show()

    '''
    点击预览后在新窗口中查看图片
    '''

    def on_preview_clicked(self, event):
        #判断是否上传了文件
        if np.all(self.preview_image == None):
            message = "请先上传文件!"
            self.showToast(message)
        else:
            height, width = self.preview_image.shape
            bytesPerLine = width
            qImg = QImage(self.preview_image.data, width, height, bytesPerLine, QImage.Format_Grayscale8)
            # viewer.show() #会闪退
            # viewer.exec_()  # QDialog，使用exec_()会造成其他窗口阻塞
            self.viewer = ImageViewer(qImg)  # 将 viewer 定义为类的属性,这样就不会打开后立马销毁
            self.viewer.show()




    '''
    选择上传pdf或图片文件
    '''

    def load_img(self,path):
        if path.endswith('pdf'):
            doc = fitz.open(path)
            page = doc.load_page(0)
            zoom_x = 4
            zoom_y = 4
            mat = fitz.Matrix(zoom_x, zoom_y)
            pix = page.get_pixmap(matrix=mat)
            image_data = pix.samples
            np_image = np.frombuffer(image_data, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            return np_image
        else:
            img = cv2.imread(path)
            return img

    def resize_img(self,img,size):
        if size == 'a4':
            h = 1600
            w = 1131
        else:
            w = 1600
            h = 1131
        while img.shape[0] // 2 >= 1600:
            img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        return img


    def upload_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "选择文件", "",
                                                   "PDF Files (*.pdf);;Image Files (*.jpg *.png *.jpeg);;All Files (*)",
                                                   options=options)
        if file_path:
            # 选择下一个文件将之前的状态置为初始,当然不止这几个文本框
            self.lineEdit_no.setText('')
            self.lineEdit_dwn.setText('')
            self.lineEdit_chk.setText('')
            self.lineEdit_apvd.setText('')
            self.lineEdit_name.setText('')
            self.lineEdit_material.setText('')
            self.lineEdit_heatTreat.setText('')
            self.lineEdit_scale.setText('')
            self.lineEdit_pev.setText('')
            self.lineEdit_date.setText('')
            self.lineEdit_1.setText('')
            self.lineEdit_2.setText('')
            self.lineEdit_3.setText('')
            self.lineEdit_4.setText('')
            self.preview_image = None  # 用于存储预览的图像
            self.exactPath = None  # 实际输出的地址:文件地址+文件名
            self.pdfmodel = None
            self.modelSelected = None  # 选择的模板地址
            self.flag = 1  # 设置标志位表示是否提取信息 1否 0是

            img = self.load_img(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            print(img.shape)
            h,w=img.shape

            if h>w:
                self.pdfmodel = 'a4'
            else:
                self.pdfmodel = 'a3'

            gray =  self.resize_img(img=img,size=self.pdfmodel)
            self.preview_image=gray.astype('uint8')
            self.selectModel()




    def add_text_to_pdf(self,input_pdf_path, output_pdf_path, no, dwn, chk, apvd, name, material, heat, scale,tuos1,tuos2,tuos3,tuos4):
        adder=Text(self.pdfmodel,input_pdf_path,output_pdf_path)
        adder.DWN_add(dwn)
        adder.APVD_add(apvd)
        adder.CHK_add(chk)
        adder.no_add(no)
        adder.HT_TR_add(heat)
        adder.name_add(name)
        adder.material_add(material)
        adder.material_add(scale)

        adder.tuos_add1(tuos1)
        adder.tuos_add2(tuos2)
        adder.tuos_add3(tuos3)
        adder.tuos_add4(tuos4)
        adder.save()





    '''
       点击确认提交后，调用插入文本函数将用户编辑框的信息插入文本到pdf内,并弹出转换结果
    '''

    def push_into_sample_pdf(self):
        # 检查点击确认转换前有没有上传文件和提取信息
        if np.all(self.preview_image == None)|(self.flag):
            if np.all(self.preview_image == None):
                message = "请先上传文件!"
            else:
                message = "请先提取信息!"
            self.showToast(message)
        else:
            no = self.lineEdit_no.text()
            name = self.lineEdit_name.text()
            material=self.lineEdit_material.text()
            heat=self.lineEdit_heatTreat.text()
            scale=self.lineEdit_scale.text()
            dwn = self.lineEdit_dwn.text()
            chk = self.lineEdit_chk.text()
            apvd = self.lineEdit_apvd.text()
            tuos1=self.lineEdit_1.text()
            tuos2 = self.lineEdit_2.text()
            tuos3 = self.lineEdit_3.text()
            tuos4 = self.lineEdit_4.text()


            self.add_pic(self.subject,'output.pdf')
            self.exactPath = os.path.join(self.outPath,self.get_unique_filename(self.outPath, self.lineEdit_no.text() + '.pdf'))
            print(self.exactPath)




            try:
                self.add_text_to_pdf('output.pdf',self.exactPath, no, dwn, chk, apvd, name, material, heat, scale,tuos1,tuos2,tuos3,tuos4)
                self.result = "Succeed！"
            except Exception as e:
                print(e)
                self.result = "Fail！"
            self.on_ok_clicked_result()



    '''
    将得到的文字信息填入编辑框
    '''
    def add_info_to_editText(self, no, dwn, chk, apvd, name, material, heat, scale):

        def set_text_with_color(line_edit, text, color_if_match='red'):
            if text in ('请输入', '请输入,以n:m形式'):
                line_edit.setStyleSheet(f"QLineEdit {{ color: {color_if_match}; }}")
            else:
                line_edit.setStyleSheet("")  # 清除样式以使用默认颜色
            line_edit.setText(text)

        set_text_with_color(self.lineEdit_no, no)
        set_text_with_color(self.lineEdit_dwn, dwn)
        set_text_with_color(self.lineEdit_chk, chk)
        set_text_with_color(self.lineEdit_apvd, apvd)
        set_text_with_color(self.lineEdit_name, name)
        set_text_with_color(self.lineEdit_material, material)
        set_text_with_color(self.lineEdit_scale, scale)
        set_text_with_color(self.lineEdit_heatTreat, heat)
        self.lineEdit_pev.setText('0')
        # 获取当前时间
        now = datetime.now()
        # 格式化当前时间为 'YYYY/MM/DD' 格式
        formatted_date = now.strftime('%Y/%m/%d')
        # 设置 lineEdit_date 的文本为当前时间
        self.lineEdit_date.setText(formatted_date)
        self.lineEdit_1.setText('0.1')
        self.lineEdit_2.setText('0.2')
        self.lineEdit_3.setText('0.5')
        self.lineEdit_4.setText('1')

    def get_segamentation(self,model,model_path,img,re_ratio=False,use_encoder=False):

        checkpoint = self.encoder.copy()
        decoder=torch.load(model_path)
        checkpoint.update(decoder)
        model.load_state_dict(checkpoint)
        model.to('cuda')
        model.eval()
        with torch.no_grad():
            if use_encoder:
                img=model.encoder(img)
            mask = model(img,extract_features=False)
        mask = mask.cpu().detach().numpy()
        h,w=self.preview_image.shape
        label = mask[0, 0, ...]
        label = cv2.resize(label, (w, h), interpolation=cv2.INTER_AREA)
        label[label >= 0.5] = 1
        label[label <= 0.5] = 0
        mixed = label * self.preview_image
        mixed[label == 0] = 255
        for i1 in range(label.shape[0]):
            if label[i1, :].sum() != 0:
                break
        for i2 in range(label.shape[0] - 1, 0, -1):
            if label[i2, :].sum() != 0:
                break
        for j1 in range(label.shape[1]):
            if label[:, j1].sum() != 0:
                break
        for j2 in range(label.shape[1] - 1, 0, -1):
            if label[:, j2].sum() != 0:
                break
        mixed = mixed[i1:i2, j1:j2]
        if use_encoder:
            mixed=(mixed,img)
        if re_ratio:
            ratio = ((i2 - i1) / label.shape[0], (j2 - j1) / label.shape[1])
            return mixed,ratio
        else:
            return mixed.astype(np.uint8)
    '''
    点击提取信息后触发add_info_to_editText
    '''
    def get_data(self):
        img = cv2.resize(self.preview_image, (800, 800), interpolation=cv2.INTER_AREA)
        img = img[np.newaxis, np.newaxis, ...]
        img = torch.tensor(img, dtype=torch.float)
        img = img.to('cuda')
        model = unet.unet(encoder_name= "efficientnet-b7")


        data,ratio=self.get_segamentation(model,'model_para/subject.bin',img.clone(),True,True)
        subject,img=data
        material = self.get_segamentation(model, 'model_para/material.bin', img.copy())
        name = self.get_segamentation(model, 'model_para/name.bin', img.copy())
        no = self.get_segamentation(model, 'model_para/no.bin', img.copy())
        scale = self.get_segamentation(model, 'model_para/scale.bin', img.copy())
        heat = self.get_segamentation(model, 'model_para/heat.bin', img.copy())
        seg_dict={'subject':(subject,ratio),'material':material,'name':name,'no':no,'scale':scale,'heat':heat}

        return seg_dict



    '''
    如果模板识别失败,弹出选择模板
    '''
    def selectModel(self):
        self.dialog = selectModel(self.pdfmodel)
        self.dialog.show()
        self.dialog.modelSelected.connect(self.onModelSelected)

    def onModelSelected(self, model_path):
        self.pdfmodel = model_path


    def add_pic(self,subject_tuple,name):
        adder = Photo(self.pdfmodel, name)
        subject,ratio=subject_tuple
        ratio_height,ratio_wid = ratio
        print(ratio)
        image = Image.fromarray(subject).convert('RGB')
        if adder.type == 'a3':
            image = image.rotate(270, expand=True)
        image.save('subject.jpg')
        adder.add_image('subject.jpg', ratio_wid, ratio_height)
        adder.save()


    def extract_text_from_image(self,image,det=True):
        result = self.ocr.ocr(image, det=det)[0]
        if det==False:
            if result[0] == None:
                return '-'
            return result[0][0]
        if result[0] == None:
            return ['-',]
        content = [x[1][0] for x in result if x[1][1] > 0.8]
        return content

    def get_name(self,image):
        text=self.extract_text_from_image(image,det=False)
        return text

    def get_material(self,image):
        text=self.extract_text_from_image(image,det=False)
        return text

    def get_heat(self,image):
        text=self.extract_text_from_image(image)
        if len(text) == 2:
            return text[1]
        elif len(text) == 1:
            return text[0]
        else:
            return '请输入'

    def get_scale(self,image):
        text=self.extract_text_from_image(image)
        if len(text) == 2:
            return text[1]
        elif len(text) == 1:
            return text[0]
        else:
            return '请输入'
    def get_no(self,image):
        text=self.extract_text_from_image(image,det=False)
        return text
    '''     
    点击提取信息后触发add_info_to_editText
    '''
    def pushButton_extract_clicked(self):
        self.flag=0
        # 检查点击提取信息前有没有上传文件
        if np.all(self.preview_image == None):
            message = "请先上传文件!"
            self.showToast(message)
        else:

            mask=self.get_data()
            self.subject=mask['subject']



            name=self.get_name(mask['name'])
            # name=name.replace("'",'_').replace('"','_')


            scale=self.get_scale(mask['scale'])

            material=self.get_material(mask['material'])
            no=self.get_no(mask['no'])
            heat = self.get_heat(mask['heat'])


            dwn = "Jason liu"
            chk = "zhulei Pan"
            apvd = "Tiger guan"

            # dwn = "Input From System"
            # chk = "Input From System"
            # apvd = "Input From System"


            self.add_info_to_editText(no, dwn, chk, apvd, name, material, heat, scale)

    '''
        输出文件命名问题,编号.pdf,检查当前目录有无同样编号的文件,有则按序号标明
    '''

    def get_unique_filename(self, folder_path, base_name):
        file_name, file_extension = os.path.splitext(base_name)
        counter = 1
        unique_name = base_name
        while os.path.exists(os.path.join(folder_path, unique_name)):
            unique_name = f"{file_name} ({counter}){file_extension}"
            counter += 1
        return unique_name


    '''
    点击确认转换后弹出转换结果，传入默认地址
    '''
    def on_ok_clicked_result(self):
        viewer = convert_result(self.result,self.outPath)
        viewer.exec_()







if __name__ == "__main__":
    app = QApplication(sys.argv)  # 创建一个应用程序对象
    myWin = MyMainForm()  # 创建窗口
    myWin.show()  # 显示窗口
    sys.exit(app.exec_())  # 运行应用程序，并在退出时清理
