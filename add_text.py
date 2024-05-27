import fitz

class Text:
    # 类属性，默认为a3模板的属性
    type = ''
    color = (0, 0, 0)
    size = 7
    size_no = 7
    size_tuos = 5
    def __init__(self, type,origin_path,output_path):
        self.type = type

        if type == 'a4':
            # 打开a4模板
            self.size = 8
            self.size_no = 12
            self.size_tuos = 6

        if type == 'a3':
            # 打开a3模板
            self.size = 10
            self.size_no = 13
            self.size_tuos = 7
        self.pdf_document = fitz.open(origin_path)
        self.page = self.pdf_document.load_page(0)
        self.output_path=output_path


    def DWN_add(self, DWN):
        if self.type == 'a4':
            self.page.insert_text((220, 686), DWN, fontsize=self.size, fontname="Times-Roman", color=self.color)
        if self.type == 'a3':
            self.page.insert_text((146, 778), DWN, fontsize=self.size, rotate=270, fontname="Times-Roman", color=self.color)

    def CHK_add(self, CHK):
        if self.type == 'a4':
            self.page.insert_text((338, 686), CHK, fontsize=self.size, fontname="Times-Roman", color=self.color)
        if self.type == 'a3':
            self.page.insert_text((119, 778), CHK, fontsize=self.size, rotate=270, fontname="Times-Roman", color=self.color)

    def APVD_add(self, APVD):
        if self.type == 'a4':
            self.page.insert_text((456, 686), APVD, fontsize=self.size, fontname="Times-Roman", color=self.color)
        if self.type == 'a3':
            self.page.insert_text((92, 778), APVD, fontsize=self.size, rotate=270, fontname="Times-Roman", color=self.color)

    # 没有输入请使用默认参数
    def material_add(self, material= "none"):
        if self.type == 'a4':
            if material=="none":
                self.page.draw_line((415.9, 708.1), (420.1, 708.1), color=self.color, width=0.1)
            else:
                self.page.insert_text((408, 711), material, fontsize=self.size, fontname="Times-Roman",color=self.color)
        if self.type == 'a3':
            if material == "none":
                self.page.draw_line((139.6, 993.4), (139.6, 997.7), color=self.color, width=0.1)
            else:
                self.page.insert_text((148, 985), material, fontsize=self.size, rotate=270, fontname="Times-Roman", color=self.color)
    #没有输入请使用默认参数
    def HT_TR_add(self, HT_TR = "none"):
        if self.type == 'a4':
            if HT_TR=="none":
                self.page.draw_line((517.2, 708.1), (521.6, 708.1), color=self.color, width=0.1)
            else:
                self.page.insert_text((513, 711), HT_TR, fontsize=self.size, fontname="Times-Roman", color=self.color)
        if self.type == 'a3':
            if HT_TR == "none":
                self.page.draw_line((139.6, 1099.7), (139.6, 1104), color=self.color, width=0.1)
            else:
                self.page.insert_text((148, 1081), HT_TR, fontsize=self.size, rotate=270, fontname="Times-Roman", color=self.color)

    def no_add(self, no):
        if self.type == 'a4':
            self.page.insert_text((310, 809), no, fontsize=self.size, fontname="Times-Roman", color=self.color)
            self.page.insert_text((53, 138), no, fontsize=self.size, rotate=90,  fontname="Times-Roman", color=self.color)
        if self.type == 'a3':
            self.page.insert_text((39, 873), no, fontsize=self.size_no, rotate=270, fontname="Times-Roman", color=self.color)
            self.page.insert_text((676, 60), no, fontsize=(self.size_no-2),  fontname="Times-Roman", color=self.color)

    def rev_add(self, rev):
        if self.type == 'a4':
            self.page.insert_text((553, 808), rev, fontsize=self.size_no, fontname="Times-Roman", color=self.color)
        if self.type == 'a3':
            self.page.insert_text((40, 1128), rev, fontsize=self.size, rotate=270, fontname="Times-Roman", color=self.color)

    #标准差
    def tuos_add0(self, plc):
        if self.type == 'a4':
            self.page.insert_text((342, 749), plc, fontsize=self.size_tuos, fontname="Times-Roman", color=self.color)
        if self.type == 'a3':
            self.page.insert_text((101, 730), plc, fontsize=self.size_tuos, rotate=270, fontname="Times-Roman", color=self.color)
    def tuos_add1(self, plc):
        if self.type == 'a4':
            self.page.insert_text((342, 756), plc, fontsize=self.size_tuos, fontname="Times-Roman", color=self.color)
        if self.type == 'a3':
            self.page.insert_text((92, 730), plc, fontsize=self.size_tuos, rotate=270, fontname="Times-Roman", color=self.color)
    def tuos_add2(self, plc):
        if self.type == 'a4':
            self.page.insert_text((342, 760), plc, fontsize=self.size_tuos, fontname="Times-Roman", color=self.color)
        if self.type == 'a3':
            self.page.insert_text((83.5, 730), plc, fontsize=self.size_tuos, rotate=270, fontname="Times-Roman", color=self.color)
    def tuos_add3(self, plc):
        if self.type == 'a4':
            self.page.insert_text((342.2, 771), plc, fontsize=self.size_tuos, fontname="Times-Roman", color=self.color)
        if self.type == 'a3':
            self.page.insert_text((75, 730), plc, fontsize=self.size_tuos, rotate=270, fontname="Times-Roman", color=self.color)

    def tuos_add4(self, plc):
        if self.type == 'a4':
            self.page.insert_text((342, 778), plc, fontsize=self.size_tuos, fontname="Times-Roman", color=self.color)
        if self.type == 'a3':
            self.page.insert_text((67, 730), plc, fontsize=self.size_tuos, rotate=270, fontname="Times-Roman", color=self.color)

    def scale_add(self, scale):
        if self.type == 'a4':
            self.page.insert_text((234.5, 809), scale, fontsize=self.size_no, fontname="Times-Roman",color=self.color)
        if self.type == 'a3':
                self.page.insert_text((39, 790), scale, fontsize=self.size_no, rotate=270, fontname="Times-Roman",color=self.color)

    # 没有输入请使用默认参数
    def name_add(self, name="none"):
        if self.type == 'a4':
            if name == "none":
                self.page.draw_line((474, 758), (480, 758), color=self.color, width=0.1)
            else:
                self.page.insert_text((382, 770), name, fontsize=15, fontname="Times-Roman", color=self.color)
        if self.type == 'a3':
            if name == "none":
                self.page.draw_line((66, 963), (66, 968.3), color=self.color, width=0.1)
            else:
                self.page.insert_text((72, 839), name, fontsize=self.size, rotate=270, fontname="Times-Roman", color=self.color)
    # 保存修改后的PDF文件,在调用完所有填写函数后调用
    def save(self):
        self.pdf_document.save(self.output_path)
        self.pdf_document.close()

# # #部分测试
# exm = Text('a4','123.pdf','123.pdf')
#
# exm.DWN_add("SUPPLIER")
# exm.APVD_add("SUPPLIER")
# exm.material_add()
# exm.HT_TR_add()
# exm.no_add("123")
# exm.tuos_add0("1")
# exm.tuos_add1("1")
# exm.tuos_add2("1")
# exm.tuos_add3("1")
# exm.tuos_add4("1")
# exm.name_add()
# exm.save()