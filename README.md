# Templete-Transformer
首先需要下载两个模型的权重文件，使用以下两个百度云链接下载，直接解压到项目的主文件夹

链接：https://pan.baidu.com/s/1hVXsR-8fkPgtvkPoYmTaMw?pwd=10le 
提取码：10le

链接：https://pan.baidu.com/s/17cANStYyHXiupYXUv-wnXw?pwd=ujur 
提取码：ujur

运行主文件夹下的call_transformerUI.py即可运行项目。

环境说明：
需要依赖的库参照environment.yml文件

由于项目同时使用了paddle和torch环境，如果同时使用高版本的torch和paddle会出现报错。
在我们的测试中torch 1.10.2+cu102 ， paddleocr 2.7.3 和 paddlepaddle 2.6.1可以成功运行，使用的显卡是rtx 1650ti。

此外，我们准备了cpu版本，即torch使用gpu，paddle只使用cpu,需要安装paddle的cpu版本，并且修改call_transformerUI.py中33~46行的代码，注释掉gpu的paddle对象，取消cpu部分的注释。

![WPS图片(1)](https://github.com/user-attachments/assets/cb6a59ba-2a4b-4308-8a39-49b388a3efb0)


