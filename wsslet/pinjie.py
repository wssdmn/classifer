# import PyPDF2
# import os

# PDF_file_list = ['1.pdf', '2.pdf', '3.pdf']

# PDF = []
# for filename in PDF_file_list:
#     pdfFileObj = open(filename, 'rb')
#     pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
#     for pageNum in range(pdfReader.numPages):
#         pageObj = pdfReader.getPage(pageNum)
#         PDF.append(pageObj)
#     pdfFileObj.close()

# pdfWriter = PyPDF2.PdfFileWriter()
# for page in PDF:
#     pdfWriter.addPage(page)
    
# with open('Merged.pdf', 'wb') as merged_file:
#     pdfWriter.write(merged_file)


from PyPDF2 import PdfFileMerger
import os

pdfRoot = "D:\code\Letcode" #保存pdf结果的文件夹
merger = PdfFileMerger() #调用PDF文件合并模块
#filelist=os.listdir(pdfRoot) #读取文件夹所有文件
filelist=['1目录.pdf', '2职称证书.pdf','3论文.pdf', '4项目结题.pdf','5证书.pdf', '8专利.pdf']
#filelist=[ '21.pdf','122.pdf', ]
for file in filelist:
    if file.endswith(".pdf") :
        merger.append(pdfRoot+"\\"+file)#合并PDF文件

merger.write("杨树强支撑材料.pdf") #写入PDF文件
