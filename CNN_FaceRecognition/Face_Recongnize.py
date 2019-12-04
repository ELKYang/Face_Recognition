import numpy as np
import tensorflow as tf
import glob
import os
import sys
from ctypes import *
import types
import cv2
from skimage import io
import tkinter as tk
from tkinter.filedialog import askopenfilename
from PIL import Image,ImageTk

#创建画布
root=tk.Tk()
root.geometry('800x400+400+100')
root.title('Face_Recongnize')
canvas = tk.Canvas(root, bg='white', width=400, height=400)
canvas.pack(side='right')
#由于convas中需要使用全局变量，这里被迫使用这两个图片预定义两个全局的image对象变量
im=Image.open('CNN_Face_Recognition\A0.png')
img=ImageTk.PhotoImage(im)
im2=Image.open('CNN_Face_Recognition\A0.png')
img2=ImageTk.PhotoImage(im2)
cate=np.load('CNN_Face_Recognition\Faces_Last-1.output\path_num.npy')

#图片识别
def Image_Identify():
    global img
    global cate
    global Student_num
    img_path = tk.filedialog.askopenfilename(title=u'Please choose a picture',filetypes=[("人脸图片", "*.png")],initialdir=(os.path.expanduser('C:/')))
    img_val=io.imread(img_path)#实际使用的图片
    im=Image.open(img_path)
    img = im.resize((300, 300),Image.ANTIALIAS)
    img=ImageTk.PhotoImage(img)#显示的图片
    canvas.create_image(200,200,image = img) 
    #将图片放入网络中
    sess = tf.Session()
    saver = tf.compat.v1.train.import_meta_graph('CNN_Face_Recognition\Faces_Last-1.output\log\model-52.meta')
    saver.restore(sess, tf.train.latest_checkpoint('CNN_Face_Recognition\Faces_Last-1.output\log'))
    logits=tf.get_collection('pred_network')[0]
    predict_=tf.nn.softmax(logits)
    predict = tf.argmax(predict_, 1)
    graph = tf.compat.v1.get_default_graph()
    data = graph.get_tensor_by_name('input/Input_data:0')
    is_training=graph.get_tensor_by_name('Placeholder:0')
    res=sess.run(predict, feed_dict={data:[img_val],is_training:False})
    Result_Text.config(state='normal')
    Result_Text.delete('1.0','end')
    Result_Text.insert('insert',cate[res[0]][-7:])#显示学号
    Result_Text.config(state='disabled')

def Instant_photo_recognition():
    Result_Text.config(state='normal')
    Result_Text.delete('1.0','end')
    global img2
    sess = tf.Session()
    saver = tf.compat.v1.train.import_meta_graph('CNN_Face_Recognition\Faces_Last-1.output\log\model-52.meta')
    saver.restore(sess, tf.train.latest_checkpoint('CNN_Face_Recognition\Faces_Last-1.output\log'))
    logits=tf.get_collection('pred_network')[0]
    predict_=tf.nn.softmax(logits)
    predict = tf.argmax(predict_, 1)
    graph = tf.compat.v1.get_default_graph()
    data = graph.get_tensor_by_name('input/Input_data:0')
    is_training=graph.get_tensor_by_name('Placeholder:0')
    #开启摄像头处理部分
    cap = cv2.VideoCapture(0)
    Max_Faces = 256
    Size_one_Face = 6 + 68 * 2
    Size_FaceLandMarks = Size_one_Face * Max_Faces
    class FaceResults(Structure): 
        _fields_ = [("face_num", c_int32), ("datas", c_int16 * Size_FaceLandMarks)]
    dll = CDLL('./dlls/fd-shiqiyu_v2.dll')#调用于仕琪人脸检测的python接口
    dll.shiqi_fd.restype = POINTER(FaceResults)
    ret, img = cap.read()
    h,w = img.shape[:2]
    st = w*3
    p_img = img.ctypes.data_as(POINTER(c_ubyte))
    p_results = dll.shiqi_fd(p_img,w,h,st) 
    face_num = p_results.contents.face_num
    #对检测到的人脸画框
    for i in range(face_num):
        j =  Size_one_Face * i
        x = p_results.contents.datas[j]
        y = p_results.contents.datas[j+1]
        w = p_results.contents.datas[j+2]
        h = p_results.contents.datas[j+3]
        confidence = p_results.contents.datas[j+4]
        angle = p_results.contents.datas[j+5]
        x1=int(x-1*w/5)
        y1=int(y-h/2)
        x2=int(x+6*w/5)
        y2=y+h
        box=(x1,y1,x2,y2)
        cv2.rectangle(img,(x1,y1) ,(x2,y2), (0,255,0), 1)
        face_img=img[y1:y2,x1:x2]
        res=cv2.resize(face_img,(128,128),interpolation=cv2.INTER_CUBIC)
        res_num=sess.run(predict, feed_dict={data:[res],is_training:False})
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,cate[res_num[0]][-7:],(x, y-10),font, 1,(0,0,255), 1, cv2.LINE_AA)
        Result_Text.insert('insert',cate[res_num[0]][-7:]+' ')
    cap.release()
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_display=Image.fromarray(img)
    img2= img_display.resize((300, 300),Image.ANTIALIAS)
    img2=ImageTk.PhotoImage(img2)#显示的图片
    canvas.create_image(200,200,image = img2)
    Result_Text.config(state='disabled')


def Instant_video_recognition():
    global img2
    sess = tf.Session()
    saver = tf.compat.v1.train.import_meta_graph('CNN_Face_Recognition\Faces_Last-1.output\log\model-52.meta')
    saver.restore(sess, tf.train.latest_checkpoint('CNN_Face_Recognition\Faces_Last-1.output\log'))
    logits=tf.get_collection('pred_network')[0]
    predict_=tf.nn.softmax(logits)
    predict = tf.argmax(predict_, 1)
    graph = tf.compat.v1.get_default_graph()
    data = graph.get_tensor_by_name('input/Input_data:0')
    is_training=graph.get_tensor_by_name('Placeholder:0')
    
    cap = cv2.VideoCapture(0)
    Max_Faces = 256
    Size_one_Face = 6 + 68 * 2
    Size_FaceLandMarks = Size_one_Face * Max_Faces
    class FaceResults(Structure): 
        _fields_ = [("face_num", c_int32), ("datas", c_int16 * Size_FaceLandMarks)]
    dll = CDLL('./dlls/fd-shiqiyu_v2.dll')
    dll.shiqi_fd.restype = POINTER(FaceResults)
    #这里其实就是循环图片检测
    while True:
        Result_Text.delete('1.0','end')
        ret, img = cap.read()
        h,w = img.shape[:2]
        st = w*3
        p_img = img.ctypes.data_as(POINTER(c_ubyte))
        p_results = dll.shiqi_fd(p_img,w,h,st) 
        face_num = p_results.contents.face_num
        for i in range(face_num):
            j =  Size_one_Face * i
            x = p_results.contents.datas[j]
            y = p_results.contents.datas[j+1]
            w = p_results.contents.datas[j+2]
            h = p_results.contents.datas[j+3]
            confidence = p_results.contents.datas[j+4]
            angle = p_results.contents.datas[j+5]
            x1=int(x-1*w/5)
            y1=int(y-h/2)
            x2=int(x+6*w/5)
            y2=y+h
            box=(x1,y1,x2,y2)
            cv2.rectangle(img,(x1,y1) ,(x2,y2), (0,255,0), 1)
            face_img=img[y1:y2,x1:x2]
            res=cv2.resize(face_img,(128,128),interpolation=cv2.INTER_CUBIC)
            res_num=sess.run(predict, feed_dict={data:[res],is_training:False})
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img,cate[res_num[0]][-7:],(x, y-10),font, 1,(0,0,255), 1, cv2.LINE_AA)
            Result_Text.insert('insert',cate[res_num[0]][-7:]+' ')
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img_display=Image.fromarray(img)
        img2= img_display.resize((300, 300),Image.ANTIALIAS)
        img2=ImageTk.PhotoImage(img2)#显示的图片
        canvas.create_image(200,200,image = img2)
        root.update_idletasks()
        root.update()
        cv2.waitKey(100)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    
Image_Identify_button=tk.Button(root,text="图片文件识别",command=Image_Identify).place(x=60,y=45)
Instant_photo_recognition_button=tk.Button(root,text="实时拍照识别",command=Instant_photo_recognition).place(x=60,y=95)
Instant_video_recognition_button=tk.Button(root,text="实时视频识别",command=Instant_video_recognition).place(x=60,y=145)
Result_Text=tk.Text(root,height=1,width=30,font=('StSong', 14),foreground='gray')
Result_Text.place(x=58,y=250)
Label1=tk.Label(root,text='辨认结果:').place(x=0,y=250)
Label2=tk.Label(root,text='姓名：杨帆').place(x=0,y=335)
Label3=tk.Label(root,text='学号：1711503').place(x=0,y=355)
Label4=tk.Label(root,text='专业：智能科学与技术').place(x=0,y=375)
root.mainloop()