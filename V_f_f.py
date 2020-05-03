# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:48:31 2020

@author: VahidSamaee
"""
import numpy as np
import cv2
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
from tkinter import *
from tkinter import filedialog

sys.path.append('C:\\Users\\VahidSamaee\\opencv')   



def path_extractore(V_path):
    text_len=len (V_path)
    presence=True
    location=0
    while presence:
        try: #(the bunch of codes that you want python to run it and it might have some errors)
            V_path.index('/',text_len-location,text_len)
        except: #(if an error happens in the try part, then python will run this part )
            location+=1
            continue
        else: # when there is no error_
            print(V_path[0:text_len-location+1])
            V_direct=V_path[0:text_len-location+1]
            print (V_path)
            break
    return V_direct



def V_framing (V_path):
    count=1
    cap=cv2.VideoCapture(V_path)
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            print('Read %d frame: ' % count, ret)
            # save frame as JPEG file
            V_direct= path_extractore (V_path)
            cv2.imwrite(os.path.join(V_direct, "frame{:d}.jpg".format(count)), frame)
            count += 1
        else:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return True



def VTMPTP(img,template): # template matchig 
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt
    img2 = img.copy()
    w, h = template.shape[::-1]

    # All the 6 methods for comparison in a list

    # methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 
    # 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    img = img2.copy()
    method = cv2.TM_SQDIFF_NORMED

        # Apply template Matching
    res = cv2.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    #print(min_val, max_val, min_loc, max_loc)
    return min_val, max_val, min_loc, max_loc



def V_image_portion (img,img2,V_cordinat):

    y_drift=V_cordinat[2][1]
    x_drift=V_cordinat[2][0]
    return img[(0+y_drift):(img2.shape[0]+y_drift),(0+x_drift):(img2.shape[1]+x_drift)]



def V_alfaBlendingPlusShift(img,img2,V_cordinat):
    y_drift=V_cordinat[2][1]
    x_drift=V_cordinat[2][0]
    print (y_drift, x_drift)
    img_2=img[(0+y_drift):(img2.shape[0]+y_drift),(0+x_drift):(img2.shape[1]+x_drift)]
    # alpha blending
    alpha = 0.5
    img3 = np.uint8(img_2*alpha + img2*(1-alpha))
    cv2.imshow('Alfa Blending', img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img3



def V_initial_lenght (V_path):
    global img, drawing_Line_1_fact, drawing_Line_2_fact 
    drawing_Line_1_fact=False
    drawing_Line_2_fact=False

    # mouse callback function 
    def draw_circle(event,x,y,flags,param):
        global img, y1, y2, drawing_Line_1_fact, drawing_Line_2_fact 
        if not drawing_Line_1_fact:
            if flags==1:
                img = cv2.imread(V_path,0)
                cv2.line(img, (0,y), (ximag,y), (125,125,125), 2)
                y1=y
            if event == 4 and flags == 0:
                drawing_Line_1_fact=True
        elif not drawing_Line_2_fact:
            if flags==1:
                img = cv2.imread(V_path,0)
                cv2.line(img, (0,y1), (ximag,y1), (125,125,125), 2)
                cv2.line(img, (0,y), (ximag,y), (125,125,125), 2)
                y2=y
            if event == 4 and flags == 0:
                drawing_Line_2_fact=True

    img = cv2.imread(V_path,0)
    ximag=img.shape[0]
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_circle)
    #cv2.setMouseCallback('image',print)

    while(1):
        cv2.imshow('image',img)
        if cv2.waitKey(20) & 0xFF == 27:
            break
        
    cv2.destroyAllWindows()
    return abs(y1-y2)



def V_selecting_a_region_on_image (V_path, scaling=0.8):
    global img, endoffunction,y1, y2, x1, x2
    endoffunction=False
    # mouse callback function
    def draw_rectangle(event,x,y,flags,param):
        global img, y1, y2, x1, x2,endoffunction
        if event==1 and flags==1:
            x1=x
            y1=y
        if event==0 and flags==1:
            x2=x
            y2=y
            img = cv2.imread(V_path,0)
            #img=cv2.resize(img, (int (img.shape[0]*scaling),int(img.shape[1]*scaling)))  
            cv2.rectangle(img, (x1,y1), (x2,y2), (125,125,125), 2)
        if event == 4 and flags == 0:
            x2=x
            y2=y
            img = cv2.imread(V_path,0)
            #img=cv2.resize(img, (int (img.shape[0]*scaling),int(img.shape[1]*scaling)))  
            cv2.rectangle(img, (x1,y1), (x2,y2), (125,125,125), 2)
            endoffunction=True
            
    img = cv2.imread(V_path,0)
    #img=cv2.resize(img, (round(img.shape[0]*scaling),round(img.shape[1]*scaling)),interpolation = cv2.INTER_AREA)
    cv2.namedWindow('select the region')
    cv2.setMouseCallback('select the region',draw_rectangle)
    #cv2.setMouseCallback('image',print)

    while(1):
        cv2.imshow('select the region',img)
        if (cv2.waitKey(20) & 0xFF == 27) or endoffunction:
            break
    cv2.destroyAllWindows()
    # return round(x1/scaling), round(y1/scaling), round(x2/scaling), round(y2/scaling)
    return x1,y1,x2,y2  



def movie_info (V_path):
    import sys
    sys.path
    sys.path.append('C:\\Users\\VahidSamaee\\opencv')
    #print(sys.path)

    # to get the video file
    V_direct=path_extractore(V_path)
  
    #reading the movie and frame rate and Number of Frames
    cap=cv2.VideoCapture(V_path)
    frame_rate=cap.get(cv2.CAP_PROP_FPS)
    V_N_Frame=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    cv2.destroyAllWindows()

    # save all the frames of the movie the directory of the movie file

    return (frame_rate, V_N_Frame)



def applying_FFT_bluring_filter(image, treshold,fft_tresh):

    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    from skimage import data
    from skimage.feature import register_translation
    from skimage.feature.register_translation import _upsampled_dft
    from scipy.ndimage import fourier_shift
    import sys
    import tkinter


    sys.path
    sys.path.append('C:\\Users\\VahidSamaee\\opencv')

    rows, cols = image.shape
    # to see how it changes by applying the bluring and also treshold 
    image = cv2.GaussianBlur(image,(5,5),0)
    (thresh, image) = cv2.threshold(image, treshold, 255, cv2.THRESH_BINARY)

    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))


    crow,ccol = round(rows/2) , round(cols/2)
    print (crow,ccol)
    fshift[(crow-fft_tresh):(crow+fft_tresh), (ccol-fft_tresh):(ccol+fft_tresh)] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    magnitude_spectrum2 = 20*np.log(np.abs(fshift))
    img_back = np.abs(img_back)

    plt.subplot(221),plt.imshow(image, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(222),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
    plt.subplot(223),plt.imshow(img_back, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(224),plt.imshow(magnitude_spectrum2, cmap = 'gray')
    plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
    plt.show()



def cross_correlation(V_direct, x1,y1,x2,y2,xx1,yy1,xx2,yy2,fft_treshold,
 bw_treshold,firstframe,lastframe,step, selected_frame):

    import numpy as np
    import tkinter
    import tkinter.filedialog
    import cv2
    import sys
    from matplotlib import pyplot as plt
    import time
    from skimage.feature import register_translation
    from skimage.feature.register_translation import _upsampled_dft


    def ImageWith_tectangeAndcircle(img,V_template,x_displacement,y_displacement):
        V_template_width= V_template.shape[0]
        V_template_hight= V_template.shape[1]                       
        cv2.rectangle(img, (x_displacement,y_displacement), 
        (x_displacement+V_template_hight ,y_displacement+V_template_width),
         (125,125,125), 2)
        return img


    def V_distacebetween_2points (xx1,yy1,x1, y1):
        return ((xx1-x1)**2 + (yy1-y1)**2)**(0.5)


    def Applying_bluring_BW(img):
        img=cv2.GaussianBlur(img,(5,5),0)
        (thresh, img) = cv2.threshold(img, bw_treshold, 255, cv2.THRESH_BINARY)
        return img


    def Applying_fft_filter(img):
        rows, cols = img.shape
        crow,ccol = round(rows/2) , round(cols/2)
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        fshift[(crow-fft_treshold):(crow+fft_treshold),(ccol-fft_treshold):(ccol+fft_treshold)]=0
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        return img_back


#####<<<<< this part first template macht the first frame areas with i'th 
##### frame, after finding the areas, it measure the sub pixel movement of areas
##### by register traslation >>>>>#####

    if x1==max(x1,x2):
        x1, x2=x2, x1
    if y1==max(y1,y2):
        y1, y2=y2, y1

    x_dimention= abs(x2-x1)
    y_dimention= abs(y2-y1)

    if xx1==max(xx1,xx2):
        xx1, xx2=xx2, xx1
    if yy1==max(yy1,yy2):
        yy1, yy2=yy2, yy1

    xx_dimention= abs(xx2-xx1)
    yy_dimention= abs(yy2-yy1)
   
    img = cv2.imread(V_direct+'frame'+str(selected_frame)+'.jpg',0) #In the frame which user select
    img=Applying_bluring_BW(img)
    V_0_template = img[int(y1):int(y1)+y_dimention,int(x1):int(x1)+x_dimention]  
    V_0_template2 = img[int(yy1):int(yy1)+yy_dimention,int(xx1):int(xx1)+xx_dimention]

    img=Applying_fft_filter(img)
    V_0_fft_template = img[int(y1):int(y1)+y_dimention,int(x1):int(x1)+x_dimention]  
    V_0_fft_template2 = img[int(yy1):int(yy1)+yy_dimention,int(xx1):int(xx1)+xx_dimention]   
    
    for i in range (firstframe,lastframe,step):








        # if i==firstframe:
        #     img_i_1 = cv2.imread(V_direct+'frame'+str(selected_frame)+'.jpg',0)
        # else:
        #     img_i_1 = cv2.imread(V_direct+'frame'+str(i-step)+'.jpg',0)
        # img=Applying_bluring_BW(img_i_1)
        
        # V_cordinat1=VTMPTP(img,V_0_template)                 
        # x1=V_cordinat1[2][0]
        # y1=V_cordinat1[2][1]
        # V_cordinat1=VTMPTP(img,V_0_template2)                    
        # xx1=V_cordinat1[2][0]
        # yy1=V_cordinat1[2][1]

        # img=Applying_fft_filter(img)

        # V_i_1_template = img[int(y1):int(y1)+y_dimention,int(x1):int(x1)+x_dimention]  
        # V_i_1_template2 = img[int(yy1):int(yy1)+yy_dimention,int(xx1):int(xx1)+xx_dimention]


        # ##### extraxting template from image i
        # img_i = cv2.imread(V_direct+'frame'+str(i)+'.jpg',0)
        # img=Applying_bluring_BW(img_i)
        # img=Applying_fft_filter(img)

        # V_i_template = img[int(y1):int(y1)+y_dimention,int(x1):int(x1)+x_dimention] 
        # V_i_template2 = img[int(yy1):int(yy1)+yy_dimention,int(xx1):int(xx1)+xx_dimention]














        if i==firstframe:
                    
            img_i = cv2.imread(V_direct+'frame'+str(i)+'.jpg',0)
            img=Applying_bluring_BW(img_i)

            V_cordinat1=VTMPTP(img,V_0_template)                 
            x1=V_cordinat1[2][0]
            y1=V_cordinat1[2][1]
            V_cordinat1=VTMPTP(img,V_0_template2)                    
            xx1=V_cordinat1[2][0]
            yy1=V_cordinat1[2][1]


            img=Applying_fft_filter(img)

            V_i_template = img[int(y1):int(y1)+y_dimention,int(x1):int(x1)+x_dimention] 
            V_i_template2 = img[int(yy1):int(yy1)+yy_dimention,int(xx1):int(xx1)+xx_dimention]


            shift, error, diffphase = register_translation(V_0_fft_template, V_i_template, 100)
            #shift, error, diffphase = register_translation(V_1_template, V_i_template, 100)
            y_sub_shift1=-shift[0] # y direction
            x_sub_shift1=-shift[1] # x direction
            ##lower
            shift, error, diffphase = register_translation(V_0_fft_template2, V_i_template2, 100)
            # shift, error, diffphase = register_translation(V_i_1_template2, V_i_template2, 100)
            y_sub_shift2=-shift[0] # y direction
            x_sub_shift2=-shift[1] # x direction
            
            ##final displacement of upper part
            x1+=x_sub_shift1
            y1+=y_sub_shift1
            #print ("o_y1",o_y1)
            xx1+=x_sub_shift2
            yy1+=y_sub_shift2
            
            o_x1=x1
            o_y1=y1
            o_xx1=xx1
            o_yy1=yy1       
            
            cv2.rectangle(img_i,(int(x1),int(y1)),(int(x1+x_dimention),int(y1+y_dimention)), 
                (125,125,125), 2)
            cv2.rectangle(img_i, (int(xx1),int(yy1)), (int(xx1+xx_dimention),int(yy1+yy_dimention)), 
                (125,125,125), 2)
            cv2.namedWindow('image')
            # resized_image=cv2.resize(img_i, (int(img_i.shape[0]*0.5),int(img_i.shape[1]*0.5)))
            cv2.imshow('image',img_i)
            cv2.waitKey(400)

            diplacement_data=[[i, V_distacebetween_2points(o_x1,o_y1,o_xx1,o_yy1)]]
            print ( i, V_distacebetween_2points(o_x1,o_y1,o_xx1,o_yy1))








        else:
            img_i = cv2.imread(V_direct+'frame'+str(i)+'.jpg',0)
            img=Applying_bluring_BW(img_i)

            V_cordinat1=VTMPTP(img,V_0_template)                 
            x1=V_cordinat1[2][0]
            y1=V_cordinat1[2][1]
            V_cordinat1=VTMPTP(img,V_0_template2)                    
            xx1=V_cordinat1[2][0]
            yy1=V_cordinat1[2][1]


            img=Applying_fft_filter(img)

            V_i_template = img[int(y1):int(y1)+y_dimention,int(x1):int(x1)+x_dimention] 
            V_i_template2 = img[int(yy1):int(yy1)+yy_dimention,int(xx1):int(xx1)+xx_dimention]


            shift, error, diffphase = register_translation(V_0_fft_template, V_i_template, 100)
            #shift, error, diffphase = register_translation(V_1_template, V_i_template, 100)
            y_sub_shift1=-shift[0] # y direction
            x_sub_shift1=-shift[1] # x direction
            ##lower
            shift, error, diffphase = register_translation(V_0_fft_template2, V_i_template2, 100)
            # shift, error, diffphase = register_translation(V_i_1_template2, V_i_template2, 100)
            y_sub_shift2=-shift[0] # y direction
            x_sub_shift2=-shift[1] # x direction
            
            ##final displacement of upper part
            x1+=x_sub_shift1
            y1+=y_sub_shift1
            #print ("o_y1",o_y1)
            xx1+=x_sub_shift2
            yy1+=y_sub_shift2
            
            o_x1=x1
            o_y1=y1
            o_xx1=xx1
            o_yy1=yy1       
            
            cv2.rectangle(img_i,(int(x1),int(y1)),(int(x1+x_dimention),int(y1+y_dimention)), 
                (125,125,125), 2)
            cv2.rectangle(img_i, (int(xx1),int(yy1)), (int(xx1+xx_dimention),int(yy1+yy_dimention)), 
                (125,125,125), 2)
            cv2.namedWindow('image')
            # resized_image=cv2.resize(img_i, (int(img_i.shape[0]*0.5),int(img_i.shape[1]*0.5)))
            cv2.imshow('image',img_i)
            cv2.waitKey(400)

            diplacement_data.append([i, V_distacebetween_2points(o_x1,o_y1,o_xx1,o_yy1)])
            print ( i, V_distacebetween_2points(o_x1,o_y1,o_xx1,o_yy1))















        # ##### to find out how much the upper and lower templates are in sub_pixel it has moved
        # #upper
        
        # shift, error, diffphase = register_translation(V_0_fft_template, V_i_template, 100)
        # #shift, error, diffphase = register_translation(V_1_template, V_i_template, 100)
        # y_sub_shift1=-shift[0] # y direction
        # x_sub_shift1=-shift[1] # x direction
        # ##lower
        # shift, error, diffphase = register_translation(V_0_fft_template2, V_i_template2, 100)
        # # shift, error, diffphase = register_translation(V_i_1_template2, V_i_template2, 100)
        # y_sub_shift2=-shift[0] # y direction
        # x_sub_shift2=-shift[1] # x direction
        
        # ##final displacement of upper part
        # x1+=x_sub_shift1
        # y1+=y_sub_shift1
        # #print ("o_y1",o_y1)
        # xx1+=x_sub_shift2
        # yy1+=y_sub_shift2
        
        # o_x1=x1
        # o_y1=y1
        # o_xx1=xx1
        # o_yy1=yy1       
        
        # cv2.rectangle(img_i,(int(x1),int(y1)),(int(x1+x_dimention),int(y1+y_dimention)), 
        #     (125,125,125), 2)
        # cv2.rectangle(img_i, (int(xx1),int(yy1)), (int(xx1+xx_dimention),int(yy1+yy_dimention)), 
        #     (125,125,125), 2)
        # cv2.namedWindow('image')
        # # resized_image=cv2.resize(img_i, (int(img_i.shape[0]*0.5),int(img_i.shape[1]*0.5)))
        # cv2.imshow('image',img_i)
        # cv2.waitKey(400)
        # if i==firstframe:
        #     diplacement_data=[[i, V_distacebetween_2points(o_x1,o_y1,o_xx1,o_yy1)]]
        # else:
        #     diplacement_data.append([i, V_distacebetween_2points(o_x1,o_y1,o_xx1,o_yy1)])

        # print ( i, V_distacebetween_2points(o_x1,o_y1,o_xx1,o_yy1))

    cv2.destroyAllWindows()
    diplacement_data=np.asarray(diplacement_data)
    np.savetxt(V_direct+'FN_D_step'+str(step)+'.txt',diplacement_data, delimiter=',')
  
    return diplacement_data






def get_text_file_and_convert():
    import tkinter.filedialog   
    sys.path
    sys.path.append('C:\\Users\\VahidSamaee\\opencv')
    data_filename=filedialog.askopenfilename(initialdir='',title='Select the file')
    return np.loadtxt(data_filename, delimiter=',')











def merging_all_data(path_Text_hysiton, Analysied_data, framerate, 
    filename, ptpstiffness, sample_area):
    """ path_Text_hysiton: is the address of the location of the hysitron file
    """
    import numpy as np
    import cv2
    import tkinter
    import tkinter.filedialog
    import os
    import V_f_f as V_functions
    import sys
    import matplotlib.pyplot as plt 


    def V_findIndex (aa, bb): # aa == np.array  , b == the searched vallu  ,  index = (np.abs(V_data-0.2)).argmin()
        index = (np.abs(bb-aa)).argmin()
        return index

    sys.path
    sys.path.append('C:\\Users\\VahidSamaee\\opencv')

    #####<<<<<      open the obtained diplacement text file       >>>>>#####
 
    z = np.zeros((Analysied_data.shape[0],4), dtype=float)
    Analysied_data=np.append(Analysied_data, z, axis=1) # the columns: Fram number, Displacement, Load, Time, Stress, Strain

    initial_lenth= Analysied_data[0,1]
    Analysied_data[:,5]=(Analysied_data[:,1]-initial_lenth)/initial_lenth # changing Distance to Strain 
    print(Analysied_data.shape[0])

    print ('Analysied_data.size',Analysied_data.size)


    #####<<<<<      getting Hysiton  file    >>>>>#####

    V_direct= V_functions.path_extractore(path_Text_hysiton)

    #reading the Hysitron text file, the  columns:"Displacement", "Time" "force" 
    f = open(path_Text_hysiton, "r")

    # analysising to see in which line data is located and then put it in an np.array (V_data)
    i=0
    numberzero=0
    for line in f:
        if line.split()==[]:
            numberzero+=1
            continue
        else:
            try:
                for item in line.split():
                    float(item)
            except:
                continue
            else:
                if i==0:
                    #print (line.split())
                    original_data=np.array( [float(line.split()[2]),float(line.split()[1])])  #"Time","load" "
                    i+=1
                else:
                    i+=1
                    #print (line.split())
                    V_added= np.array ( [float(line.split()[2]), float(line.split()[1])] )
                    original_data= np.vstack((original_data, V_added))
                    
    print (i)
    f.close()


    #####<<<<<      getting the file of Hysiton Text file    >>>>>#####

    for i in range (0, Analysied_data.shape[0]):                                     
        gg= V_findIndex(original_data[:,0],(Analysied_data[i,0]*(1/framerate)))
        if abs(original_data[gg,0]-(Analysied_data[i,0]*(1/framerate)))<1:
            Analysied_data[i,2] =original_data[gg,1] #load
            Analysied_data[i,3] =original_data[gg,0] #time                    
            Analysied_data[i,4] =(1000000*(Analysied_data[i,2]-(ptpstiffness*initial_lenth*Analysied_data[i,1])))/sample_area #stress    
    print (Analysied_data)
    np.savetxt(V_direct+filename+'_FN_D_L_T_Ss_Sr.txt',Analysied_data, delimiter=',')


    #####<<<<<      drawing part   >>>>>##### 
    plt.plot(Analysied_data[:,5] , Analysied_data[:,4]) 
    plt.xlabel('Strain')   # naming the x axis 
    plt.ylabel('Stress (MPa)')   # naming the y axis 
    plt.title('Stress-Strain')   # giving a title to my graph 
    plt.show()   # function to show the plot 


