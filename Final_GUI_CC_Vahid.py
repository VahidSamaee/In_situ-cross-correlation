#####<<<<<             GUI of cross corelation of In-Situ TEM nano mechanical testing                >>>>>#####
import sys
sys.path
sys.path.append('C:\\Users\\VahidSamaee\\opencv')
import cv2
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import V_f_f as VFuntions
import numpy as np
import matplotlib.pyplot as plt
from tkinter import ttk

main=Tk()
main.title('Cross Correlation of In-Situ TEM nano tensile tests')
# main.geometry('800x250')

main_nb=ttk.Notebook(main)
main_nb.grid(row=0, column=0, sticky='NE')
root=Frame(main_nb)
main_nb.add(root,text='  Files, Checking Frames, and Selecting Shoulders  ')

root2=Frame(main_nb)
main_nb.add(root2,text='   Applying Filters   ')

root3=Frame(main_nb)
main_nb.add(root3,text='   Cross Correlation   ')

#####<<<<<  asking the movie and giving back the address and frame rate and number of frams   >>>>>#####
global frame_for_cutting, BW_treshold
frame_for_cutting=1
BW_treshold=0
def ask_for_the_file():
	global data_filename, data_folder
	data_filename=filedialog.askopenfilename(initialdir='',title='Select the movie')
	data_folder=VFuntions.path_extractore(data_filename)
	# path_label=Label(framing_frame_in,text= 'The movie:  '+ data_filename)
	# path_label.grid(row=2, column=0, sticky=W)
	path_label=Label(framing_frame_in,text= data_folder)
	path_label.grid(row=0, column=1, sticky=W)
	#import V_framingMovie
	# a windows showing the advancement of the processing is needed
	return

def ask_for_the_text_file():
	global Hysitron_data_filename
	Hysitron_data_filename=filedialog.askopenfilename(initialdir='',title='Select the movie')

	path_label2=Label(framing_frame_in,text= Hysitron_data_filename)
	path_label2.grid(row=1, column=1, sticky=W)
	#import V_framingMovie
	# a windows showing the advancement of the processing is needed
	return

def do_framing():
	VFuntions.V_framing(data_filename) 
	return

def movie_info_request():

	global movie_information
	movie_information=VFuntions.movie_info (data_filename)
	path_label=Label(framing_frame_in2,text= 'frame rate: '+ str(movie_information[0])+
	' and number of frames are: '+ str(movie_information[1]))
	path_label.grid(row=5, column=2,sticky=W)
	showing_all_frames()
	ending_frame_input.insert(0, str(movie_information[1]))
	return


movie_information=(0,0)

framing_frame = LabelFrame(root, text='Use the bottun to select the movie file', padx=2, pady=2)
framing_frame.grid(row=0, column=0, padx=2, pady=2, columnspan=50, sticky=W)

framing_frame_in = LabelFrame(framing_frame, padx=2, pady=2)
framing_frame_in.grid(row=0, column=0, padx=2, pady=2, sticky=W)

framing_botton=Button (framing_frame_in, text='Select the movie', command = ask_for_the_file)
framing_botton.grid(row=0, column=0, padx=2, pady=2,sticky=W) 	

framing_botton=Button (framing_frame_in, text='Select Hysitron text file', command = ask_for_the_text_file)
framing_botton.grid(row=1, column=0, padx=2, pady=2,sticky=W) 

# button to start framing

framing_frame_in2 = LabelFrame(framing_frame, padx=2, pady=2)
framing_frame_in2.grid(row=1, column=0, padx=2, pady=2, sticky=W)

framing_botton=Button (framing_frame_in2, text='Do Framing', command = do_framing)
framing_botton.grid(row=4, column=0, padx=5, pady=2,sticky=W) 
framing_botton=Button (framing_frame_in2, text='movie info', command = movie_info_request)
framing_botton.grid(row=4, column=1, padx=5, pady=2,sticky=W)

#####<<<<< getting the crop areas   >>>>>#####
getting_cropped_areas = LabelFrame(framing_frame, text='applying the ffd and bluring')
getting_cropped_areas.grid(row=5, column=0, columnspan=10, padx=2, pady=2,sticky=W)
global x1,y1,x2,y2, xx1,yy1,xx2,yy2
x1=y1=x2=y2= xx1=yy1=xx2=yy2=0

def crpping_2_template():
	global  x1,y1,x2,y2, xx1,yy1,xx2,yy2,frame_for_cutting 
	x1,y1,x2,y2=VFuntions.V_selecting_a_region_on_image(data_folder+"frame"+str(frame_for_cutting)+".jpg",1)
	xx1,yy1,xx2,yy2=VFuntions.V_selecting_a_region_on_image(data_folder+"frame"+str(frame_for_cutting)+".jpg",1)
	if x1>x2:
		x1,x2=x2,x1
	if y1>y2:
		y1,y2=y2,y1
	if xx1>xx2:
		xx1,xx2=xx2,xx1
	if yy1>yy2:
		yy1,yy2=yy2,yy1
	return

do_croping_button=Button(getting_cropped_areas, text='choose 2 areas', command=crpping_2_template)
do_croping_button.grid(row=0, column=0, padx=2, pady=2,sticky=W)


# showing_frames_frame = LabelFrame(root, text='Frames', padx=5, pady=5, width=10, height=10)
def showing_all_frames():

	def show_frame(frame_number):
		img=cv2.imread(data_folder+'frame'+str(horizontal_scale.get())+'.jpg')
		scaling=0.5
		img=cv2.resize(img, (round(img.shape[0]*scaling),round(img.shape[1]*scaling)),interpolation = cv2.INTER_AREA)
		cv2.imshow('image',img)
		return

	def slide(var):
		show_frame(horizontal_scale.get())
		global frame_for_cutting
		frame_for_cutting= horizontal_scale.get()

	horizontal_scale=Scale(framing_frame_in2, from_=1, to=movie_information[1],
	length= 200, orient=HORIZONTAL, command= slide)
	horizontal_scale.grid(row=4, column=2, padx=2, pady=2)
	print(horizontal_scale.get())
	return horizontal_scale.get()

def showing_all_frames_with_changes():

	def slide2():
		plt.close()
		frame_number=horizontal_scale.get()
		if full_frame.get()=='no':
			Fimage=cv2.imread(data_folder+'frame'+str(frame_number)+'.jpg',0)
		elif full_frame.get()=='yes':
			if which_cropped_frame.get()=='uper frame':
				# the area of the uper frame should be loaded here
				Fimage=cv2.imread(data_folder+'frame'+str(frame_number)+'.jpg',0)[y1:y2,x1:x2]
			elif which_cropped_frame.get()=='lower frame':
				# the area of the uper frame should be loaded here
				Fimage=cv2.imread(data_folder+'frame'+str(frame_number)+'.jpg',0)[yy1:yy2,xx1:xx2]

			# this should change based on the drob down value 
			
		VFuntions.applying_FFT_bluring_filter(Fimage, BW_treshold,fft_value)
		return
		
	showing_frames_frame_changes = LabelFrame(root2, text='Frames', width=600, height=10)
	showing_frames_frame_changes.grid(row=3, column=0, padx=2, pady=2, sticky=W)

	horizontal_scale=Scale(showing_frames_frame_changes, from_=1, to=movie_information[1],
	length= 400, orient=HORIZONTAL)
	horizontal_scale.grid(row=0, column=0, padx=2, pady=2, sticky=W)

	framing_botton=Button (showing_frames_frame_changes, text='show the changes', command= slide2)
	framing_botton.grid(row=4, column=1, padx=2, pady=2,sticky=W) 

def apply_filter():
	global fft_value, BW_treshold
	fft_value=int(fft_radious.get())
	BW_treshold=int(bluring_treshold.get())
	showing_all_frames_with_changes()
	return


applying_FFT = LabelFrame(root2, text='applying the ffd and bluring')
applying_FFT.grid(row=2, column=0, padx=2, pady=2,sticky=W)

label_fft=Label(applying_FFT, text=' R valye, FFT,ifft=')
label_fft.grid(row=0, column=0, padx=2, pady=2,sticky=E)

fft_radious= Entry(applying_FFT, width=10) 
fft_radious.grid(row=0, column=1, padx=2, pady=2,sticky=W)
fft_radious.insert(0, '100')

label_bluring=Label(applying_FFT, text=' BW Treshold=')
label_bluring.grid(row=0, column=2, padx=2, pady=2, sticky=E)

bluring_treshold= Entry(applying_FFT, width=10) 
bluring_treshold.grid(row=0, column=3, padx=2, pady=2, sticky=W)
bluring_treshold.insert(0, '200')

check_filter_changes =Button (applying_FFT, text='Apply the Changes', command = apply_filter)
check_filter_changes.grid(row=0, column=4, padx=2, pady=2, sticky=W) 

full_frame=StringVar()
check_button=Checkbutton(applying_FFT,text='Apply changes on:',
	variable=full_frame, onvalue='yes', offvalue='no')
check_button.deselect()
check_button.grid(row=2, column=0, padx=2, pady=2, sticky=W)

which_cropped_frame=StringVar()
which_cropped_frame.set ("uper frame")
mylist=["uper frame","lower frame" ]
mydropbox= OptionMenu(applying_FFT, which_cropped_frame, *mylist )
mydropbox.grid(row=2, column=1, padx=2, pady=2, sticky=W)

#####<<<<<     making movie by frames     >>>>>#####	

#####<<<<<     checking if the values are fine or not     >>>>>#####
def CC_function ():
	# here the function of cross correlation should be run
	#input: x1,y1,x2,y2,xx1,yy1,xx2,yy2, fft_value, BW_treshold
	global data_folder, x1,y1,x2,y2,xx1,yy1,xx2,yy2,fft_value,BW_treshold,frame_for_cutting
	global diplacement_data_from_movie
	firstframe = int(startin_frame_input.get())
	lastframe = int(ending_frame_input.get())
	step = int(step_step_input.get())
	fft_value=int(fft_radious.get())
	BW_treshold=int(bluring_treshold.get())
	diplacement_data_from_movie=VFuntions.cross_correlation(data_folder, x1,y1,x2,y2,xx1,yy1,xx2,yy2,fft_value,
		BW_treshold,firstframe,lastframe,step, frame_for_cutting) 
	return

run_cross_correlation = LabelFrame(root3, text='cross correlation')
run_cross_correlation.grid(row=6, column=0, padx=2, pady=2,sticky=W)

check_filter_changes =Button (run_cross_correlation, text='cross correlation', command = CC_function)
check_filter_changes.grid(row=0, column=0, padx=2, pady=2, sticky=W) 

startin_frame=Label(run_cross_correlation, text=' from frame=')
startin_frame.grid(row=0, column=1, padx=2, pady=2,sticky=E)
startin_frame_input= Entry(run_cross_correlation, width=10) 
startin_frame_input.grid(row=0, column=2, padx=2, pady=2,sticky=W)
startin_frame_input.insert(0, '1')

ending_frame=Label(run_cross_correlation, text=' to frame=')
ending_frame.grid(row=0, column=3, padx=2, pady=2, sticky=E)
ending_frame_input= Entry(run_cross_correlation, width=10) 
ending_frame_input.grid(row=0, column=4, padx=2, pady=2, sticky=W)
ending_frame_input.insert(0, str(movie_information[1]))

step_step=Label(run_cross_correlation, text=' step size=')
step_step.grid(row=0, column=5, padx=2, pady=2, sticky=E)
step_step_input= Entry(run_cross_correlation, width=10) 
step_step_input.grid(row=0, column=6, padx=2, pady=2, sticky=W)
step_step_input.insert(0, '200')


def cc_al_done_function():
	global diplacement_data_from_movie
	diplacement_data_from_movie=VFuntions.get_text_file_and_convert()
	print(diplacement_data_from_movie)
	return

cc_al_done =Button (run_cross_correlation, 
	text='If Cross Correlation is already done, use this to select the text file !', 
	command = cc_al_done_function)
cc_al_done.grid(row=1, column=0, padx=2, pady=2, sticky=W) 

cc_al_done_label=Label(run_cross_correlation, text=' the formate of the file is "displacement_step*.txt"')
cc_al_done_label.grid(row=2, column=0, padx=2, pady=2,sticky=E)

#####<<<<< combining 2 set of data  >>>>>#####
def mergingPTP ():
	# here the function of cross correlation should be run
	#input: x1,y1,x2,y2,xx1,yy1,xx2,yy2, fft_value, BW_treshold
	global diplacement_data_from_movie, Hysitron_data_filename,movie_information 
	filename_PTPiinput.get()
	VFuntions.merging_all_data(
		Hysitron_data_filename, 
		diplacement_data_from_movie,
		movie_information[0], 
		filename_PTPiinput.get(), 
		float(stiffness_PTPiinput.get()), 
		float(Sample_crosssecctioninput.get())
		) 
	return

merging_data = LabelFrame(root3, text='mergingdata and dra the curve')
merging_data.grid(row=7, column=0, padx=2, pady=2,sticky=W)

check_filter_changes =Button (merging_data, text='Merge and Draw', command = mergingPTP)
check_filter_changes.grid(row=0, column=0, padx=2, pady=2, sticky=W) 

filename_PTP=Label(merging_data, text='added to file name: ')
filename_PTP.grid(row=0, column=1, padx=2, pady=2,sticky=E)
filename_PTPiinput= Entry(merging_data, width=10) 
filename_PTPiinput.grid(row=0, column=2, padx=2, pady=2,sticky=W)
filename_PTPiinput.insert(0, 'step')

stiffness_PTP=Label(merging_data, text=' PTP Stiffness (MN/Pixel)=')
stiffness_PTP.grid(row=1, column=1, padx=2, pady=2,sticky=E)
stiffness_PTPiinput= Entry(merging_data, width=10) 
stiffness_PTPiinput.grid(row=1, column=2, padx=2, pady=2,sticky=W)
stiffness_PTPiinput.insert(0, '0')

Sample_crosssecction=Label(merging_data, text=' cross secction area (nm * nm)=')
Sample_crosssecction.grid(row=2, column=1, padx=2, pady=2,sticky=E)
Sample_crosssecctioninput= Entry(merging_data, width=10) 
Sample_crosssecctioninput.grid(row=2, column=2, padx=2, pady=2,sticky=W)
Sample_crosssecctioninput.insert(0, '1')




# ending_frame=Label(merging_data, text=' to frame=')
# ending_frame.grid(row=0, column=3, padx=2, pady=2, sticky=E)

# ending_frame_input= Entry(merging_data, width=10) 
# ending_frame_input.grid(row=0, column=4, padx=2, pady=2, sticky=W)
# ending_frame_input.insert(0, str(movie_information[1]))


# step_step=Label(merging_data, text=' step size=')
# step_step.grid(row=0, merging_data=5, padx=2, pady=2, sticky=E)

# step_step_input= Entry(merging_data, width=10) 
# step_step_input.grid(row=0, column=6, padx=2, pady=2, sticky=W)
# step_step_input.insert(0, '200')


main.mainloop()