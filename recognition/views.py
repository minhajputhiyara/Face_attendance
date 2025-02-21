from django.shortcuts import render,redirect
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import authenticate,login,logout
from django.contrib.auth.decorators import login_required
import cv2
import dlib
import numpy as np
from PIL import Image 
import os
import csv
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import json
import base64
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from . models import UserfaceData, Present, Time
import pickle
import time
from datetime import datetime 
from django.http import JsonResponse
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
import datetime
from django_pandas.io import read_frame
import seaborn as sns
from django.db.models import Count
from .forms import usernameForm,DateForm,UsernameAndDateForm, DateForm_2
import imutils
from imutils.face_utils import FaceAligner
from imutils import face_utils
from imutils.video import VideoStream
from imutils.face_utils import rect_to_bb
from attendance_system_facial_recognition.settings import BASE_DIR
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams
import math
from pandas.plotting import register_matplotlib_converters

mpl.use('Agg')


#utility functions:
def username_present(username):
	if User.objects.filter(username=username).exists():
		return True
	
	return False

def create_dataset(username):
	id = username
	if(os.path.exists('face_recognition_data/training_dataset/{}/'.format(id))==False):
		os.makedirs('face_recognition_data/training_dataset/{}/'.format(id))
	directory='face_recognition_data/training_dataset/{}/'.format(id)

	print("[INFO] Loading the facial detector")
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_68_face_landmarks.dat')

	print("[INFO] Initializing Video stream")
	vs = VideoStream(src=0).start()
	sampleNum = 0
	required_samples = 50  
    
	print(f"[INFO] Starting capture. Need {required_samples} good samples.")
	print("[INFO] Please look at the camera and move your head slightly to capture different angles.")

	while(True):
		frame = vs.read()
		frame = imutils.resize(frame, width=800)
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = detector(gray_frame, 0)

		for face in faces:
			try:
				(x, y, w, h) = face_utils.rect_to_bb(face)
				
				# Instead of using FaceAligner, we'll extract the face region directly
				face_img = frame[y:y+h, x:x+w]
				if face_img is not None and face_img.size > 0:
					# Resize to a standard size
					face_img = cv2.resize(face_img, (96, 96))
					
					sampleNum = sampleNum + 1
					cv2.imwrite(directory + '/' + str(sampleNum) + '.jpg', face_img)
					
					# Draw rectangle and progress
					cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 1)
					progress = f"Progress: {sampleNum}/{required_samples}"
					cv2.putText(frame, progress, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
					
					print(f"\rCapturing sample {sampleNum}/{required_samples}", end="")
					cv2.waitKey(50)
			except Exception as e:
				print(f"\nError processing face: {str(e)}")
				continue

		cv2.imshow("Add Images", frame)
		cv2.waitKey(1)
		
		if(sampleNum >= required_samples):
			break
	
	print("\n[INFO] Dataset capture completed!")
	vs.stop()
	cv2.destroyAllWindows()


def predict(face_aligned,svc,threshold=0.7):
	face_encodings=np.zeros((1,128))
	try:
		x_face_locations=face_recognition.face_locations(face_aligned)
		faces_encodings=face_recognition.face_encodings(face_aligned,known_face_locations=x_face_locations)
		if(len(faces_encodings)==0):
			return ([-1],[0])

	except:

		return ([-1],[0])

	prob=svc.predict_proba(faces_encodings)
	result=np.where(prob[0]==np.amax(prob[0]))
	if(prob[0][result[0]]<=threshold):
		return ([-1],prob[0][result[0]])

	return (result[0],prob[0][result[0]])


def vizualize_Data(embedded, targets,):
	
	X_embedded = TSNE(n_components=2).fit_transform(embedded)

	for i, t in enumerate(set(targets)):
		idx = targets == t
		plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=t)

	plt.legend(bbox_to_anchor=(1, 1));
	rcParams.update({'figure.autolayout': True})
	plt.tight_layout()	
	plt.savefig('./recognition/static/recognition/img/training_visualisation.png')
	plt.close()



def update_attendance_in_db_in(present):
    today = datetime.date.today()
    time_now = datetime.datetime.now()
    
    for person, is_present in present.items():
        try:
            user = User.objects.get(username=person)
            if is_present:
                # Update or create Present record
                Present.objects.update_or_create(
                    user=user,
                    date=today,
                    defaults={'present': True}
                )
                
                # Create Time record for check-in
                Time.objects.create(
                    user=user,
                    date=today,
                    time=time_now,
                    out=False
                )
                print(f"Marked attendance for {person} at {time_now.strftime('%H:%M:%S')}")
        except User.DoesNotExist:
            print(f"User {person} not found in database")
        except Exception as e:
            print(f"Error marking attendance for {person}: {str(e)}")

def update_attendance_in_db_out(present):
    today = datetime.date.today()
    time_now = datetime.datetime.now()
    
    for person, is_present in present.items():
        try:
            user = User.objects.get(username=person)
            if is_present:
                # Create Time record for check-out
                Time.objects.create(
                    user=user,
                    date=today,
                    time=time_now,
                    out=True
                )
                print(f"Marked check-out for {person} at {time_now.strftime('%H:%M:%S')}")
        except User.DoesNotExist:
            print(f"User {person} not found in database")
        except Exception as e:
            print(f"Error marking check-out for {person}: {str(e)}")

def check_validity_times(times_all):
    if not times_all.exists():
        return False, 0
        
    # Get all records for the day sorted by time
    times_all = times_all.order_by('time')
    
    # Check if last action was check-out
    if times_all.last().out == False:
        return False, 0
        
    # Check if check-ins and check-outs alternate
    prev_was_out = True  # We expect a check-in first
    total_break_hours = 0
    prev_time = None
    
    for record in times_all:
        if record.out == prev_was_out:  # Two ins or two outs in a row
            return False, 0
            
        if record.out:  # This is a check-out
            if prev_time:
                break_hours = (record.time - prev_time).total_seconds() / 3600
                total_break_hours += break_hours
                
        prev_was_out = record.out
        prev_time = record.time
    
    return True, total_break_hours


def convert_hours_to_hours_mins(hours):
	
	h=int(hours)
	hours-=h
	m=hours*60
	m=math.ceil(m)
	return str(str(h)+ " hrs " + str(m) + "  mins")

		

#used
def hours_vs_date_given_employee(present_qs,time_qs,admin=True):
	register_matplotlib_converters()
	df_hours=[]
	df_break_hours=[]
	qs=present_qs

	for obj in qs:
		date=obj.date
		times_in=time_qs.filter(date=date).filter(out=False).order_by('time')
		times_out=time_qs.filter(date=date).filter(out=True).order_by('time')
		times_all=time_qs.filter(date=date).order_by('time')
		obj.time_in=None
		obj.time_out=None
		obj.hours=0
		obj.break_hours=0
		if (len(times_in)>0):			
			obj.time_in=times_in.first().time
			
		if (len(times_out)>0):
			obj.time_out=times_out.last().time

		if(obj.time_in is not None and obj.time_out is not None):
			ti=obj.time_in
			to=obj.time_out
			hours=((to-ti).total_seconds())/3600
			obj.hours=hours
		

		else:
			obj.hours=0

		(check,break_hourss)= check_validity_times(times_all)
		if check:
			obj.break_hours=break_hourss


		else:
			obj.break_hours=0


		
		df_hours.append(obj.hours)
		df_break_hours.append(obj.break_hours)
		obj.hours=convert_hours_to_hours_mins(obj.hours)
		obj.break_hours=convert_hours_to_hours_mins(obj.break_hours)
			
	
	
	
	df = read_frame(qs)	
	
	
	df["hours"]=df_hours
	df["break_hours"]=df_break_hours

	print(df)
	
	sns.barplot(data=df,x='date',y='hours')
	plt.xticks(rotation='vertical')
	rcParams.update({'figure.autolayout': True})
	plt.tight_layout()
	if(admin):
		plt.savefig('./recognition/static/recognition/img/attendance_graphs/hours_vs_date/1.png')
		plt.close()
	else:
		plt.savefig('./recognition/static/recognition/img/attendance_graphs/employee_login/1.png')
		plt.close()
	return qs
	

#used
def hours_vs_employee_given_date(present_qs,time_qs):
	register_matplotlib_converters()
	df_hours=[]
	df_break_hours=[]
	df_username=[]
	qs=present_qs

	for obj in qs:
		user=obj.user
		times_in=time_qs.filter(user=user).filter(out=False)
		times_out=time_qs.filter(user=user).filter(out=True)
		times_all=time_qs.filter(user=user)
		obj.time_in=None
		obj.time_out=None
		obj.hours=0
		obj.hours=0
		if (len(times_in)>0):			
			obj.time_in=times_in.first().time
		if (len(times_out)>0):
			obj.time_out=times_out.last().time
		if(obj.time_in is not None and obj.time_out is not None):
			ti=obj.time_in
			to=obj.time_out
			hours=((to-ti).total_seconds())/3600
			obj.hours=hours
		else:
			obj.hours=0
		(check,break_hourss)= check_validity_times(times_all)
		if check:
			obj.break_hours=break_hourss


		else:
			obj.break_hours=0

		
		df_hours.append(obj.hours)
		df_username.append(user.username)
		df_break_hours.append(obj.break_hours)
		obj.hours=convert_hours_to_hours_mins(obj.hours)
		obj.break_hours=convert_hours_to_hours_mins(obj.break_hours)

	



	df = read_frame(qs)	
	df['hours']=df_hours
	df['username']=df_username
	df["break_hours"]=df_break_hours


	sns.barplot(data=df,x='username',y='hours')
	plt.xticks(rotation='vertical')
	rcParams.update({'figure.autolayout': True})
	plt.tight_layout()
	plt.savefig('./recognition/static/recognition/img/attendance_graphs/hours_vs_employee/1.png')
	plt.close()
	return qs


def total_number_employees():
	qs=User.objects.all()
	return (len(qs) -1)
	# -1 to account for admin 



def employees_present_today():
	today=datetime.date.today()
	qs=Present.objects.filter(date=today).filter(present=True)
	return len(qs)




#used	
def this_week_emp_count_vs_date():
	today=datetime.date.today()
	some_day_last_week=today-datetime.timedelta(days=7)
	monday_of_last_week=some_day_last_week-  datetime.timedelta(days=(some_day_last_week.isocalendar()[2] - 1))
	monday_of_this_week = monday_of_last_week + datetime.timedelta(days=7)
	qs=Present.objects.filter(date__gte=monday_of_this_week).filter(date__lte=today)
	str_dates=[]
	emp_count=[]
	str_dates_all=[]
	emp_cnt_all=[]
	cnt=0
	
	



	for obj in qs:
		date=obj.date
		str_dates.append(str(date))
		qs=Present.objects.filter(date=date).filter(present=True)
		emp_count.append(len(qs))


	while(cnt<5):

		date=str(monday_of_this_week+datetime.timedelta(days=cnt))
		cnt+=1
		str_dates_all.append(date)
		if(str_dates.count(date))>0:
			idx=str_dates.index(date)

			emp_cnt_all.append(emp_count[idx])
		else:
			emp_cnt_all.append(0)

	
	
	



	df=pd.DataFrame()
	df["date"]=str_dates_all
	df["Number of employees"]=emp_cnt_all
	
	
	sns.lineplot(data=df,x='date',y='Number of employees')
	plt.savefig('./recognition/static/recognition/img/attendance_graphs/this_week/1.png')
	plt.close()






#used
def last_week_emp_count_vs_date():
	today=datetime.date.today()
	some_day_last_week=today-datetime.timedelta(days=7)
	monday_of_last_week=some_day_last_week-  datetime.timedelta(days=(some_day_last_week.isocalendar()[2] - 1))
	monday_of_this_week = monday_of_last_week + datetime.timedelta(days=7)
	qs=Present.objects.filter(date__gte=monday_of_last_week).filter(date__lt=monday_of_this_week)
	str_dates=[]
	emp_count=[]


	str_dates_all=[]
	emp_cnt_all=[]
	cnt=0
	
	



	for obj in qs:
		date=obj.date
		str_dates.append(str(date))
		qs=Present.objects.filter(date=date).filter(present=True)
		emp_count.append(len(qs))


	while(cnt<5):

		date=str(monday_of_last_week+datetime.timedelta(days=cnt))
		cnt+=1
		str_dates_all.append(date)
		if(str_dates.count(date))>0:
			idx=str_dates.index(date)

			emp_cnt_all.append(emp_count[idx])
			
		else:
			emp_cnt_all.append(0)

	
	
	



	df=pd.DataFrame()
	df["date"]=str_dates_all
	df["emp_count"]=emp_cnt_all
	

	
	
	sns.lineplot(data=df,x='date',y='emp_count')
	plt.savefig('./recognition/static/recognition/img/attendance_graphs/last_week/1.png')
	plt.close()


		





# Create your views here.
def home(request):

	return render(request, 'recognition/home.html')

@login_required
def dashboard(request):
    if request.user.username.startswith(('admin', 'admin2')):
        print("admin")
        return render(request, 'recognition/admin_dashboard.html')
    else:
        print("not admin")
        return render(request, 'recognition/employee_dashboard.html')

@login_required
def add_photos(request):
    if not request.user.username.startswith(('admin', 'admin2')):
        return redirect('not-authorised')
        
    if request.method == 'POST':
        form = usernameForm(request.POST)
        data = request.POST.copy()
        username = data.get('username')
        if username_present(username):
            create_dataset(username)
            messages.success(request, f'Dataset Created')
            return redirect('add-photos')
        else:
            messages.warning(request, f'No such username found. Please register employee first.')
            return redirect('dashboard')


    else:
        

        form = usernameForm()
        return render(request, 'recognition/add_photos.html', {'form': form})



def mark_your_attendance(request):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_68_face_landmarks.dat')
    svc_save_path = "face_recognition_data/svc.sav"    

    with open(svc_save_path, 'rb') as f:
        svc = pickle.load(f)
    encoder = LabelEncoder()
    encoder.classes_ = np.load('face_recognition_data/classes.npy')

    faces_encodings = np.zeros((1, 128))
    no_of_faces = len(svc.predict_proba(faces_encodings)[0])
    count = dict()
    present = dict()
    log_time = dict()
    start = dict()
    for i in range(no_of_faces):
        count[encoder.inverse_transform([i])[0]] = 0
        present[encoder.inverse_transform([i])[0]] = False

    vs = VideoStream(src=0)
    print("[INFO] Starting video stream...")
    vs.start()
    time.sleep(2.0)  # Allow camera to warm up
    
    retries = 0
    max_retries = 5
    
    while(True):
        try:
            frame = vs.read()
            if frame is None:
                print("[ERROR] Failed to capture frame")
                retries += 1
                if retries > max_retries:
                    print("[ERROR] Maximum retries reached. Camera might be in use or not available.")
                    vs.stop()
                    cv2.destroyAllWindows()
                    messages.error(request, "Camera not available. Please try again.")
                    return redirect('home')
                time.sleep(0.5)  # Wait before next try
                continue
                
            retries = 0  # Reset retries on successful frame capture
            frame = imutils.resize(frame, width=800)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray_frame, 0)

            for face in faces:
                try:
                    (x, y, w, h) = face_utils.rect_to_bb(face)
                    
                    # Extract and resize face region
                    face_img = frame[y:y+h, x:x+w]
                    if face_img is not None and face_img.size > 0:
                        face_img = cv2.resize(face_img, (96, 96))
                        
                        # Get face encoding and predict
                        face_encoding = face_recognition.face_encodings(face_img)
                        if len(face_encoding) > 0:
                            (pred, prob) = predict(face_img, svc)
                            
                            if pred != [-1]:
                                person_name = encoder.inverse_transform(np.ravel([pred]))[0]
                                pred = person_name
                                if count[pred] == 0:
                                    start[pred] = time.time()
                                    count[pred] = count.get(pred, 0) + 1

                                if count[pred] == 4 and (time.time()-start[pred]) > 1.2:
                                    count[pred] = 0
                                else:
                                    present[pred] = True
                                    log_time[pred] = datetime.datetime.now()
                                    count[pred] = count.get(pred, 0) + 1
                                    print(f"Recognized: {pred} (Count: {count[pred]})")
                                
                                # Draw rectangle and name
                                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                                cv2.putText(frame, person_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
                
                except Exception as e:
                    print(f"Error processing face: {str(e)}")
                    continue

            cv2.imshow("Mark Attendance - Press 'q' to quit", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
                
        except Exception as e:
            print(f"Error capturing frame: {str(e)}")
            retries += 1
            if retries > max_retries:
                print("[ERROR] Maximum retries reached. Camera might be in use or not available.")
                vs.stop()
                cv2.destroyAllWindows()
                messages.error(request, "Camera not available. Please try again.")
                return redirect('home')
            time.sleep(0.5)  # Wait before next try
            continue

    vs.stop()
    cv2.destroyAllWindows()
    update_attendance_in_db_in(present)
    messages.success(request, "Attendance marked successfully! Welcome!")
    return redirect('home')


def mark_your_attendance_out(request):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_68_face_landmarks.dat')
    svc_save_path = "face_recognition_data/svc.sav"    

    with open(svc_save_path, 'rb') as f:
        svc = pickle.load(f)
    encoder = LabelEncoder()
    encoder.classes_ = np.load('face_recognition_data/classes.npy')

    faces_encodings = np.zeros((1, 128))
    no_of_faces = len(svc.predict_proba(faces_encodings)[0])
    count = dict()
    present = dict()
    log_time = dict()
    start = dict()
    for i in range(no_of_faces):
        count[encoder.inverse_transform([i])[0]] = 0
        present[encoder.inverse_transform([i])[0]] = False

    vs = VideoStream(src=0)
    print("[INFO] Starting video stream...")
    vs.start()
    time.sleep(2.0)  # Allow camera to warm up
    
    retries = 0
    max_retries = 5
    
    while(True):
        try:
            frame = vs.read()
            if frame is None:
                print("[ERROR] Failed to capture frame")
                retries += 1
                if retries > max_retries:
                    print("[ERROR] Maximum retries reached. Camera might be in use or not available.")
                    vs.stop()
                    cv2.destroyAllWindows()
                    messages.error(request, "Camera not available. Please try again.")
                    return redirect('home')
                time.sleep(0.5)  # Wait before next try
                continue
                
            retries = 0  # Reset retries on successful frame capture
            frame = imutils.resize(frame, width=800)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray_frame, 0)

            for face in faces:
                try:
                    (x, y, w, h) = face_utils.rect_to_bb(face)
                    
                    # Extract and resize face region
                    face_img = frame[y:y+h, x:x+w]
                    if face_img is not None and face_img.size > 0:
                        face_img = cv2.resize(face_img, (96, 96))
                        
                        # Get face encoding and predict
                        face_encoding = face_recognition.face_encodings(face_img)
                        if len(face_encoding) > 0:
                            (pred, prob) = predict(face_img, svc)
                            
                            if pred != [-1]:
                                person_name = encoder.inverse_transform(np.ravel([pred]))[0]
                                pred = person_name
                                if count[pred] == 0:
                                    start[pred] = time.time()
                                    count[pred] = count.get(pred, 0) + 1

                                if count[pred] == 4 and (time.time()-start[pred]) > 1.2:
                                    count[pred] = 0
                                else:
                                    present[pred] = True
                                    log_time[pred] = datetime.datetime.now()
                                    count[pred] = count.get(pred, 0) + 1
                                    print(f"Recognized: {pred} (Count: {count[pred]})")
                                
                                # Draw rectangle and name
                                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                                cv2.putText(frame, person_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
                
                except Exception as e:
                    print(f"Error processing face: {str(e)}")
                    continue

            cv2.imshow("Mark Attendance Out - Press 'q' to quit", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
                
        except Exception as e:
            print(f"Error capturing frame: {str(e)}")
            retries += 1
            if retries > max_retries:
                print("[ERROR] Maximum retries reached. Camera might be in use or not available.")
                vs.stop()
                cv2.destroyAllWindows()
                messages.error(request, "Camera not available. Please try again.")
                return redirect('home')
            time.sleep(0.5)  # Wait before next try
            continue

    vs.stop()
    cv2.destroyAllWindows()
    update_attendance_in_db_out(present)
    messages.success(request, "Check-out recorded successfully! Have a great day!")
    return redirect('home')


@login_required
def train(request):
    if not request.user.username.startswith(('admin', 'admin2')):
        return redirect('not-authorised')

    training_dir = 'face_recognition_data/training_dataset'
    
    count = 0
    for person_name in os.listdir(training_dir):
        curr_directory = os.path.join(training_dir, person_name)
        if not os.path.isdir(curr_directory):
            continue
        for imagefile in image_files_in_folder(curr_directory):
            count += 1

    X = []
    y = []
    i = 0

    for person_name in os.listdir(training_dir):
        print(str(person_name))
        curr_directory = os.path.join(training_dir, person_name)
        if not os.path.isdir(curr_directory):
            continue
        for imagefile in image_files_in_folder(curr_directory):
            print(str(imagefile))
            image = cv2.imread(imagefile)
            try:
                X.append((face_recognition.face_encodings(image)[0]).tolist())
                y.append(person_name)
                i += 1
            except:
                print("removed")
                os.remove(imagefile)

    targets = np.array(y)
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    X1 = np.array(X)
    print("shape: " + str(X1.shape))
    np.save('face_recognition_data/classes.npy', encoder.classes_)
    svc = SVC(kernel='linear', probability=True, random_state=42)
    svc.fit(X1, y)
    svc_save_path = "face_recognition_data/svc.sav"
    with open(svc_save_path, 'wb') as f:
        pickle.dump(svc, f, protocol=4)

    vizualize_Data(X1, targets)
    
    messages.success(request, f'Training Complete.')

    return render(request, "recognition/train.html")


@login_required
def not_authorised(request):
	return render(request,'recognition/not_authorised.html')



@login_required
def view_attendance_home(request):
	total_num_of_emp=total_number_employees()
	emp_present_today=employees_present_today()
	this_week_emp_count_vs_date()
	last_week_emp_count_vs_date()
	return render(request,"recognition/view_attendance_home.html", {'total_num_of_emp' : total_num_of_emp, 'emp_present_today': emp_present_today})


@login_required
def view_attendance_date(request):
	if request.user.username not in ('admin', 'admin2'):
		return redirect('not-authorised')
	qs=None
	time_qs=None
	present_qs=None


	if request.method=='POST':
		form=DateForm(request.POST)
		if form.is_valid():
			date=form.cleaned_data.get('date')
			print("date:"+ str(date))
			time_qs=Time.objects.filter(date=date)
			present_qs=Present.objects.filter(date=date)
			if(len(time_qs)>0 or len(present_qs)>0):
				qs=hours_vs_employee_given_date(present_qs,time_qs)


				return render(request,'recognition/view_attendance_date.html', {'form' : form,'qs' : qs })
			else:
				messages.warning(request, f'No records for selected date.')
				return redirect('view-attendance-date')


			
			
			
		


	else:
		

			form=DateForm()
			return render(request,'recognition/view_attendance_date.html', {'form' : form, 'qs' : qs})


@login_required
def view_attendance_employee(request):
	if request.user.username not in ('admin', 'admin2'):
		return redirect('not-authorised')
	time_qs=None
	present_qs=None
	qs=None

	if request.method=='POST':
		form=UsernameAndDateForm(request.POST)
		if form.is_valid():
			username=form.cleaned_data.get('username')
			if username_present(username):
				
				u=User.objects.get(username=username)
				
				time_qs=Time.objects.filter(user=u)
				present_qs=Present.objects.filter(user=u)
				date_from=form.cleaned_data.get('date_from')
				date_to=form.cleaned_data.get('date_to')
				
				if date_to < date_from:
					messages.warning(request, f'Invalid date selection.')
					return redirect('view-attendance-employee')
				else:
					

					time_qs=time_qs.filter(date__gte=date_from).filter(date__lte=date_to).order_by('-date')
					present_qs=present_qs.filter(date__gte=date_from).filter(date__lte=date_to).order_by('-date')
					
					if (len(time_qs)>0 or len(present_qs)>0):
						qs=hours_vs_date_given_employee(present_qs,time_qs,admin=True)
						return render(request,'recognition/view_attendance_employee.html', {'form' : form, 'qs' :qs})
					else:
						#print("inside qs is None")
						messages.warning(request, f'No records for selected duration.')
						return redirect('view-attendance-employee')



			
			
				
			else:
				print("invalid username")
				messages.warning(request, f'No such username found.')
				return redirect('view-attendance-employee')


	else:
		

			form=UsernameAndDateForm()
			return render(request,'recognition/view_attendance_employee.html', {'form' : form, 'qs' :qs})




@login_required
def view_my_attendance_employee_login(request):
	if request.user.username in ('admin', 'admin2'):
		return redirect('not-authorised')
	qs=None
	time_qs=None
	present_qs=None
	if request.method=='POST':
		form=DateForm_2(request.POST)
		if form.is_valid():
			u=request.user
			time_qs=Time.objects.filter(user=u)
			present_qs=Present.objects.filter(user=u)
			date_from=form.cleaned_data.get('date_from')
			date_to=form.cleaned_data.get('date_to')
			if date_to < date_from:
					messages.warning(request, f'Invalid date selection.')
					return redirect('view-my-attendance-employee-login')
			else:
					

					time_qs=time_qs.filter(date__gte=date_from).filter(date__lte=date_to).order_by('-date')
					present_qs=present_qs.filter(date__gte=date_from).filter(date__lte=date_to).order_by('-date')
				
					if (len(time_qs)>0 or len(present_qs)>0):
						qs=hours_vs_date_given_employee(present_qs,time_qs,admin=False)
						return render(request,'recognition/view_my_attendance_employee_login.html', {'form' : form, 'qs' :qs})
					else:
						
						messages.warning(request, f'No records for selected duration.')
						return redirect('view-my-attendance-employee-login')
	else:
		

			form=DateForm_2()
			return render(request,'recognition/view_my_attendance_employee_login.html', {'form' : form, 'qs' :qs})

@login_required
def register(request):
    if not request.user.username.startswith(('admin', 'admin2')):
        return redirect('not-authorised')
    
    form = UserRegisterForm()
    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            messages.success(request, f'Account created for {username}!')
            form.save()
            return redirect('dashboard')
    context = {'form': form}
    return render(request, 'recognition/register.html', context)