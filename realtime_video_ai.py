import cv2

#load pre-trained data from haarcascade
trained_face_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


#video for detection
video=cv2.VideoCapture("Let's Dance 4 Kids 30 sec.mp4")

while True:

    succesful_frame,frame=video.read()

    #change video to grayscale because this support bw image
    bwimage=cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    #code for detect face coordinates
    detectedimage=trained_face_data.detectMultiScale(bwimage)

   

#to stop the video
video.release()

print("code completed")