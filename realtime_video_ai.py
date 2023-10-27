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

    # #appply this coordinates into video (also used loop to find all the face within one run)
    for (x,y,w,h) in detectedimage:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,250,0), 2)

    #video show
    cv2.imshow("Face detection",frame)

    key=cv2.waitKey(50)

    #stop if q press suing ascii(81 = Q , 113=q)
    if key==81 or key==113:
        break

#to stop the video
video.release()

print("code completed")