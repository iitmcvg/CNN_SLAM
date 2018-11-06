import cv2
import numpy as np 
import glob

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) #Termination criteria
objp = np.zeros((7*6,3),np.float32) #3Dimensions, 7x6 points
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)  #.T is transpose. #first two columns of objp = 0th to 6th row and 0th to 5tf column of mgrid transposed and reshaped to have two columns
objpoints = []
imgpoints = []
#images = ''
#img = glob.glob('*.jpg')

for fname in range(1,15):
	img = cv2.imread('Images2/im'+str(fname)+'.jpg')
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	ret,corners = cv2.findChessboardCorners(gray,(7,6),None) #Returns corner points
	if ret == True:
		objpoints.append(objp)
		cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria) #Does corner refinement
		imgpoints.append(corners)
		cv2.drawChessboardCorners(img,(7,6),corners,ret)
	#cv2.imshow('dawd',img)
	#cv2.waitKey(500)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None) #Calibrates and returns everything needed

#For undistortion
img = cv2.imread('Images2/im1.jpg')
h,w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h)) #Refines the camera matrix: can pass 1 or 0

dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
x,y,w,h = roi
dst = dst[y:y+h,x:x+w]
#cv2.imshow('Original',img)
#cv2.imshow('After Calibration',dst)
#cv2.waitKey(500)


"""
mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
"""

mean_error = 0
for i in xrange(len(objpoints)):
	imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
	error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
	mean_error += error
print "total error: ", mean_error/len(objpoints)

print "distortion"
print dist
