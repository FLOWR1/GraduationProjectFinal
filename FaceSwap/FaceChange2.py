import cv2
import numpy as np
import threading
import sys
import os

import dlib
import models
import NonLinearLeastSquares
import ImageProcessing
from drawing import *
import FaceRendering
import utils
import timeit

import time

#INTRO_VID_PATH = '../ROEM 2014 Spring SUZY 1080p.mp4'
BACKGROUND_VID_PATH = ['../1.mp4', '../2.mp4', '../3.mp4', '../4.mp4', '../5.mp4']
BACKGROUND_VID_PATH_NUM = 0
SAVE_VID_PATH = '../out.avi'
MOVEMENT_THRESHOLD = 1.2#1.7 #higher is bigger movement
GUIDE_SHOW_TIME = 7.0 #seconds
GUIDE_WAIT_TIME = 4.0
VIDEO_CAPTURE_CAM_NUM = 0

#loading the keypoint detection model, the image and the 3D model
predictor_path = "../shape_predictor_68_face_landmarks.dat"
#the smaller this value gets the faster the detection will work
#if it is too small, the user's face might not be detected
maxImageSizeForDetection = 320

#카메라로 찍히는 영상에서 얼굴을 찾는 detector
detector = dlib.get_frontal_face_detector()
#사람 얼굴을 찾는 입과 눈의 구석, 코의 끝과 같은 중요한 얼굴 표식의 위치를 식별하는 점들의 집합
predictor = dlib.shape_predictor(predictor_path)
# candide = 3D face model source
# mean3Dshape : 얼굴의 중립상태에 해당하는 정점 리스트
# blendshapes : 중립상태인 얼굴에서 추가하여 수정할 수 있는 얼굴
	# ex) 미소, 눈썹 올라가는 부분
	# candide에 정의된 애니메이션 Units에서 파생된다.
# mesh : Candide가 얼굴 목록으로 제공한 원래의 mesh
# idxs3D, idxs2D: Candide 모델(idxs3D)과 얼굴 정렬점 세트(idxs2D)사이에 해당하는 지점들의 인덱스들이다.
mean3DShape, blendshapes, mesh, idxs3D, idxs2D = utils.load3DFaceModel("../candide.npz")
projectionModel = models.OrthographicProjectionBlendshapes(blendshapes.shape[0])

def movement_detection():
	ret, frame1 = cv2.VideoCapture(VIDEO_CAPTURE_CAM_NUM).read()
	frame1 =  cv2.resize(frame1, (100, 50))
	prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
	hsv = np.zeros_like(frame1)
	hsv[...,1] = 255

	mov_check_cap = cv2.VideoCapture(VIDEO_CAPTURE_CAM_NUM)
	while(True):
		ret, frame2 = mov_check_cap.read()
		frame2 =  cv2.resize(frame2, (100, 50))
		next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
		flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
		mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
		mag_mat = np.matrix(mag)
		cam_movement = mag_mat.mean()
		if cam_movement > MOVEMENT_THRESHOLD:
			break

		prvs = next

def video():
	t = threading.Thread(target=movement_detection)
	t.start()
	cap_intro_vid = cv2.VideoCapture(BACKGROUND_VID_PATH[BACKGROUND_VID_PATH_NUM])

	while(cap_intro_vid.isOpened()):
		ret, frame = cap_intro_vid.read()
		cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
		cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, 1)
		cv2.imshow('frame',frame)

		if not t.isAlive():
			guide_facechange()
			break

		key = cv2.waitKey(1)
		if key == 27 & 0xFF == ord('q'):
			break


def guide_facechange():
	def tmp():
		print("")
	t = threading.Timer(GUIDE_SHOW_TIME, tmp)
	t.start()

	t_wait = threading.Timer(GUIDE_WAIT_TIME, tmp)
	t_wait.start()


	print("Press T to draw the keypoints and the 3D model")
	print("Press W to start recording to a video file")
	print("Press R to restart")
	print("Press Q or ESC to Quit")

	modelParams = None
	lockedTranslation = False
	drawOverlay = False

	global BACKGROUND_VID_PATH_NUM
	#cap = cv2.VideoCapture(VIDEO_CAPTURE_CAM_NUM)
	cap = cv2.VideoCapture(BACKGROUND_VID_PATH[BACKGROUND_VID_PATH_NUM])

	writer = None
	cameraImg = cap.read()[1]   # face swap하여 붙일 영상의 img
	textureImg = cv2.VideoCapture(VIDEO_CAPTURE_CAM_NUM).read()[1]

	#print("광고영상 shape : \t\t",cameraImg.shape[1],cameraImg.shape[0])
	#print("카메라 캡쳐영상 shape : ",textureImg.shape[1],textureImg.shape[0])

	###### face detection with guide
	cap_guide_cam = cv2.VideoCapture(VIDEO_CAPTURE_CAM_NUM)

	if (cap_guide_cam.isOpened() == False):
		print("Unable to read camera feed")

	frame_width = int(cap_guide_cam.get(3))
	frame_height = int(cap_guide_cam.get(4))
	str="match your face"
	str2="O"
	str3="ATTENTION"
	while(True):
		ret, frame = cap_guide_cam.read()
		frame_org = frame

		cv2.putText(frame,str,(int(frame_width/3),int(frame_height/6)),cv2.FONT_HERSHEY_SIMPLEX,int(frame_width/600),(0,0,0),int(frame_width/300))
		cv2.putText(frame,str2,(int(frame_width/3),int(frame_width/2)),cv2.FONT_HERSHEY_SIMPLEX,int(frame_width/60),(0,0,255),int(frame_width/300))
		cv2.putText(frame,str3,(int((frame_width*2)/3),int((frame_height*2)/3)),cv2.FONT_HERSHEY_SIMPLEX,int(frame_width/650),(0,0,0),int(frame_width/300))

		cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
		cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, 1)
		cv2.imshow('frame',frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

		if not t_wait.isAlive():
			dets = detector(frame_org, 1) #처음 camera로 촬영한 캡쳐를 넣어서 얼굴을 찾음.

			if len(dets) > 0:
				print("detected")
				break
			else:
				print("now detecting")

		if not t.isAlive():
			video()

	try:
		# 찍은 영상의 캡쳐를 3D로 재구성하여 합침
		textureCoords = utils.getFaceTextureCoords(textureImg, mean3DShape, blendshapes, idxs2D, idxs3D, detector, predictor)
		# 찍은 얼굴의 데이터를 영상의 얼굴에 rendering
		renderer = FaceRendering.FaceRenderer(cameraImg, textureImg, textureCoords, mesh)
	except:
		BACKGROUND_VID_PATH_NUM = 1
		guide_facechange()

	doProcess=False
	meanTime=[[0]*4 for i in range(4)]

	while True:
		#영상 캡쳐
		cameraImg = cap.read()[1]
		shapes2D = utils.getFaceKeypoints(cameraImg, detector, predictor, maxImageSizeForDetection)

		doProcess = not doProcess

		if doProcess is not True:
			continue
		else:
			if shapes2D is not None:
				for shape2D in shapes2D:
					start = timeit.default_timer()
					#3D model parameter initialization
					modelParams = projectionModel.getInitialParameters(mean3DShape[:, idxs3D], shape2D[:, idxs2D])
					stop = timeit.default_timer()
					meanTime[0][0]+=stop-start
					meanTime[0][1]+=1
					#print(1, float(meanTime[0][0]/meanTime[0][1]))

					start = timeit.default_timer()
					#3D model parameter optimization
					modelParams = NonLinearLeastSquares.GaussNewton(modelParams, projectionModel.residual, projectionModel.jacobian, ([mean3DShape[:, idxs3D], blendshapes[:, :, idxs3D]], shape2D[:, idxs2D]), verbose=0)
					stop = timeit.default_timer()
					meanTime[1][0]+=stop-start
					meanTime[1][1]+=1
					#print(2, float(meanTime[1][0]/meanTime[1][1]))

					start = timeit.default_timer()
					#rendering the model to an image
					#다듬기
					shape3D = utils.getShape3D(mean3DShape, blendshapes, modelParams)
					renderedImg = renderer.render(shape3D)
					stop = timeit.default_timer()
					meanTime[2][0]+=stop-start
					meanTime[2][1]+=1
					#print(3, float(meanTime[2][0]/meanTime[2][1]))\

					start = timeit.default_timer()
					#blending of the rendered face with the image
					mask = np.copy(renderedImg[:, :, 0])
					renderedImg = ImageProcessing.colorTransfer(cameraImg, renderedImg, mask)
					cameraImg = ImageProcessing.blendImages(renderedImg, cameraImg, mask)
					stop = timeit.default_timer()
					meanTime[3][0] += stop - start
					meanTime[3][1] += 1
					#print(4, float(meanTime[3][0] / meanTime[3][1]))

					#drawing of the mesh and keypoints
					# 't'를 누를 때, 적용. facepoint가 표시됨.
					if drawOverlay:
						drawPoints(cameraImg, shape2D.T)
						drawProjectedShape(cameraImg, [mean3DShape, blendshapes], projectionModel, mesh, modelParams, lockedTranslation)

			if writer is not None:
				writer.write(cameraImg)

			cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
			cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, 1)
			cv2.imshow('image',cameraImg)

			key = cv2.waitKey(1)

			if key == 27 or key == ord('q'):
				break
			if key == ord('t'):
				drawOverlay = not drawOverlay
			if key == ord('r'):
				cv2.destroyAllWindows()
				video()
			if key == ord('1'):
				cv2.destroyAllWindows()
				BACKGROUND_VID_PATH_NUM = 0
				video()
				break
			if key == ord('2'):
				cv2.destroyAllWindows()
				BACKGROUND_VID_PATH_NUM = 1
				video()
				break
			if key == ord('3'):
				cv2.destroyAllWindows()
				BACKGROUND_VID_PATH_NUM = 2
				video()
				break
			if key == ord('4'):
				cv2.destroyAllWindows()
				BACKGROUND_VID_PATH_NUM = 3
				video()
				break
			if key == ord('5'):
				cv2.destroyAllWindows()
				BACKGROUND_VID_PATH_NUM = 4
				video()
				break

			if key == ord('w'):
				if writer is None:
					print("Starting video writer")
					writer = cv2.VideoWriter(SAVE_VID_PATH, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 13, (cameraImg.shape[1], cameraImg.shape[0]))

					if writer.isOpened():
						print("Writer succesfully opened")
					else:
						writer = None
						print("Writer opening failed")
				else:
					print("Stopping video writer")
					writer.release()
					writer = None

	cap.release()
	cap_intro_vid.release()
	cap_guide_cam.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	video()
