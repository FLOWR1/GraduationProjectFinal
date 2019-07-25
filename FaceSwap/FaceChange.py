import dlib
import cv2
import numpy as np

import models
import NonLinearLeastSquares
import ImageProcessing

from drawing import *

import FaceRendering
import utils

import timeit
#
print("Press T to draw the keypoints and the 3D model")
print("Press R to start recording to a video file")

#you need to download shape_predictor_68_face_landmarks.dat from the link below and unpack it where the solution file is
#http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2

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
#
projectionModel = models.OrthographicProjectionBlendshapes(blendshapes.shape[0])

modelParams = None
lockedTranslation = False
drawOverlay = False
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("../1.mp4")

writer = None
cameraImg = cap.read()[1]   # face swap하여 붙일 영상의 img
textureImg = cv2.VideoCapture(0).read()[1] #cv2.imread(image_name)

print("광고영상 shape : ",cameraImg.shape[1],"*",cameraImg.shape[0])
print("카메라 캡쳐영상 shape : ",textureImg.shape[1],"*",textureImg.shape[0])

while True:
	textureImg = cv2.VideoCapture(0).read()[1]
	dets = detector(textureImg, 1) #처음 camera로 촬영한 캡쳐를 넣어서 얼굴을 찾음.
	if len(dets) > 0:
		print("detected")
		break;
	else:
		print("now detecting")

# 찍은 영상의 캡쳐를 3D로 재구성하여 합침
textureCoords = utils.getFaceTextureCoords(textureImg, mean3DShape, blendshapes, idxs2D, idxs3D, detector, predictor)
# 찍은 얼굴의 데이터를 영상의 얼굴에 rendering
renderer = FaceRendering.FaceRenderer(cameraImg, textureImg, textureCoords, mesh)

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
                #print(3, float(meanTime[2][0]/meanTime[2][1]))


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

        cv2.imshow('Converted Video', cameraImg)

        key = cv2.waitKey(1)

        if key == 27:
            break
        if key == ord('t'):
            drawOverlay = not drawOverlay
        if key == ord('r'):
            if writer is None:
                print("Starting video writer")
                writer = cv2.VideoWriter("../out.mp4", cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 13, (cameraImg.shape[1], cameraImg.shape[0]))

                if writer.isOpened():
                    print("Writer succesfully opened")
                else:
                    writer = None
                    print("Writer opening failed")
            else:
                print("Stopping video writer")
                writer.release()
                writer = None
