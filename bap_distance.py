import numpy as np
import time
import cv2
from cv2 import aruco
import pyqtgraph as pg
from scipy.signal import argrelmin
import argparse


def smooth(y, box_pts):
    if len(y) < box_pts:
        return y[-1]
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    return y_smooth


parser = argparse.ArgumentParser(description='BaP distance measurement.')
parser.add_argument('id', type=str, help='Team id')
args = parser.parse_args()

cam = cv2.VideoCapture(0)
if not cam.isOpened():
    raise Exception("Could not open camera!")

# create detector and set parameters
ar_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
params = aruco.DetectorParameters_create()

dist_px = []
dist_cm = []
timestamps = []
smoothed_cm = []
start_measure = False
finish_measure = False

ret, img = cam.read()

win = pg.GraphicsWindow(title="marker distances")
plot = win.addPlot()
plot.setXRange(0, 50)
plot.setYRange(0, 50)
points = plot.plot()
const = plot.plot()
soll = 20.0
const.setData([0, 400], [soll, soll], pen='b')
offset = None


starttime = time.time()
lap_times = []

camera_matrix = np.load('camera_matrix.npy')
dist_coeffs = np.load('dist_coeff.npy')
print('camera matrix: ', camera_matrix)
print('dist_coeffs: ', dist_coeffs)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
vwriter = cv2.VideoWriter('./out/dist_'+args.id+'.avi', fourcc=fourcc, fps=25, frameSize=(640, 480), isColor=True)

while True:
    try:
        ret, img = cam.read()
        corners, ids, reprojImgPts = aruco.detectMarkers(img, ar_dict)

        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 15.0, camera_matrix, dist_coeffs)
        # print('map_rvecs', map_rvecs)

        dist = np.linalg.norm(tvecs[1]-tvecs[0])

        if start_measure:
            if not offset:
                offset = soll-dist
                starttime = time.time()

            dist_cm.append(dist+offset)
            smoothed_cm.append(smooth(dist_cm[-5:], 5))
            timestamps.append(time.time() - starttime)

            points.setData(timestamps, smoothed_cm, pen='r')

        img = aruco.drawDetectedMarkers(img, corners, ids)
        img = aruco.drawAxis(img, camera_matrix, dist_coeffs, rvecs[0], tvecs[0], 5.0)
        img = aruco.drawAxis(img, camera_matrix, dist_coeffs, rvecs[1], tvecs[1], 5.0)
        cv2.line(img, tuple(corners[0][0][0]), tuple(corners[1][0][0]), (0, 0, 255), 1, cv2.LINE_AA)

        cv2.putText(img, "team " + args.id,
                    (520, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1)

        cv2.putText(img, "dist " + "{:.1f}".format(dist if not offset else dist+offset) + " cm",
                    (520, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1)

        cv2.imshow('race', img)
        vwriter.write(img)

        c = cv2.waitKey(30)
        if c == 27:
            break
        elif c is 32:
            if start_measure is False:
                start_measure = True
            elif start_measure is True:
                start_measure = False
                finish_measure = True

        if finish_measure:
            from sklearn.metrics import mean_squared_error
            from math import sqrt

            rmse_dist = sqrt(mean_squared_error([soll]*len(smoothed_cm), smoothed_cm))
            print("distance rmse {:.1f}".format(rmse_dist))

            cv2.putText(img, "rmse dist " + "{:.1f}".format(rmse_dist) + " cm",
                        (30, 450),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1.0,
                        (0, 0, 255),
                        1)

            while True:
                cv2.imshow('race', img)
                vwriter.write(img)
                c = cv2.waitKey(30)
                if c == 27:
                    break
            break  # leave outer while loop

    except KeyboardInterrupt:
        print("interrupted!")
        break

    except IndexError:
        print("Caught IndexError")
        cv2.imshow('race', img)
        c = cv2.waitKey(30)
        if c == 27:
            break
        continue

    except ValueError as err:
        print("Caught ValueError")
        print(err)
        cv2.imshow('race', img)
        c = cv2.waitKey(30)
        if c == 27:
            break
        continue

    except TypeError:
        print("Caught TypeError")
        cv2.imshow('race', img)
        c = cv2.waitKey(30)
        if c == 27:
            break
        continue

vwriter.release()
cv2.destroyAllWindows()
