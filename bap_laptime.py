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


parser = argparse.ArgumentParser(description='BaP laptimer.')
parser.add_argument('id', type=str, help='Team id')
parser.add_argument('--rounds', metavar='R', type=int, help='How many rounds.', default=10,
                    choices=list(range(1, 11)))
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

ret, img = cam.read()

win = pg.GraphicsWindow(title="marker distances")
plot = win.addPlot()
plot.setXRange(0, 400)
plot.setYRange(0, 140)
points = plot.plot()

starttime = time.time()
lap_times = []
num_rounds = args.rounds

camera_matrix = np.load('camera_matrix.npy')
dist_coeffs = np.load('dist_coeff.npy')
print('camera matrix: ', camera_matrix)
print('dist_coeffs: ', dist_coeffs)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
vwriter = cv2.VideoWriter('./out/'+args.id+'.avi', fourcc=fourcc, fps=25, frameSize=(640, 480), isColor=True)

while True:
    try:
        ret, img = cam.read()
        corners, ids, reprojImgPts = aruco.detectMarkers(img, ar_dict)

        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 15.0, camera_matrix, dist_coeffs)
        # print('map_rvecs', map_rvecs)

        dist_cm.append(np.linalg.norm(tvecs[1]-tvecs[0]))
        # print(dist_cm[-1])
        timestamps.append(time.time() - starttime)

        img = aruco.drawDetectedMarkers(img, corners, ids)
        img = aruco.drawAxis(img, camera_matrix, dist_coeffs, rvecs[0], tvecs[0], 5.0)
        img = aruco.drawAxis(img, camera_matrix, dist_coeffs, rvecs[1], tvecs[1], 5.0)
        cv2.line(img, tuple(corners[0][0][0]), tuple(corners[1][0][0]), (0, 0, 255), 1, cv2.LINE_AA)

        smoothed_cm.append(smooth(dist_cm[-5:], 5))
        points.setData(timestamps, smoothed_cm, pen='r')

        cv2.putText(img, "team "+args.id,
                    (550, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1)

        if len(smoothed_cm) > 100:
            mins = argrelmin(np.array(smoothed_cm), order=125)[0]  # for laptime
            if mins.shape[0] >= 2:
                lap_times = []
                for i in range(mins.shape[0]-1):
                    lap_times.append(timestamps[mins[i+1]] - timestamps[mins[i]])

        for i, t in enumerate(lap_times):
            if i == num_rounds:
                break
            cv2.putText(img, "lap {:d}: {:.2f} s".format(i+1, t),
                        (10, 20*(i+1)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1)

        if len(lap_times) > num_rounds:
            mean_time = np.mean(lap_times[:num_rounds])
            print("{:d} lap mean time: {:.2f}".format(num_rounds, mean_time))
            cv2.putText(img, "mean time: {:.2f} s".format(mean_time),
                        (10, 20*(num_rounds+2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1)

            while True:
                cv2.imshow('race', img)
                vwriter.write(img)
                c = cv2.waitKey(30)
                if c == 27:
                    break

            break  # leave outer while loop

        cv2.imshow('race', img)
        vwriter.write(img)

        c = cv2.waitKey(30)
        if c == 27:
            break

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

np.save('./out/laptimes'+args.id+'.npy', np.array([timestamps, smoothed_cm]))
vwriter.release()
cv2.destroyAllWindows()
