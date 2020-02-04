from datetime import datetime as dt

import cv2
import joblib
import numpy as np

from ext.realsense import RealSense
from src.antispoof_processor import ProcessAntiSpoof

path_to_source_coordinates = "./src/io/source_coords.tsv"
path_to_model = './src/models/models9.joblib'

source_coords = np.loadtxt(path_to_source_coordinates)
source_coords = source_coords.T.tolist()

process_antispoof = ProcessAntiSpoof(source_coords=source_coords, mode='prediction_from_camera',temporal_smoothing=1)
clfs = joblib.load(path_to_model)
# realsense = RealSense(online=True)
# realsense.start()

theshold_of_spoof = 0.7
accc=[]

fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
vw = cv2.VideoWriter('/home/maksym/Documents/Antispoofing based on RGB+Depth camera' '.avi', fourcc, 12, (1280,720))
while True:
    try:
    # shot = realsense.get_image_depth_color()
    # image = np.asarray(shot["image"], dtype=np.uint8)
    # depth = np.asarray(shot["depth"])
    #
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #
    # cv2.imshow("image", image)
        st = dt.now()
        return_code, prob,dets, shapes,image = process_antispoof.do_the_work(draw_icp=False, classifier=clfs[-1])
        if prob:
            accc.append(prob)
        if (np.mean(accc[-3:])  > theshold_of_spoof) or (accc[-1]>0.9):
            print("Bad face")
            color=(0,0,255)
        else:
            color = (0, 255, 0)
        image = cv2.rectangle(image,( dets[0].left()-int(dets[0].width()/5), dets[0].top()-int(dets[0].width()/5)),
                              ( dets[0].right()+int(dets[0].width()/5), dets[0].bottom()+int(dets[0].width()/5)), color, 2)
        cv2.putText(image,"Fake" if color[1]==0 else "Good",( dets[0].left()-int(dets[0].width()/5), dets[0].top()-int(dets[0].width()/4)),
                    2, 2, color, thickness=2, lineType=3)
        cv2.imshow('img', image)
        print(f'Prof of spoof: {prob :.2f}, time taken: {str((dt.now() - st).total_seconds())}') if prob else None

        vw.write(image)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    except:
        continue
vw.release()
