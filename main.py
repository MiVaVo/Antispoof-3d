import joblib
import numpy as np
import cv2
from datetime import datetime as dt
from ext.realsense import RealSense
from src.antispoof_processor import ProcessAntiSpoof

path_to_source_coordinates="./src/io/source_coords.tsv"
path_to_model='./src/models/models9.joblib'

source_coords=np.loadtxt(path_to_source_coordinates)
source_coords=source_coords.T.tolist()

process_antispoof = ProcessAntiSpoof(source_coords=source_coords,mode='prediction_from_image_and_depth')
clfs = joblib.load(path_to_model)
realsense = RealSense(online=True)
realsense.start()


theshold_of_spoof=0.4
while True:
    shot = realsense.get_image_depth_color()
    image=np.asarray(shot["image"], dtype=np.uint8)
    depth=np.asarray(shot["depth"])

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imshow("image", image)
    st=dt.now()
    return_code, prob = process_antispoof.do_the_work(draw_icp=False, classifier=clfs[-1], image=image,
                                                        depth=depth)
    print(f'Prof of spoof: {prob :.2f}, time taken: {str((dt.now() -st).total_seconds()) }') if prob else None
    if prob and prob > theshold_of_spoof:
        print("Bad face")
    key=cv2.waitKey(1)
    if key==ord('q'):
        break

