from functools import wraps

import cv2
import dlib
import numpy as np

def take_care_of_large_img(f):
    @wraps(f)
    def wrapper(*args):
        image_height=args[1].shape[0]
        if image_height==720:
            downsize_factor = 4
        elif image_height==480:
            downsize_factor = 2
        else:
            downsize_factor = 1

        new_tuple=[None,None]
        new_tuple[0]=args[0]
        new_tuple[1]=cv2.resize(args[1],None,fx=1/downsize_factor,fy=1/downsize_factor)
        args=tuple(new_tuple)
        result = f(*args)
        old_rectangles=result[0]
        old_dots=result[1]
        new_dots=[]
        for dots_array in old_dots:
            new_dots.append(dots_array*downsize_factor)
        new_rectangles=dlib.rectangles([dlib.rectangle(int(rect.left()*downsize_factor),
                                        int(rect.top()*downsize_factor),
                                        int(rect.right()*downsize_factor),
                                        int(rect.bottom()*downsize_factor)) for rect in old_rectangles])
        result=new_rectangles,new_dots
        return result
    return wrapper


class LandmarksFinderDlib():

    def __init__(self):
        super().__init__()
        path_to_predictor = './src/io/configs/shape_predictor_68_face_landmarks.dat'
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(path_to_predictor)

    def shape_to_np(self, shape, dtype="int"):
        # initialize the list of (x, y)-coordinates
        coords = np.zeros((68, 2), dtype=dtype)

        # loop over the 68 facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)

        # return the list of (x, y)-coordinates
        return coords

    def rect_to_bb(self, rect):
        # take a bounding predicted by dlib and convert it
        # to the format (x, y, w, h) as we would normally do
        # with OpenCV
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y

        # return a tuple of (x, y, w, h)
        return (x, y, w, h)

    @take_care_of_large_img
    def get_landmarks(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets = self.detector(img, 1)
        shapes = []
        for (i, rect) in enumerate(dets):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = self.predictor(img, rect)
            shape = self.shape_to_np(shape)
            shapes.append(shape)
        return dets, shapes

    def visualize_landmarks(self, img, dets, shapes):
        i = 0
        for rect, shape in zip(dets, shapes):
            (x, y, w, h) = self.rect_to_bb(rect)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # show the face number
            cv2.putText(img, "Face #{}".format(i + 1), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
            i += 1
