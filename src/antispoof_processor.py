from __future__ import absolute_import

import os
import re
from datetime import datetime as dt
from pickle import UnpicklingError

import numpy as np

from src import icp
from src.icp import draw_registration_result
from src.process_data import Landmarks3DFinder
from src.process_data import LandmarksFinderDlib
from src.streaming import RSStreaming
from src.utils import save_ds, absoluteFilePaths, get_image_depth, classify
from datetime import datetime as dt

class ProcessAntiSpoof():
    def __init__(self, mode, source_coords, path_to_folder=None):
        self.path_to_folder = path_to_folder
        self.dlib_lands = LandmarksFinderDlib()
        self.source_coords = source_coords
        self.mode = mode
        self.path_to_save_data=None
        if self.mode in ['data_processing_to_features', 'prediction_from_folder', 'prediction_from_image_and_depth']:
            self.landmarks_3d = Landmarks3DFinder(rs_streaming=None)

            if path_to_folder is None and self.mode != 'prediction_from_image_and_depth':
                raise ValueError("path_to_folder")
            else:
                self.path_gen = absoluteFilePaths(self.path_to_folder)

            if self.mode in ['data_processing_to_features']:
                self.features = []

        elif self.mode in ['prediction_from_camera', 'visualize', 'data_collection_from_camera']:
            self.rs_streaming = RSStreaming(temporal_smoothing=3)
            self.landmarks_3d = Landmarks3DFinder(rs_streaming=self.rs_streaming)

        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented")

    def get_frameset(self):
        if self.mode in ['data_processing_to_features', 'prediction_from_folder']:
            try:
                image, depth_frame = get_image_depth(next(self.path_gen))
            except StopIteration:
                return None, None

        elif self.mode in ['prediction_from_camera', 'data_collection_from_camera', 'visualize']:
            depth_frame, image_frame = self.rs_streaming.get_filtered_frameset()
            image = np.asanyarray(image_frame.get_data())
            depth_frame = np.asanyarray(depth_frame.get_data())
        elif self.mode in ['prediction_from_image_and_depth']:
            raise ImportError(f"Mode {self.mode} does not require gettinf frameset as it already should exist")
        return image, depth_frame

    def get_dets_shapes(self, image):
        dets, shapes = self.dlib_lands.get_landmarks(image)
        return dets, shapes

    def get_distances_from_icp(self, shapes, depth_frame, source_raw, draw_icp=False, landmarks_fildering_type='all'):
        # print(len(dets))
        target_raw = self.landmarks_3d.get_3d_landmarks(depth_frame, shapes)
        source_raw, target_raw = self.landmarks_3d.normalize_landmars(source_raw, target_raw)

        # print(target_3d_coords)
        source, current = Landmarks3DFinder.filter_landmarks(target_raw, source_raw, type=landmarks_fildering_type)
        # draw_registration_result(current, source, None)
        T, distances, iterations = icp.icp(np.asanyarray(current).T, np.asanyarray(source).T, max_iterations=100,
                                           tolerance=0.00001)
        if draw_icp:
            draw_registration_result(current, source, T)
        return T, distances, iterations

    def do_the_work(self, draw_icp=False, **kwargs):

        if self.mode in ['prediction_from_image_and_depth']:
            image = kwargs['image']
            depth = kwargs['depth']
            dets, shapes = self.get_dets_shapes(image)
            if not shapes:
                print("No face found")
                return 0, None
            T, distances, iterations = self.get_distances_from_icp(shapes, depth, self.source_coords, draw_icp=draw_icp)
            classifier = kwargs['classifier']
            prob_of_fake = classify(classifier, distances)
            return 0, prob_of_fake

        if self.mode in ['data_collection_from_camera']:

            image, depth = self.get_frameset()
            RSStreaming.visualize_img(image)
            RSStreaming.visualize_depth(depth)

            data_collection_type = kwargs['ds_type']
            data_collection_mode = kwargs['ds_mode']
            if data_collection_type not in ["true",'fake']:
                raise ValueError
            if data_collection_mode not in ['train','test']:
                raise ValueError
            prob_of_save = kwargs['prob_of_save'] if "prob_of_save" in kwargs.keys() else 0.5

            appendix_to_string=str(dt.now().strftime("%Y%m%d%H%M%S"))
            if self.path_to_save_data is None:
                self.path_to_save_data = f'./ext/ds/{data_collection_type }_{data_collection_mode}_'+appendix_to_string
            if not os.path.exists(self.path_to_save_data):
                os.makedirs(self.path_to_save_data)
            save_ds(self.path_to_save_data, image, depth) if np.random.rand() > prob_of_save else None

            return 0, None

        elif self.mode in ['visualize']:
            image, depth = self.get_frameset()
            RSStreaming.visualize_img(image)
            RSStreaming.visualize_depth(depth)


        elif self.mode in ['prediction_from_camera', 'prediction_from_folder']:
            image, depth = self.get_frameset()
            if image is None:
                return 1, None
            RSStreaming.visualize_img(image)
            RSStreaming.visualize_depth(depth)
            dets, shapes = self.get_dets_shapes(image)
            if not shapes:
                print("No face found")
                return 0, None
            T, distances, iterations = self.get_distances_from_icp(shapes, depth, self.source_coords, draw_icp=draw_icp)
            classifier = kwargs['classifier']
            prob_of_fake = classify(classifier, distances)
            return 0, prob_of_fake

        elif self.mode in ['data_processing_to_features']:
            try:
                image, depth = self.get_frameset()
            except EOFError:
                return 0, None
            except UnpicklingError:
                return 0, None
            csv_ds_folder = "./ext/ds/csv_ds"
            if image is None:
                np.savetxt(
                    f'{os.path.join(csv_ds_folder, self.path_to_folder.split("/")[-1])}_{re.sub("[^0-9]", "", str(dt.now()))}.csv',
                    np.asanyarray(self.features), delimiter=';')
                return 1, None
            RSStreaming.visualize_img(image)
            RSStreaming.visualize_depth(depth)
            dets, shapes = self.get_dets_shapes(image)
            if not shapes:
                print("No face found")
                return 0, None
            T, distances, iterations = self.get_distances_from_icp(shapes, depth, self.source_coords, draw_icp=draw_icp)
            self.features.append(distances)
            return 0, None
