import cv2
import numpy as np
import pyrealsense2 as rs


class DepthFiltering():
    def __init__(self, temporal_smoothing=5):
        self.temporal_smoothing = temporal_smoothing
        self.dec_filter = rs.decimation_filter()  # Decimation - reduces depth frame density
        self.spat_filter = rs.spatial_filter()
        self.spat_filter.set_option(rs.option.holes_fill, 3)
        self.temp_filter = rs.temporal_filter()  # Temporal   - reduces temporal noise
        self.depth_to_disparity = rs.disparity_transform(True)
        self.disparity_to_depth = rs.disparity_transform(False)
        self.hole_filling = rs.hole_filling_filter()

    def apply_filters(self, depth_frame):
        depth_frame = self.depth_to_disparity.process(depth_frame)
        depth_frame = self.spat_filter.process(depth_frame)
        depth_frame = self.temp_filter.process(depth_frame)
        depth_frame = self.disparity_to_depth.process(depth_frame)
        depth_frame = self.hole_filling.process(depth_frame)
        return depth_frame


class Visualizer():
    @staticmethod
    def visualize_depth(depth):
        depth_image = np.asanyarray(depth.get_data()) is depth if not isinstance(depth,
                                                                                 (np.ndarray, np.generic)) else depth
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        cv2.namedWindow('depth_colormap', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('depth_colormap', depth_colormap)

    @staticmethod
    def visualize_img(img):
        img = np.asanyarray(img.get_data()) is img if not isinstance(img, (np.ndarray, np.generic)) else img
        cv2.namedWindow('img', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('img', img)

    @staticmethod
    def visualize_aligned_depth_to_image(aligned_depth_frame, color_frame, profile):

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        # print("Depth Scale is: ", depth_scale)

        # We will be removing the background of objects more than
        #  clipping_distance_in_meters meters away
        clipping_distance_in_meters = 1  # 1 meter
        clipping_distance = clipping_distance_in_meters / depth_scale
        if isinstance(aligned_depth_frame, (np.ndarray, np.generic)):
            depth_image = aligned_depth_frame
            color_image = color_frame
        else:
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack(
            (depth_image, depth_image, depth_image))  # depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

        # Render images
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((bg_removed, depth_colormap))
        cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Align Example', images)
