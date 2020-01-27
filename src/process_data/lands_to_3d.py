import numpy as np
import pyrealsense2 as rs

from .helpers import LandmarksHelper


class Landmarks3DFinder(LandmarksHelper):
    # camera_attributes = [name for name in dir(rs.intrinsics()) if not name.startswith('_')]
    # TODO: save camera attributesw
    def __init__(self, rs_streaming=None):
        # if rs_streaming is None:
        if rs_streaming is not None:
            depth_profile = rs.video_stream_profile(rs_streaming.profile.get_stream(rs.stream.depth))
            self.depth_intrinsics = depth_profile.get_intrinsics()
            defaul_intrinsics = {i: self.depth_intrinsics.__getattribute__(i) for i in dir(self.depth_intrinsics) if
                                 not i.startswith("__")}
            print(defaul_intrinsics)

        elif rs_streaming is None:
            from pyrealsense2.pyrealsense2 import distortion

            defaul_intrinsics = {'coeffs': [0.0, 0.0, 0.0, 0.0, 0.0],
                                 'fx': 629.5709838867188,
                                 'fy': 629.5709838867188,
                                 'height': 480,
                                 'model': 'brown_conrady',
                                 'ppx': 316.71661376953125,
                                 'ppy': 233.6705780029297,
                                 'width': 640}
            java_users_intrincics = {'coeffs': [0.0, 0.0, 0.0, 0.0, 0.0],
                                     'fx': 630.746337890625,
                                     'fy': 630.746337890625,
                                     'height': 480,
                                     'model': distortion.brown_conrady,
                                     'ppx': 321.7749938964844,
                                     'ppy': 242.17218017578125,
                                     'width': 640}
            defaul_intrinsics = java_users_intrincics
            if not isinstance(defaul_intrinsics['model'], rs.distortion):
                defaul_intrinsics['model'] = distortion.brown_conrady if defaul_intrinsics[
                                                                             'model'] == 'brown_conrady' else ValueError
            self.depth_intrinsics = rs.pyrealsense2.intrinsics()
            [self.depth_intrinsics.__setattr__(i, j) for (i, j) in defaul_intrinsics.items()]

            pass

    def get_correct_i_j(self, depth_frame, i, j):
        return max(0, min(i, depth_frame.shape[1] - 1)), max(0, min(j, depth_frame.shape[0] - 1))

    def get_3d_landmarks(self, depth_frame, shapes, ):
        max_area = 0
        result_3d_coords = []
        for shape in shapes:
            current_area = np.product(np.max(shape, axis=0) - np.min(shape, axis=0))
            if current_area < max_area:
                continue
            else:
                max_area = current_area
            for (i, j) in shape:
                if isinstance(depth_frame, rs.pyrealsense2.frame):
                    depth = depth_frame.as_depth_frame().get_distance(i, j)
                elif isinstance(depth_frame, (np.ndarray, np.generic)):
                    i, j = self.get_correct_i_j(depth_frame, i, j)
                    depth = depth_frame[j, i] / 1000
                else:
                    raise (NotImplementedError)
                result_3d_coords.append(rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, [i, j], depth))
            # sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals=np.asarray(res_dots).reshape(3,-1).tolist()

        return result_3d_coords
