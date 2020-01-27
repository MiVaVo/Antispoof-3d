import copy
import functools

import numpy as np
import open3d as o3d


def points_to_pointcloud(func):
    @functools.wraps(func)
    def ensure_is_pointcloud(*args, **kwargs):
        if 'left_points_array' in kwargs.keys() and 'right_points_array' in kwargs.keys():
            if not isinstance(kwargs['left_points_array'], o3d.geometry.PointCloud):
                kwargs['left_points_array'] = xyz_to_pointcloud(kwargs['left_points_array'])
                kwargs['right_points_array'] = xyz_to_pointcloud(kwargs['right_points_array'])
        else:
            if isinstance(args[0], RigidICPRegistration):
                increment = 1
            else:
                increment = 0
            if not isinstance(args[0 + increment], o3d.geometry.PointCloud):
                args = list(args)
                args[0 + increment] = xyz_to_pointcloud(args[0 + increment])
                args[1 + increment] = xyz_to_pointcloud(args[1 + increment])
                args = tuple(args)
        return func(*args, **kwargs)

    return ensure_is_pointcloud


def xyz_to_pointcloud(*args):
    if len(args) == 3:
        array = np.asarray([args[0], args[1], args[2]]).T
    else:
        array = args[0]
        array = np.asarray(array)

        any_shape_is_3 = np.asarray(list(array.shape)) == 3
        if np.any(any_shape_is_3):
            array = array.T if any_shape_is_3[0] else array
    point_cloud_instance = o3d.geometry.PointCloud()
    point_cloud_instance.points = o3d.utility.Vector3dVector(array)
    return point_cloud_instance


class RigidICPRegistration():
    def __init__(self):
        self.threshold = 0.2
        self.trans_init = np.asarray([[1., 0., 0., 0.],
                                      [0., 1., 0., 0.],
                                      [0., 0., 1., 0.],
                                      [0., 0., 0., 1.]])
        self.estimation = o3d.registration.TransformationEstimationPointToPoint(with_scaling=True)
        self.criteria = o3d.registration.ICPConvergenceCriteria(max_iteration=200)

    @points_to_pointcloud
    def register(self, left_points_array, right_points_array):
        print(np.asanyarray(left_points_array.points))
        print(np.asanyarray(right_points_array.points))

        self.reg_p2p = o3d.registration.registration_icp(
            left_points_array,
            right_points_array,
            self.threshold,
            self.trans_init,
            self.estimation,
            self.criteria)


@points_to_pointcloud
def draw_registration_result(left_points_array, right_points_array, transformation):
    left_points_pointcloud = copy.deepcopy(left_points_array)
    right_points_pointcloud = copy.deepcopy(right_points_array)
    left_points_pointcloud.paint_uniform_color([1, 0.706, 0])
    right_points_pointcloud.paint_uniform_color([0, 0.651, 0.929])
    if transformation is not None:
        left_points_pointcloud.transform(transformation)
    o3d.visualization.draw_geometries([left_points_pointcloud, right_points_pointcloud], width=640, height=480)


def concatenate(**words):
    result = ""
    for arg in words.values():
        result += arg
    return result
