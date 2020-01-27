import numpy as np
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D


class LandmarksHelper():
    @staticmethod
    def normalize_single_array(array_input, noze_point=33, eyes_points=[39, 42]):
        centered_to_noze = (np.asanyarray(array_input)) - (np.asanyarray(array_input))[noze_point]
        # dist_between_eyes=np.sqrt(np.sum((centered_to_noze[eyes_points[0]]-centered_to_noze[eyes_points[1]]) ** 2, axis=0))
        # centered_to_noze_norlalize_do_dist=centered_to_noze/dist_between_eyes
        return centered_to_noze.tolist()

    @staticmethod
    def normalize_landmars(source, target):
        if len(source) == 3:
            source = np.asanyarray(source).T.tolist()
        if len(target) == 3:
            target = np.asanyarray(target).T.tolist()
        return LandmarksHelper.normalize_single_array(source), LandmarksHelper.normalize_single_array(target)

    @staticmethod
    def filter_landmarks(source, target, type='partial'):
        if type == "partial":
            landmarks_to_include = [i - 1 for i in range(1, 69) if
                                    i not in ([i for i in range(1, 6)] +
                                              [i for i in range(13, 18)] +
                                              [i for i in range(6, 13)] +
                                              [i for i in range(18, 22)] +
                                              [i for i in range(24, 28)])]
        elif type == "all":
            landmarks_to_include = [i - 1 for i in range(1, 69)]
        else:
            raise NotImplementedError
        source = np.asarray(source)[
            landmarks_to_include].T.tolist()

        target = np.asarray(target)[
            landmarks_to_include].T.tolist()

        # mean = np.mean(target[2], axis=0)
        # sd = np.std(target[2], axis=0)

        # final_list = [(x > mean - 1.6 * sd)  for x in target[2]]
        final_list = [True for x in target[2]]

        target_points = ((np.asanyarray(target)).T[final_list].T).tolist()
        source_points = ((np.asanyarray(source)).T[final_list].T).tolist()

        return source_points, target_points

    @staticmethod
    def vizualize_3d_dots(x_current, y_current, z_current):
        fig = pyplot.figure()
        ax = Axes3D(fig)
        ax.scatter(x_current, y_current, z_current)
        pyplot.show()
