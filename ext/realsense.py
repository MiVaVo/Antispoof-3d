import os

import numpy as np
import pyrealsense2 as rs

# Детекция и векторизация из пилота для Пятерочки
try:
    from matching.insightface import ArcFaceModel
except:
    pass

def resource_path(path):
    root = os.path.dirname(__file__)
    return os.path.join(root, path)


class NoFacesDetected(ValueError):
    pass


class RealSense:
    def __init__(self, smooth=3, online=True):
        self.smooth = smooth
        self.online = online

        if not self.online:
            # noinspection PyTypeChecker
            sample: dict = np.load(
                resource_path("resource/sample.npy"), allow_pickle=True
            )
            self.sample = {
                "image": sample["image"].tolist(),
                "depth": sample["depth"].tolist(),
                "color": sample["color"].tolist(),
            }
            return

        # Пайплайн получает картинки с камеры
        # noinspection PyArgumentList
        self.pipe = rs.pipeline()

        # Большее выровненное друг на друга разрешение, кажется, не получить
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

        # Операция для выравнивания 2д и 3д друг на друга
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        # Переводит глубину в цветное представление от холодного к теплому (не критично)
        self.colorizer = rs.colorizer()

        # Пространственное сглаживание
        self.spat_filter = rs.spatial_filter()
        self.spat_filter.set_option(rs.option.holes_fill, 3)

        # Сглаживает карту глубины по нескольким кадрам
        self.temp_filter = rs.temporal_filter()  # Temporal   - reduces temporal noise

        # Сглаживает карту глубины на основе диспаритета двух 2д камер
        self.depth_to_disparity = rs.disparity_transform(True)
        self.disparity_to_depth = rs.disparity_transform(False)

        # Затирает провалы глубины по картинке из-за шумов
        self.hole_filling = rs.hole_filling_filter()



    def start(self):
        if self.online:
            self.pipe.start(self.config)

    def stop(self):
        if self.online:
            self.pipe.stop()

    def get_image_depth_color(self, *_):
        if not self.online:
            return self.sample.copy()

        # self.start()

        images = []
        depths = []

        # Получение выровненных 2д и глубины во внутреннем формате
        for x in range(self.smooth):
            frames = self.pipe.wait_for_frames()
            aligned = self.align.process(frames)
            images.append(aligned.get_color_frame())
            depths.append(aligned.get_depth_frame())

        depth = None
        # Применение всех фильтров из конструктора
        for x in range(self.smooth):
            depth = depths[x]
            depth = self.depth_to_disparity.process(depth)
            depth = self.spat_filter.process(depth)
            depth = self.temp_filter.process(depth)
            depth = self.disparity_to_depth.process(depth)
            depth = self.hole_filling.process(depth)

        # Перевод внутреннего представления либы в обычные массивы
        image = np.asanyarray(images[-1].get_data())
        color = np.asanyarray(self.colorizer.colorize(depth).get_data())
        depth = np.asanyarray(depth.get_data())

        # plt.figure(figsize=(12, 6))
        # plt.imshow(np.hstack([1 - np.stack([depth] * 3, -1) / 65535, image / 255]))
        # plt.imshow(np.hstack([depth / 255, image / 255]))
        # plt.show()

        # self.stop()

        return {
            "image": image.tolist(),
            "depth": depth.tolist(),
            "color": color.tolist(),
        }

        # return np.load("data.npy", allow_pickle=True).item()

    @staticmethod
    def get_depth_crop(image, depth, model):
        """
        Получение кропа карты глубины, соответствующего лицу
        """

        _, bbox, landmarks = model.detect(image[..., ::-1])

        if bbox is None:
            raise NoFacesDetected()

        bbox = bbox.astype(int)
        # ccrop = color[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        dcrop = depth[bbox[1] : bbox[3], bbox[0] : bbox[2]]
        dcrop[dcrop > 1000] = 1000

        # plt.imshow(np.hstack([ccrop / 255, 1 - np.stack([dcrop] * 3, -1) / 1000]))

        # def normalize(a):
        #     return (a - a.min()) / a.max()

        clip = [dcrop.mean() - 2 * dcrop.std(), dcrop.mean() + 2 * dcrop.std()]

        dcopy = dcrop.copy()
        dcopy[dcopy < clip[0]] = clip[0]
        dcopy[dcopy > clip[1]] = clip[1]
        # dist = 1 - normalize(dcopy)
        # plt.imshow(dist, cmap='rainbow')
        # plt.imshow(dcopy)
        # plt.show()

        return dcopy, landmarks - bbox[:2]

    @staticmethod
    def spoof(dcrop, landmarks, debug=False):
        """
        Наивная проверка на спуфинг — глаза не дальше от камеры, чем нос (с порогом)
        """

        landmarks = landmarks.astype(int)

        eye1 = dcrop[
            landmarks[0][1] - 10 : landmarks[0][1] + 10,
            landmarks[0][0] - 10 : landmarks[0][0] + 10,
        ]
        eye2 = dcrop[
            landmarks[1][1] - 10 : landmarks[1][1] + 10,
            landmarks[1][0] - 10 : landmarks[1][0] + 10,
        ]
        nose = dcrop[
            landmarks[2][1] - 10 : landmarks[2][1] + 10,
            landmarks[2][0] - 10 : landmarks[2][0] + 10,
        ]

        if debug:
            print(eye1.mean(), eye2.mean(), nose.mean(), dcrop.mean())

        return {
            "eye1": int(eye1.mean()),
            "eye2": int(eye2.mean()),
            "nose": int(nose.mean()),
            "decision": not eye1.mean() > nose.mean() + 20
            or not eye2.mean() > nose.mean() + 20,
        }

    def anti3d(self, data, model):
        dcrop, landmarks = self.get_depth_crop(data["image"], data["depth"], model)

        return {
            "faceSize": f"{dcrop.shape[0]}x{dcrop.shape[1]}",
            # "depthMax": dcopy.max().item(),
            # "depthMin": dcopy.min().item(),
            # "depthStd": round(dcopy.std().item()),
            "spoofing": self.spoof(dcrop, landmarks),
        }


def demo():
    """
    Проверка живучести камеры
    """

    realsense = RealSense()
    realsense.start()

    model = ArcFaceModel()

    while True:
        try:
            shot = realsense.get_image_depth_color()
            # noinspection PyTypeChecker
            print(realsense.anti3d(shot, model))
        except (NoFacesDetected, ValueError, RuntimeError):
            print("No faces detected")
        except KeyboardInterrupt:
            return
        except:
            realsense.stop()
            raise


if __name__ == "__main__":
    demo()
