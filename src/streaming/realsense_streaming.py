import pyrealsense2 as rs

from .helpers import DepthFiltering, Visualizer
from ..utils import timing


class RSStreaming(DepthFiltering, Visualizer):
    def __init__(self, temporal_smoothing):
        super().__init__(temporal_smoothing)

        self.__create_pipeline()
        self.__create_configs()
        self.__start_streaming()
        self.__create_alignment_object()

    def __create_pipeline(self):
        self.pipeline = rs.pipeline()

    def __create_configs(self, width=1280, height=720):
        self.config = rs.config()
        # self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        # self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    def __start_streaming(self):
        self.profile = self.pipeline.start(self.config)

    def __create_alignment_object(self):
        # align_to = rs.stream.depth
        align_to = rs.stream.color

        self.align = rs.align(align_to=align_to)

    @timing
    def get_aligned_frameset(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()
        return aligned_depth_frame, color_frame

    @timing
    def get_filtered_frameset(self):
        images = []
        depths = []
        for x in range(self.temporal_smoothing):
            aligned_depth_frame, color_frame = self.get_aligned_frameset()
            images.append(color_frame)
            depths.append(aligned_depth_frame)

        depth = None
        for i in range(self.temporal_smoothing):
            depth = depths[i]
            depth = self.apply_filters(depth)
        return depth, images[-1]
