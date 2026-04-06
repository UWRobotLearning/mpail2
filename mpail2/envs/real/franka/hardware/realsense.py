"""Adapted from `frankz/perception/realsense.py` in the frankz repository.

Source: https://github.com/memmelma/frankz/tree/mpail2
Author: memmelma

Modifications have been made to fit this project.
"""

import time

import cv2
import numpy as np
import pyrealsense2 as rs


def gather_realsense_cameras(
    rgb=True,
    depth=False,
    ir=False,
    high_res_rgb=False,
    align=None,
    hardware_reset=False,
    serial_numbers=None,
):
    if serial_numbers is None:
        serial_numbers = []

    context = rs.context()
    all_devices = list(context.devices)
    all_rs_cameras = []

    for device in all_devices:
        serial = device.get_info(rs.camera_info.serial_number)
        print(serial)

        if serial_numbers and serial not in serial_numbers:
            continue

        if hardware_reset:
            device.hardware_reset()
            time.sleep(1)

        rs_camera = RealSenseCamera(
            device,
            rgb=rgb,
            depth=depth,
            ir=ir,
            high_res_rgb=high_res_rgb,
            align=align,
        )
        all_rs_cameras.append(rs_camera)

    return all_rs_cameras


class RealSenseCamera:
    def __init__(self, device, rgb=True, depth=False, ir=False, high_res_rgb=False, align=None):
        del high_res_rgb

        self._pipeline = rs.pipeline()
        self._serial_number = str(device.get_info(rs.camera_info.serial_number))
        self._config = rs.config()
        self._config.enable_device(self._serial_number)

        self.ir = ir
        self.depth = depth
        self.rgb = rgb

        if self.rgb or align == "rgb":
            self._config.enable_stream(rs.stream.color, 320, 240, rs.format.bgr8, 30)
        if self.depth or align == "depth":
            self._config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        if self.ir:
            self._config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
            self._config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)

        if self._serial_number == "825312070216":
            self.pipeline = rs.pipeline()
            self._config = rs.config()
            self._config.enable_device(self._serial_number)
            self._config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        self._pipeline.start(self._config)
        if align == "depth":
            self._align = rs.align(rs.stream.depth)
        elif align == "color":
            self._align = rs.align(rs.stream.color)
        else:
            self._align = None

        profile = self._pipeline.get_active_profile()
        self.calibration = {"intrinsics": {}}

        if self.rgb:
            color_stream = profile.get_stream(rs.stream.color)
            color_int = color_stream.as_video_stream_profile().get_intrinsics()
            self.calibration["intrinsics"]["rgb"] = self._process_intrinsics(color_int)
        if self.depth:
            depth_stream = profile.get_stream(rs.stream.depth)
            depth_int = depth_stream.as_video_stream_profile().get_intrinsics()
            self.calibration["intrinsics"]["depth"] = self._process_intrinsics(depth_int)
        if self.ir:
            ir_left_stream = profile.get_stream(rs.stream.infrared, 1)
            ir_left_int = ir_left_stream.as_video_stream_profile().get_intrinsics()
            self.calibration["intrinsics"]["ir_left"] = self._process_intrinsics(ir_left_int)
            ir_right_stream = profile.get_stream(rs.stream.infrared, 2)
            ir_right_int = ir_right_stream.as_video_stream_profile().get_intrinsics()
            self.calibration["intrinsics"]["ir_right"] = self._process_intrinsics(ir_right_int)

            extrinsics = ir_left_stream.get_extrinsics_to(ir_right_stream)
            self.calibration["ir_baseline_left_to_right"] = abs(extrinsics.translation[0])

        self._fovy = 65
        color_sensor = device.query_sensors()[0]
        device_name = device.get_info(rs.camera_info.name)

        if "D455" in device_name or "D435" in device_name or "D415" in device_name:
            exposure = 1000
        elif "D405" in device_name:
            exposure = 150000
        else:
            raise NotImplementedError

        color_sensor.set_option(rs.option.exposure, exposure)

    def _process_intrinsics(self, params):
        intrinsics = {}
        intrinsics["cameraMatrix"] = np.array(
            [[params.fx, 0, params.ppx], [0, params.fy, params.ppy], [0, 0, 1]]
        )
        intrinsics["distCoeffs"] = np.array(list(params.coeffs))
        return intrinsics

    def read_camera(self):
        out = {}
        frames = self._pipeline.wait_for_frames()

        if self.ir:
            ir_left_frame = frames.get_infrared_frame(1)
            ir_right_frame = frames.get_infrared_frame(2)
            out["ir_left"] = cv2.cvtColor(np.asanyarray(ir_left_frame.get_data()), cv2.COLOR_GRAY2RGB)
            out["ir_right"] = cv2.cvtColor(np.asanyarray(ir_right_frame.get_data()), cv2.COLOR_GRAY2RGB)

        if self._align is not None:
            frames = self._align.process(frames)

        if self.rgb:
            color_frame = frames.get_color_frame()
            out["rgb"] = cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_BGR2RGB)

        if self.depth:
            depth_frame = frames.get_depth_frame()
            out["depth"] = np.asanyarray(depth_frame.get_data())

        out["read_time"] = time.time()
        return out

    def disable_camera(self):
        self._pipeline.stop()
        self._config.disable_all_streams()
