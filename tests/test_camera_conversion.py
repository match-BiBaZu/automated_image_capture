from __future__ import annotations

import numpy as np
import pytest

from automated_image_capture.hardware.camera import (
    MAX_CONSECUTIVE_FETCH_TIMEOUTS,
    camera_error_message,
    convert_to_rgb,
    is_camera_fetch_timeout,
    should_retry_camera_fetch,
)


def test_mono8_to_rgb() -> None:
    source = np.array([[0, 64], [128, 255]], dtype=np.uint8)
    result = convert_to_rgb(source.ravel(), 2, 2, "Mono8")

    assert result.shape == (2, 2, 3)
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result[..., 0], source)
    np.testing.assert_array_equal(result[..., 1], source)


def test_mono12_scales_for_preview() -> None:
    source = np.array([0, 2048, 4095, 1024], dtype=np.uint16)
    result = convert_to_rgb(source, 2, 2, "Mono12")

    assert result[0, 0, 0] == 0
    assert 126 <= result[0, 1, 0] <= 128
    assert result[1, 0, 0] == 255


@pytest.mark.parametrize("pixel_format", ["BayerRG8", "BayerBG8", "BayerGR8", "BayerGB8"])
def test_bayer_variants_produce_rgb(pixel_format: str) -> None:
    source = np.arange(64, dtype=np.uint8).reshape(8, 8)
    result = convert_to_rgb(source, 8, 8, pixel_format)

    assert result.shape == (8, 8, 3)
    assert result.flags.c_contiguous


def test_rgb_and_bgr_channel_order() -> None:
    rgb_source = np.array([[[10, 20, 30]]], dtype=np.uint8)
    bgr_source = np.array([[[30, 20, 10]]], dtype=np.uint8)

    np.testing.assert_array_equal(convert_to_rgb(rgb_source, 1, 1, "RGB8"), rgb_source)
    np.testing.assert_array_equal(convert_to_rgb(bgr_source, 1, 1, "BGR8"), rgb_source)


def test_packed_format_has_clear_error() -> None:
    with pytest.raises(ValueError, match="Gepacktes Pixelformat"):
        convert_to_rgb(np.zeros(3, dtype=np.uint8), 2, 2, "Mono12Packed")


def test_exclusive_access_error_mentions_camera_explorer() -> None:
    message = camera_error_message(RuntimeError("Open device with exclusive access failed"))

    assert "Camera Explorer" in message


def test_empty_gentl_timeout_exception_is_recognized_by_type() -> None:
    class TimeoutException(Exception):
        pass

    error = TimeoutException()

    assert str(error) == ""
    assert is_camera_fetch_timeout(error)
    assert should_retry_camera_fetch(error, 1)
    assert should_retry_camera_fetch(error, MAX_CONSECUTIVE_FETCH_TIMEOUTS - 1)
    assert not should_retry_camera_fetch(error, MAX_CONSECUTIVE_FETCH_TIMEOUTS)


def test_non_timeout_camera_error_is_not_retried() -> None:
    error = RuntimeError("Buffer payload is invalid")

    assert not is_camera_fetch_timeout(error)
    assert not should_retry_camera_fetch(error, 1)
