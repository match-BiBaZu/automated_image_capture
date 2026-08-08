"""Read-only Baumer/GenTL throughput benchmark and node diagnostics."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from harvesters.core import Harvester

DEFAULT_CTI = Path(r"C:\Program Files\Baumer Camera Explorer\bgapi2_gige.cti")
NODE_NAMES = (
    "Width",
    "Height",
    "PixelFormat",
    "ExposureAuto",
    "ExposureTime",
    "AcquisitionFrameRateEnable",
    "AcquisitionFrameRate",
    "ResultingFrameRate",
    "DeviceLinkThroughputLimit",
    "DeviceLinkCurrentThroughput",
    "GevSCPSPacketSize",
    "GevSCPD",
)


def _attribute(node: Any, name: str) -> Any:
    try:
        return getattr(node, name)
    except Exception:
        return None


def _describe(node_map: Any, name: str) -> str:
    try:
        node = getattr(node_map, name)
    except Exception:
        return "nicht verfügbar"
    parts = [f"value={_attribute(node, 'value')!r}"]
    for attribute in ("min", "max", "inc", "access_mode"):
        value = _attribute(node, attribute)
        if value is not None:
            parts.append(f"{attribute}={value!r}")
    return ", ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cti", type=Path, default=DEFAULT_CTI)
    parser.add_argument("--frames", type=int, default=300)
    args = parser.parse_args()

    harvester = Harvester()
    acquirer = None
    try:
        harvester.add_file(str(args.cti), check_existence=True, check_validity=True)
        harvester.update()
        if not harvester.device_info_list:
            raise RuntimeError("Keine GenTL-Kamera gefunden.")
        info = harvester.device_info_list[0]
        print(f"Kamera: {info}")
        acquirer = harvester.create(0)
        node_map = acquirer.remote_device.node_map
        for name in NODE_NAMES:
            print(f"{name}: {_describe(node_map, name)}")

        acquirer.start()
        byte_count = 0
        started = time.perf_counter()
        for _ in range(args.frames):
            with acquirer.fetch(timeout=2.0) as buffer:
                component = buffer.payload.components[0]
                byte_count += int(component.data.nbytes)
        elapsed = time.perf_counter() - started
        fps = args.frames / elapsed
        throughput = byte_count / elapsed / 1_000_000
        print(
            f"Rohabruf: {args.frames} Frames in {elapsed:.3f} s = {fps:.2f} FPS, "
            f"{throughput:.1f} MB/s"
        )
        return 0
    finally:
        if acquirer is not None:
            try:
                acquirer.stop()
            finally:
                acquirer.destroy()
        harvester.reset()


if __name__ == "__main__":
    raise SystemExit(main())
