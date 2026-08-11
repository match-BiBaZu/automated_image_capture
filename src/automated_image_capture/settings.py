from __future__ import annotations

import json
from dataclasses import dataclass, replace
from ipaddress import ip_address
from pathlib import Path

from PyQt6.QtCore import QSettings

DEFAULT_CTI_PATH = r"C:\Program Files\Baumer Camera Explorer\bgapi2_gige.cti"


@dataclass(slots=True)
class AppSettings:
    camera_ip: str = "169.254.117.70"
    robot_ip: str = "10.10.10.10"
    plc_ip: str = "192.168.10.23"
    plc_ams_net_id: str = "10.145.4.14.1.1"
    plc_port: int = 851
    conveyor_forward_direction: str = ""
    camera_cti_path: str = DEFAULT_CTI_PATH
    camera_serial: str = ""
    light_address: str = ""
    light_name: str = ""
    light_2_address: str = ""
    light_2_name: str = ""
    preview_max_fps: int = 15
    maximize_camera_frame_rate: bool = True
    auto_reconnect: bool = True

    def validated(self) -> AppSettings:
        ip_address(self.camera_ip)
        ip_address(self.robot_ip)
        ip_address(self.plc_ip)
        net_id_parts = self.plc_ams_net_id.strip().split(".")
        if len(net_id_parts) != 6 or any(
            not part.isdigit() or not 0 <= int(part) <= 255 for part in net_id_parts
        ):
            raise ValueError("Die AMS-Net-ID muss aus sechs Zahlen zwischen 0 und 255 bestehen.")
        if not 1 <= self.plc_port <= 65535:
            raise ValueError("Der TwinCAT-PLC-Port muss zwischen 1 und 65535 liegen.")
        if self.conveyor_forward_direction not in {"", "left", "right"}:
            raise ValueError("Die Förderband-Vorwärtsrichtung ist ungültig.")
        if not self.camera_cti_path.strip():
            raise ValueError("Der Pfad zum GenTL-Producer darf nicht leer sein.")
        if not 1 <= self.preview_max_fps <= 60:
            raise ValueError("Die Vorschau-Bildrate muss zwischen 1 und 60 liegen.")
        return replace(
            self,
            camera_ip=self.camera_ip.strip(),
            robot_ip=self.robot_ip.strip(),
            plc_ip=self.plc_ip.strip(),
            plc_ams_net_id=self.plc_ams_net_id.strip(),
            camera_cti_path=str(Path(self.camera_cti_path.strip())),
        )


class SettingsStore:
    ORGANIZATION = "LeibnizUniversitaetHannover"
    APPLICATION = "AutomatedImageCapture"

    def __init__(self, backend: QSettings | None = None) -> None:
        self._settings = backend or QSettings(self.ORGANIZATION, self.APPLICATION)

    def load(self) -> AppSettings:
        defaults = AppSettings()
        return AppSettings(
            camera_ip=str(self._settings.value("camera/ip", defaults.camera_ip)),
            robot_ip=str(self._settings.value("robot/ip", defaults.robot_ip)),
            plc_ip=str(self._settings.value("plc/ip", defaults.plc_ip)),
            plc_ams_net_id=str(
                self._settings.value("plc/ams_net_id", defaults.plc_ams_net_id)
            ),
            plc_port=int(self._settings.value("plc/port", defaults.plc_port)),
            conveyor_forward_direction=str(
                self._settings.value(
                    "plc/conveyor_forward_direction", defaults.conveyor_forward_direction
                )
            ),
            camera_cti_path=str(self._settings.value("camera/cti_path", defaults.camera_cti_path)),
            camera_serial=str(self._settings.value("camera/serial", "")),
            light_address=str(self._settings.value("light/address", "")),
            light_name=str(self._settings.value("light/name", "")),
            light_2_address=str(self._settings.value("light_2/address", "")),
            light_2_name=str(self._settings.value("light_2/name", "")),
            preview_max_fps=int(
                self._settings.value("camera/preview_max_fps", defaults.preview_max_fps)
            ),
            maximize_camera_frame_rate=self._settings.value(
                "camera/maximize_frame_rate",
                defaults.maximize_camera_frame_rate,
                type=bool,
            ),
            auto_reconnect=self._settings.value(
                "connections/auto_reconnect", defaults.auto_reconnect, type=bool
            ),
        )

    def save(self, config: AppSettings) -> None:
        config = config.validated()
        self._settings.setValue("camera/ip", config.camera_ip)
        self._settings.setValue("robot/ip", config.robot_ip)
        self._settings.setValue("plc/ip", config.plc_ip)
        self._settings.setValue("plc/ams_net_id", config.plc_ams_net_id)
        self._settings.setValue("plc/port", config.plc_port)
        self._settings.setValue(
            "plc/conveyor_forward_direction", config.conveyor_forward_direction
        )
        self._settings.setValue("camera/cti_path", config.camera_cti_path)
        self._settings.setValue("camera/serial", config.camera_serial)
        self._settings.setValue("light/address", config.light_address)
        self._settings.setValue("light/name", config.light_name)
        self._settings.setValue("light_2/address", config.light_2_address)
        self._settings.setValue("light_2/name", config.light_2_name)
        self._settings.setValue("camera/preview_max_fps", config.preview_max_fps)
        self._settings.setValue(
            "camera/maximize_frame_rate", config.maximize_camera_frame_rate
        )
        self._settings.setValue("connections/auto_reconnect", config.auto_reconnect)
        self._settings.sync()

    def load_acquisition(self):
        from automated_image_capture.acquisition import AcquisitionSettings

        defaults = AcquisitionSettings(
            output_directory=Path.home() / "Pictures" / "AutomatedImageCapture"
        )
        return AcquisitionSettings(
            output_directory=Path(
                str(
                    self._settings.value(
                        "acquisition/output_directory",
                        str(defaults.output_directory),
                    )
                )
            ),
            capture_mode=str(
                self._settings.value("acquisition/capture_mode", defaults.capture_mode)
            ),
            pose_start=int(self._settings.value("acquisition/pose_start", defaults.pose_start)),
            pose_end=int(self._settings.value("acquisition/pose_end", defaults.pose_end)),
            light_1_start=int(
                self._settings.value("acquisition/light_1_start", defaults.light_1_start)
            ),
            light_1_end=int(self._settings.value("acquisition/light_1_end", defaults.light_1_end)),
            light_1_step=int(
                self._settings.value("acquisition/light_1_step", defaults.light_1_step)
            ),
            light_2_start=int(
                self._settings.value("acquisition/light_2_start", defaults.light_2_start)
            ),
            light_2_end=int(self._settings.value("acquisition/light_2_end", defaults.light_2_end)),
            light_2_step=int(
                self._settings.value("acquisition/light_2_step", defaults.light_2_step)
            ),
            exposure_enabled=self._settings.value(
                "acquisition/exposure_enabled", defaults.exposure_enabled, type=bool
            ),
            exposure_start_us=int(
                self._settings.value("acquisition/exposure_start_us", defaults.exposure_start_us)
            ),
            exposure_end_us=int(
                self._settings.value("acquisition/exposure_end_us", defaults.exposure_end_us)
            ),
            exposure_step_us=int(
                self._settings.value("acquisition/exposure_step_us", defaults.exposure_step_us)
            ),
            light_settle_ms=int(
                self._settings.value("acquisition/light_settle_ms", defaults.light_settle_ms)
            ),
            robot_settle_ms=int(
                self._settings.value("acquisition/robot_settle_ms", defaults.robot_settle_ms)
            ),
            camera_settle_ms=int(
                self._settings.value("acquisition/camera_settle_ms", defaults.camera_settle_ms)
            ),
            ramp_duration_s=float(
                self._settings.value("acquisition/ramp_duration_s", defaults.ramp_duration_s)
            ),
            ramp_image_rate_fps=int(
                self._settings.value(
                    "acquisition/ramp_image_rate_fps", defaults.ramp_image_rate_fps
                )
            ),
            ramp_light_1_period_s=float(
                self._settings.value(
                    "acquisition/ramp_light_1_period_s", defaults.ramp_light_1_period_s
                )
            ),
            ramp_light_2_period_s=float(
                self._settings.value(
                    "acquisition/ramp_light_2_period_s", defaults.ramp_light_2_period_s
                )
            ),
            robot_control_mode=str(
                self._settings.value(
                    "acquisition/robot_control_mode", defaults.robot_control_mode
                )
            ),
            angle_start_deg=float(
                self._settings.value("acquisition/angle_start_deg", defaults.angle_start_deg)
            ),
            angle_end_deg=float(
                self._settings.value("acquisition/angle_end_deg", defaults.angle_end_deg)
            ),
            angle_step_deg=float(
                self._settings.value("acquisition/angle_step_deg", defaults.angle_step_deg)
            ),
            conveyor_enabled=self._settings.value(
                "acquisition/conveyor_enabled", defaults.conveyor_enabled, type=bool
            ),
            conveyor_motion_mode=str(
                self._settings.value(
                    "acquisition/conveyor_motion_mode", defaults.conveyor_motion_mode
                )
            ),
            conveyor_max_offset_mm=float(
                self._settings.value(
                    "acquisition/conveyor_max_offset_mm", defaults.conveyor_max_offset_mm
                )
            ),
            conveyor_step_mm=float(
                self._settings.value(
                    "acquisition/conveyor_step_mm", defaults.conveyor_step_mm
                )
            ),
            conveyor_speed_mm_per_s=float(
                self._settings.value(
                    "acquisition/conveyor_speed_mm_per_s", defaults.conveyor_speed_mm_per_s
                )
            ),
            conveyor_settle_ms=int(
                self._settings.value(
                    "acquisition/conveyor_settle_ms", defaults.conveyor_settle_ms
                )
            ),
        ).validated()

    def save_acquisition(self, config) -> None:
        config = config.validated()
        for field_name in (
            "output_directory",
            "capture_mode",
            "pose_start",
            "pose_end",
            "light_1_start",
            "light_1_end",
            "light_1_step",
            "light_2_start",
            "light_2_end",
            "light_2_step",
            "exposure_enabled",
            "exposure_start_us",
            "exposure_end_us",
            "exposure_step_us",
            "light_settle_ms",
            "robot_settle_ms",
            "camera_settle_ms",
            "ramp_duration_s",
            "ramp_image_rate_fps",
            "ramp_light_1_period_s",
            "ramp_light_2_period_s",
            "robot_control_mode",
            "angle_start_deg",
            "angle_end_deg",
            "angle_step_deg",
            "conveyor_enabled",
            "conveyor_motion_mode",
            "conveyor_max_offset_mm",
            "conveyor_step_mm",
            "conveyor_speed_mm_per_s",
            "conveyor_settle_ms",
        ):
            value = getattr(config, field_name)
            if field_name == "output_directory":
                value = str(value)
            self._settings.setValue(f"acquisition/{field_name}", value)
        self._settings.sync()

    def load_labeling(self):
        from automated_image_capture.labeling import LabelingConfig, LabelSource

        pictures = Path.home() / "Pictures"

        def latest_capture(root: Path) -> Path:
            captures = sorted(
                root.glob("capture_*"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            return captures[0] if captures else root

        default_sources = (
            LabelSource("Pose 1", latest_capture(pictures / "Kk1")),
            LabelSource("Pose 2", latest_capture(pictures / "kk1_flipped")),
            LabelSource(
                "Leere Rutsche",
                latest_capture(pictures / "empty_slide"),
                is_empty=True,
            ),
        )
        sources: tuple[LabelSource, ...] = default_sources
        serialized_sources = str(self._settings.value("labeling/sources_json", "")).strip()
        if serialized_sources:
            try:
                entries = json.loads(serialized_sources)
                parsed = tuple(
                    LabelSource(
                        str(entry["name"]),
                        Path(str(entry["directory"])),
                        bool(entry.get("is_empty", False)),
                    )
                    for entry in entries
                )
                if parsed:
                    sources = parsed
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                sources = default_sources
        elif self._settings.contains("labeling/foreground_directory"):
            # One-time migration from the original two-directory labeling dialog.
            sources = (
                LabelSource(
                    "Pose 1",
                    Path(str(self._settings.value("labeling/foreground_directory"))),
                ),
                default_sources[1],
                LabelSource(
                    "Leere Rutsche",
                    Path(
                        str(
                            self._settings.value(
                                "labeling/background_directory",
                                str(default_sources[2].directory),
                            )
                        )
                    ),
                    is_empty=True,
                ),
            )
        return LabelingConfig(
            sources=sources,
            output_directory=Path(
                str(
                    self._settings.value(
                        "labeling/output_directory",
                        str(pictures / "multi_pose_yolo_obb"),
                    )
                )
            ),
            validation_fraction=float(
                self._settings.value(
                    "labeling/validation_fraction",
                    0.2,
                )
            ),
            minimum_difference=int(
                self._settings.value(
                    "labeling/minimum_difference",
                    80,
                )
            ),
            consensus_fraction=float(
                self._settings.value(
                    "labeling/consensus_fraction",
                    0.55,
                )
            ),
            include_background_negatives=self._settings.value(
                "labeling/include_background_negatives",
                True,
                type=bool,
            ),
            prefer_hardlinks=self._settings.value(
                "labeling/prefer_hardlinks",
                True,
                type=bool,
            ),
        )

    def save_labeling(self, config) -> None:
        self._settings.setValue(
            "labeling/sources_json",
            json.dumps(
                [
                    {
                        "name": source.name,
                        "directory": str(source.directory),
                        "is_empty": source.is_empty,
                    }
                    for source in config.sources
                ],
                ensure_ascii=False,
            ),
        )
        for field_name in (
            "output_directory",
            "validation_fraction",
            "minimum_difference",
            "consensus_fraction",
            "include_background_negatives",
            "prefer_hardlinks",
        ):
            value = getattr(config, field_name)
            if isinstance(value, Path):
                value = str(value)
            self._settings.setValue(f"labeling/{field_name}", value)
        self._settings.sync()

    def load_training_paths(self, defaults) -> dict[str, Path | None]:
        dataset_value = str(self._settings.value("training/dataset_directory", "")).strip()
        source_value = str(self._settings.value("training/source_dataset", "")).strip()
        return {
            "source_dataset": Path(source_value) if source_value else defaults.source_dataset,
            "output_root": Path(
                str(self._settings.value("training/output_root", str(defaults.output_root)))
            ),
            "dataset_directory": Path(dataset_value) if dataset_value else None,
        }

    def save_training_paths(
        self,
        source_dataset: Path,
        output_root: Path,
        dataset_directory: Path | None,
    ) -> None:
        self._settings.setValue("training/source_dataset", str(source_dataset))
        self._settings.setValue("training/output_root", str(output_root))
        self._settings.setValue(
            "training/dataset_directory",
            "" if dataset_directory is None else str(dataset_directory),
        )
        self._settings.sync()

    def load_live_inference(self):
        from automated_image_capture.inference import LiveInferenceConfig, find_latest_model

        saved_model = str(self._settings.value("inference/model_path", "")).strip()
        model_path = Path(saved_model) if saved_model else find_latest_model()
        return LiveInferenceConfig(
            model_path=model_path,
            confidence=float(self._settings.value("inference/confidence", 0.25)),
            image_size=int(self._settings.value("inference/image_size", 640)),
            max_fps=float(self._settings.value("inference/max_fps", 5.0)),
            device=str(self._settings.value("inference/device", "0")),
        ).validated(require_model=False)

    def save_live_inference(self, config) -> None:
        config = config.validated(require_model=False)
        self._settings.setValue("inference/model_path", str(config.model_path))
        self._settings.setValue("inference/confidence", config.confidence)
        self._settings.setValue("inference/image_size", config.image_size)
        self._settings.setValue("inference/max_fps", config.max_fps)
        self._settings.setValue("inference/device", config.device)
        self._settings.sync()

    def load_cleanup(self):
        from automated_image_capture.cleanup import CleanupSettings

        return CleanupSettings(
            root_directory=Path(
                str(
                    self._settings.value(
                        "cleanup/root_directory",
                        str(Path.home() / "Pictures"),
                    )
                )
            ),
            output_format=str(self._settings.value("cleanup/output_format", "png")),
            max_edge=int(self._settings.value("cleanup/max_edge", 0)),
            png_compression=int(self._settings.value("cleanup/png_compression", 3)),
            quality=int(self._settings.value("cleanup/quality", 90)),
            remove_caches=self._settings.value(
                "cleanup/remove_caches", True, type=bool
            ),
            deduplicate=self._settings.value(
                "cleanup/deduplicate", True, type=bool
            ),
        )

    def save_cleanup(self, config) -> None:
        for field_name in (
            "root_directory",
            "output_format",
            "max_edge",
            "png_compression",
            "quality",
            "remove_caches",
            "deduplicate",
        ):
            value = getattr(config, field_name)
            if isinstance(value, Path):
                value = str(value)
            self._settings.setValue(f"cleanup/{field_name}", value)
        self._settings.sync()
