from __future__ import annotations

from dataclasses import dataclass, replace
from ipaddress import ip_address
from pathlib import Path

from PyQt6.QtCore import QSettings

DEFAULT_CTI_PATH = r"C:\Program Files\Baumer Camera Explorer\bgapi2_gige.cti"


@dataclass(slots=True)
class AppSettings:
    camera_ip: str = "169.254.117.70"
    robot_ip: str = "10.10.10.10"
    camera_cti_path: str = DEFAULT_CTI_PATH
    camera_serial: str = ""
    light_address: str = ""
    light_name: str = ""
    light_2_address: str = ""
    light_2_name: str = ""
    preview_max_fps: int = 15
    auto_reconnect: bool = True

    def validated(self) -> AppSettings:
        ip_address(self.camera_ip)
        ip_address(self.robot_ip)
        if not self.camera_cti_path.strip():
            raise ValueError("Der Pfad zum GenTL-Producer darf nicht leer sein.")
        if not 1 <= self.preview_max_fps <= 60:
            raise ValueError("Die Vorschau-Bildrate muss zwischen 1 und 60 liegen.")
        return replace(
            self,
            camera_ip=self.camera_ip.strip(),
            robot_ip=self.robot_ip.strip(),
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
            camera_cti_path=str(self._settings.value("camera/cti_path", defaults.camera_cti_path)),
            camera_serial=str(self._settings.value("camera/serial", "")),
            light_address=str(self._settings.value("light/address", "")),
            light_name=str(self._settings.value("light/name", "")),
            light_2_address=str(self._settings.value("light_2/address", "")),
            light_2_name=str(self._settings.value("light_2/name", "")),
            preview_max_fps=int(
                self._settings.value("camera/preview_max_fps", defaults.preview_max_fps)
            ),
            auto_reconnect=self._settings.value(
                "connections/auto_reconnect", defaults.auto_reconnect, type=bool
            ),
        )

    def save(self, config: AppSettings) -> None:
        config = config.validated()
        self._settings.setValue("camera/ip", config.camera_ip)
        self._settings.setValue("robot/ip", config.robot_ip)
        self._settings.setValue("camera/cti_path", config.camera_cti_path)
        self._settings.setValue("camera/serial", config.camera_serial)
        self._settings.setValue("light/address", config.light_address)
        self._settings.setValue("light/name", config.light_name)
        self._settings.setValue("light_2/address", config.light_2_address)
        self._settings.setValue("light_2/name", config.light_2_name)
        self._settings.setValue("camera/preview_max_fps", config.preview_max_fps)
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
        ).validated()

    def save_acquisition(self, config) -> None:
        config = config.validated()
        for field_name in (
            "output_directory",
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
        ):
            value = getattr(config, field_name)
            if field_name == "output_directory":
                value = str(value)
            self._settings.setValue(f"acquisition/{field_name}", value)
        self._settings.sync()
