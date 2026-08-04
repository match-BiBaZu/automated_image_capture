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
            camera_cti_path=str(
                self._settings.value("camera/cti_path", defaults.camera_cti_path)
            ),
            camera_serial=str(self._settings.value("camera/serial", "")),
            light_address=str(self._settings.value("light/address", "")),
            light_name=str(self._settings.value("light/name", "")),
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
        self._settings.setValue("camera/preview_max_fps", config.preview_max_fps)
        self._settings.setValue("connections/auto_reconnect", config.auto_reconnect)
        self._settings.sync()

