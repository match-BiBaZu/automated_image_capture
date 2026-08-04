# Automated Image Capture

PyQt6-Hardware-Dashboard als erster Baustein einer automatisierten Aufnahme von
YOLO-Trainingsbildern. Die Anwendung verbindet:

- eine Baumer GigE-Vision-Kamera über den installierten GenTL-Producer,
- einen Universal Robots UR16e ausschließlich lesend über RTDE und Dashboard Server,
- zwei Neewer RGB660 Pro II über Bluetooth Low Energy.

Die aktuelle Version zeigt das Kamera-Livebild und Gerätestatus an und erlaubt die
unabhängige manuelle Steuerung beider Lichtpanels. Roboterbewegungen, automatische Aufnahmesequenzen,
Bildspeicherung und Annotation sind noch nicht enthalten.

## Voraussetzungen

- Windows 10/11 x64
- [uv](https://docs.astral.sh/uv/)
- Baumer Camera Explorer beziehungsweise der Producer
  `C:\Program Files\Baumer Camera Explorer\bgapi2_gige.cti`
- Kamera-Netzwerk `169.254.0.0/16`, hier Kamera `169.254.117.70`
- UR-Netzwerk `10.10.10.0/24`, hier Roboter `10.10.10.10`
- Bluetooth-LE-Adapter und bis zu zwei eingeschaltete RGB660 Pro II

Der Baumer Camera Explorer muss geschlossen sein, da eine Kamera gewöhnlich nicht von zwei
GenTL-Clients gleichzeitig geöffnet werden kann. Die Neewer-Smartphone-App muss vom Panel
getrennt sein. Eine Windows-Bluetooth-Kopplung ist für den BLE-Zugriff normalerweise nicht nötig.

## Installation und Start

```powershell
uv sync --extra dev
uv run python -m automated_image_capture
```

Alternativ steht nach der Installation das Kommando zur Verfügung:

```powershell
uv run yolo-capture-dashboard
```

Beim ersten Start werden keine Geräte automatisch verbunden und keine Lichtwerte verändert.
IPs, CTI-Pfad und Wiederverbindung lassen sich über **Einstellungen** ändern. Die erfolgreiche
Kamera- und Lichtauswahl wird über `QSettings` im Windows-Benutzerprofil gespeichert.

## Bedienung

1. Camera Explorer und Neewer-App schließen.
2. Kamera, Roboter und beide Lichter einschalten.
3. **Alle verbinden** oder die einzelnen Schaltflächen verwenden.
4. Das Kamerabild sowie Modell, Seriennummer, Pixelformat und Bildrate prüfen.
5. Beim UR die RTDE-/Dashboard-Anzeigen und insbesondere den Safety Mode prüfen.
6. Die Lichter erst nach erfolgreicher Verbindung über Ein/Aus, Helligkeit, CCT oder HSI ändern.

Die beiden BLE-Adressen werden getrennt gespeichert, sodass „Alle verbinden“ nicht beide
Adapter demselben Panel zuordnet. Die Lichtwerte sind als „bestätigter letzter Befehl“
gekennzeichnet. Das BLE-Protokoll liefert
nicht in jeder Firmware verlässliche physische Istwerte zurück. Beim Beenden wird das Panel
nicht automatisch ausgeschaltet oder umgestellt.

## Sicherheit

Die UR-Integration importiert ausschließlich `rtde_receive`. Der eingebaute Dashboard-Client
akzeptiert nur folgende Abfragen:

- `robotmode`
- `safetymode`
- `programState`
- `is in remote control`
- `PolyscopeVersion`

Es gibt keine Bewegungs-, Power-, Brake-, I/O- oder URScript-Funktion in diesem Meilenstein.

## Tests

```powershell
uv run pytest
uv run ruff check .
```

Hardwaretests sind standardmäßig deaktiviert:

```powershell
$env:RUN_HARDWARE_TESTS = "1"
uv run pytest -m hardware tests/hardware
```

Die automatischen Hardwaretests öffnen die Kamera, lesen 100 Bilder und beobachten den UR
30 Sekunden lang. Der Lichttest prüft nur Scan und Verbindung. Sichtbare Lichtbefehle werden
bewusst manuell über die GUI abgenommen, damit ein Testlauf die Beleuchtung nicht überraschend
verändert. Roboterbefehle werden auch in Hardwaretests nicht gesendet.

## Fehlerdiagnose

- **Keine Kamera:** CTI-Pfad, dedizierte Netzwerkkarte und Camera Explorer prüfen.
- **Kamera belegt:** Camera Explorer oder einen anderen GenTL-Client schließen.
- **Nur ein UR-Kanal:** RTDE kann in den Robot Security Settings deaktiviert sein; Dashboard und
  RTDE werden deshalb getrennt als „eingeschränkt“ dargestellt.
- **Kein Licht:** Bluetooth-Symbol am Panel aktivieren, Smartphone-App trennen und näher an den
  PC bringen.

Ein passiver BLE-Scan kann bei Bedarf mit `uv run python scripts/probe_ble.py` ausgeführt werden.
Mit `--connect <Adresse>` liest das Skript zusätzlich nur die angebotenen GATT-Service-UUIDs.

Rotierende Logs liegen unter
`%LOCALAPPDATA%\AutomatedImageCapture\logs\dashboard.log`. Es werden keine Bilder protokolliert.
