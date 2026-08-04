# Automated Image Capture

PyQt6-Hardware-Dashboard als erster Baustein einer automatisierten Aufnahme von
YOLO-Trainingsbildern. Die Anwendung verbindet:

- eine Baumer GigE-Vision-Kamera über den installierten GenTL-Producer,
- einen Universal Robots UR16e über einen Statusmonitor und einen begrenzten
  RTDE-Pose-Handshake,
- zwei Neewer RGB660 Pro II über Bluetooth Low Energy.

Die aktuelle Version zeigt das Kamera-Livebild und Gerätestatus an und erlaubt die
unabhängige manuelle Steuerung beider Lichtpanels. Sie kann außerdem eine von sieben fest im
UR-Programm freigegebenen Ansichten anfordern. Automatische Aufnahmesequenzen mit getrennten
Panelhelligkeiten, optionaler Belichtungsvariation und Bild-/YAML-Speicherung sind enthalten.
Automatische Annotation ist noch nicht enthalten.

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
6. Für einen Pose-Auftrag das vorbereitete `BiBaZu`-Programm manuell starten, eine Ansicht
   auswählen und die einmalige Bewegungsfreigabe bestätigen.
7. Die Lichter erst nach erfolgreicher Verbindung über Ein/Aus, Helligkeit, CCT oder HSI ändern.

## Automatische Aufnahme

1. `BiBaZu_GUI.urp` am Teach Pendant laden und manuell starten.
2. Kamera, UR und beide Panels in der GUI verbinden.
3. Unter **Aufnahme konfigurieren** Speicherort, UR-Start-/Endpose sowie Start, Ende und
   Schrittweite beider Panelhelligkeiten festlegen.
4. Optional die Belichtungsvariation aktivieren, wenn die verbundene Kamera eine manuell
   beschreibbare `ExposureTime` meldet.
5. **Aufnahme starten** wählen und die Freigabe des Arbeitsraums bestätigen.

Die Reihenfolge ist Pose → Panel 2 → Panel 1 → Belichtung. Panel 1 läuft damit vollständig
durch, bevor Panel 2 erhöht wird; erst nach sämtlichen Lichtkombinationen wird der UR verfahren.
Jede Sitzung erhält einen eigenen Ordner `capture_YYYYMMDD_HHMMSS`. Zu jeder verlustfreien
PNG-Datei wird eine gleichnamige YAML-Datei mit Kamera-, Roboter-, Licht- und Sequenzdaten
gespeichert. Eine optionale Belichtungsvariation wird beim Abschluss oder Stoppen auf den Wert
vom Kamera-Verbindungsaufbau zurückgesetzt.

**Aufnahme stoppen** verhindert weitere Aufträge und Bilder. Eine bereits begonnene Bewegung
wird aus Sicherheitsgründen nicht durch einen externen Stop-Befehl unterbrochen.

Die beiden BLE-Adressen werden getrennt gespeichert, sodass „Alle verbinden“ nicht beide
Adapter demselben Panel zuordnet. Die Lichtwerte sind als „bestätigter letzter Befehl“
gekennzeichnet. Das BLE-Protokoll liefert
nicht in jeder Firmware verlässliche physische Istwerte zurück. Beim Beenden wird das Panel
nicht automatisch ausgeschaltet oder umgestellt.

## Sicherheit

Der UR-Statusmonitor verwendet `rtde_receive`. Der separate `rtde_io`-Kanal darf ausschließlich
die Input-Integer-Register 42 und 43 für eine freigegebene Pose und deren Befehlsnummer schreiben.
Der eingebaute Dashboard-Client akzeptiert weiterhin nur folgende Abfragen:

- `robotmode`
- `safetymode`
- `programState`
- `get loaded program`
- `is in remote control`
- `PolyscopeVersion`

Es gibt keine direkte Bewegungs-, Power-, Brake-, Dashboard-Play- oder URScript-Upload-Funktion.
Der Roboter bewegt sich ausschließlich durch die zuvor lokal geprüften Bewegungsnodes seines
laufenden PolyScope-Programms. Das Registerprotokoll und die Inbetriebnahme sind unter
[`ur_program/README.md`](ur_program/README.md) beschrieben.

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
