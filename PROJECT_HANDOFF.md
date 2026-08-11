# Projektübergabe: Automated Image Capture

Stand: 11. August 2026

Repository: `https://github.com/match-BiBaZu/automated_image_capture`

Referenzstand bei Aktualisierung: Branch `main`, Commit `2d29191`, zuzüglich der in diesem
Dokument beschriebenen lokalen, noch nicht vollständig eingecheckten Änderungen und Artefakte.

## 1. Zielbild des Projekts

Das Projekt automatisiert die Erzeugung, Aufbereitung und Prüfung von Trainingsdaten für
YOLO-OBB-Modelle in einer realen Versuchszelle. Eine gemeinsame PyQt6-Anwendung koordiniert:

- eine Baumer-Industriekamera,
- einen Universal Robots UR16e,
- zwei Neewer RGB660 Pro II,
- ein über Beckhoff/TwinCAT ADS gesteuertes Förderband,
- automatische OBB-Erzeugung,
- Review und Kuratierung des Datensatzes,
- YOLO26n-OBB-Training und Auswertung,
- Live-Inferenz im Kamerabild.

Das langfristige Ziel ist ein möglichst automatischer und reproduzierbarer Datenzyklus:

1. Bauteil in verschiedenen Orientierungen, Positionen, Beleuchtungen und optionalen
   Kamera-Belichtungen aufnehmen.
2. Zu jeder Aufnahme alle relevanten Soll- und Messwerte speichern.
3. OBBs aus einer passenden Leerbildserie erzeugen und unsichere Ergebnisse sichtbar machen.
4. Nur tatsächlich falsche Bilder im Review einzeln ausschließen.
5. Einen strikt getrennten Train-/Validation-/Test-Datensatz erzeugen.
6. Ein YOLO-OBB-Modell trainieren und ehrlich auf Validation, Test und Leerbildern auswerten.
7. Das Modell anschließend live auf dem Baumer-Kamerabild prüfen.

Die Anwendung ist kein Sicherheitscontroller. Roboter- und Bandbewegungen setzen weiterhin
die vorhandene Sicherheitssteuerung, einen geprüften Arbeitsraum und eine bewusste Freigabe
durch den Bediener voraus.

## 2. Aktuell verwendete Hardware und Adressen

| Komponente | Aktuelle Vorgabe | Schnittstelle |
| --- | --- | --- |
| Baumer-Kamera | `169.254.117.70` | GigE Vision / GenTL / GenICam |
| Baumer GenTL Producer | `C:\Program Files\Baumer Camera Explorer\bgapi2_gige.cti` | Harvester |
| UR16e | `10.10.10.10` | RTDE und Dashboard-Port 29999 |
| Beckhoff-SPS | `192.168.10.23` | ADS/TCP 48898 |
| SPS AMS-Net-ID | `10.145.4.14.1.1` | TwinCAT ADS |
| SPS PLC-Port | `851` | TwinCAT-3-Runtime 1 |
| Licht 1 | Benutzerangabe `D296EBA9A286` bzw. `D2:96:EB:A9:A2:86` | BLE |
| Licht 2 | identisches RGB660 Pro II; Adresse in QSettings gespeichert | BLE |
| Bluetooth-Adapter | BT540, Treiber wurde installiert | Windows BLE |

Auf einem neuen PC müssen die Netzwerkschnittstellen die drei getrennten Netze erreichen.
Die Anwendung verändert weder IP-Konfigurationen noch AMS-Routen selbst.

## 3. Technologie und reproduzierbare Umgebung

- Windows 10/11 x64
- Python `>=3.12,<3.13`
- Paketverwaltung mit `uv`
- PyQt6 und `qasync`
- `harvesters`/GenTL für die Kamera
- `ur-rtde==1.6.3` für UR-Status und den begrenzten Register-Handshake
- `neewerlite==0.3.3` und Bleak für die Panels
- `pyads==3.6.0` für das Förderband
- OpenCV und NumPy für Bildverarbeitung
- Ultralytics, PyTorch 2.7.1 und Torchvision 0.22.1
- NVIDIA RTX 2000 Ada Generation Embedded GPU mit 8 GB VRAM
- PyTorch `2.7.1+cu118`, Torchvision 0.22.1, CUDA 11.8 und Ultralytics 8.4.102

`pyproject.toml` und `uv.lock` sind versioniert. Installation und Start:

```powershell
git clone https://github.com/match-BiBaZu/automated_image_capture
cd automated_image_capture
uv sync --extra dev
uv run python -m automated_image_capture
```

Alternativ:

```powershell
uv run yolo-capture-dashboard
```

## 4. Architektur und wichtige Dateien

Der Python-Code liegt unter `src/automated_image_capture`.

| Datei/Ordner | Verantwortung |
| --- | --- |
| `app.py`, `__main__.py` | Anwendung und Qt/qasync-Eventloop starten |
| `models.py` | gemeinsame Status- und Datenmodelle |
| `settings.py` | persistente QSettings und Standardwerte |
| `hardware/camera.py` | Kameraerkennung, Stream, Konvertierung, Exposure |
| `hardware/robot.py` | RTDE-Monitor, read-only Dashboard, Register-Handshake |
| `hardware/light.py` | BLE-Scan, Verbindung, Kommandos und Wiederverbindung |
| `hardware/conveyor.py` | ADS-Status, sichere serielle Fahrbefehle und Nullpunkt |
| `acquisition.py` | Aufnahmeplan, Zustandsmaschine, Metadaten und Fortsetzen |
| `labeling.py` | Bildpaarung, Segmentierung, Förderbandbahn und OBB-Ausgabe |
| `dataset.py` | Review-Datensätze, Curation, Splits, Manifest und Prüfung |
| `training.py` | Training, Validation, Test und Leerbild-Fehlalarme |
| `inference.py` | Hintergrund-Inferenz und OBB-Overlay im Livebild |
| `ui/main_window.py` | Hauptfenster und Hardware-Dashboard |
| `ui/widgets.py` | Gerätekarten und Aufnahmekonfiguration |
| `ui/labeling_dialog.py` | automatische OBB-Erzeugung |
| `ui/training_dialog.py` | Review, Kuratierung, Training und Resultate |
| `scripts/benchmark_camera.py` | isolierter Kamera-Rohabruf- und Durchsatzbenchmark |
| `scripts/generate_obb_cli.py` | reproduzierbare OBB-Erzeugung ohne GUI |
| `ur_program/` | kontinuierliches UR-Programm und Bau-/Installationshinweise |
| `tests/` | Unit-, GUI- und optionale Hardwaretests |

Hardwareadapter sind von den Widgets getrennt. Kamera, UR und Förderband besitzen eigene
Worker-Threads. BLE läuft im mit Qt integrierten Asyncio-Eventloop. Widgets werden nur im
GUI-Thread geändert.

Alle Adapter verwenden dieselben logischen Zustände:

- `DISCONNECTED`
- `DISCOVERING`
- `CONNECTING`
- `CONNECTED`
- `DEGRADED`
- `ERROR`

Ein Gerätefehler darf die Statusanzeige und Bedienung der anderen Geräte nicht zerstören.

## 5. Persistente Einstellungen

Die Anwendung verwendet:

- Organisation: `LeibnizUniversitaetHannover`
- Anwendung: `AutomatedImageCapture`

Unter Windows liegen diese QSettings normalerweise in der Registry unter:

```text
HKEY_CURRENT_USER\Software\LeibnizUniversitaetHannover\AutomatedImageCapture
```

Gespeichert werden unter anderem Kamera-IP/Seriennummer, CTI-Pfad, UR-IP, SPS-Zugang,
Vorwärtsrichtung, beide BLE-Geräte, Aufnahmeparameter, Labelquellen, Trainingspfade und die
Live-Inferenzkonfiguration. Diese Werte werden nicht durch Git übertragen.

Optionaler Export vor dem PC-Wechsel:

```powershell
reg export "HKCU\Software\LeibnizUniversitaetHannover\AutomatedImageCapture" `
  AutomatedImageCapture-settings.reg
```

Die Registry-Datei kann geräteabhängige Pfade und Adressen enthalten. Sie sollte auf dem neuen
PC geprüft und nicht ungeprüft in Git eingecheckt werden.

## 6. Hauptfenster und Hardware-Dashboard

Das Hauptfenster bietet:

- „Alle verbinden/trennen“,
- einen Einstellungsdialog,
- ein rotierendes Ereignisprotokoll,
- Gerätekarten für Kamera, UR, Förderband und beide Panels,
- Kamera-Livebild,
- optionale YOLO-Live-Inferenz,
- Aufnahmekonfiguration und Start/Stop/Fortsetzen,
- „OBB-Labels …“,
- „YOLO-Training …“.

„Alle verbinden“ darf keine Bewegung und keine automatische Änderung von Licht- oder
Kamerawerten auslösen. Beim Beenden werden Worker, Netzwerkverbindungen und BLE-Tasks mit
Timeouts geschlossen.

Rotierende Logs liegen unter:

```text
%LOCALAPPDATA%\AutomatedImageCapture\logs\dashboard.log
```

Bilddaten werden nicht protokolliert.

## 7. Kamera

### Anforderungen und Verhalten

- Entdeckung über den Baumer-GenTL-Producer.
- Auswahl nach gespeicherter Seriennummer beziehungsweise IP.
- Stream in einem separaten Thread.
- Ein GenTL-Buffer wird vor der Freigabe vollständig in ein NumPy-Bild kopiert.
- Der Worker leert den GenTL-Stream mit der tatsächlich erreichbaren Rohbildrate.
- Rohabruf, Aufnahme und GUI-Vorschau besitzen getrennte Pfade. Die Vorschau erhält nur das
  neueste Bild und ist separat begrenzt; dieses Limit bremst die Hintergrundaufnahme nicht.
- Das kameraseitige `AcquisitionFrameRate`-Limit kann in den Einstellungen deaktiviert werden
  und ist standardmäßig deaktiviert.
- Mono-Aufnahmen bleiben auf dem Aufnahme- und Schreibpfad einkanalig und werden nicht
  unnötig zu RGB verdreifacht.
- Unterstützt werden Mono-, hochbitige Mono-, Bayer-, RGB- und BGR-Formate.
- Hochbitige Bilder werden nur für die Vorschau skaliert; gespeicherte Daten bleiben
  verlustfrei.
- Status: Modell, Seriennummer, IP, Auflösung, Pixelformat, Kamera-/Rohabruf-/Vorschau-FPS und
  Belichtungszeit.
- Exposure kann in der Kamerakarte manuell gesetzt werden. Dazu darf `ExposureAuto` für die
  aktuelle Verbindung deaktiviert werden. Beim Trennen werden die beim Verbinden gelesenen
  Werte wiederhergestellt.

Der Baumer Camera Explorer muss geschlossen sein. Andernfalls ist die Kamera meist exklusiv
belegt.

Die Belichtung setzt eine harte physikalische Obergrenze: 250000 µs entsprechen höchstens
ungefähr 4 FPS, 5000 µs belichtungsseitig höchstens 200 FPS. Netzwerk, Kamera und Host können
die praktische Rate weiter begrenzen. Für eine Messung ohne GUI:

```powershell
uv run python scripts/benchmark_camera.py --frames 300
```

Die schnelle Rampenaufnahme lässt sich bis 240 Bilder/s konfigurieren, startet aber nur, wenn
der Preflight den erforderlichen Rohabruf tatsächlich misst. PNGs werden verlustfrei und bei
Hochgeschwindigkeitsserien unkomprimiert durch mehrere Hintergrund-Writer gespeichert.

### Robustheit

Fetch-Timeouts werden abhängig von der Exposure bewertet. Kurze Aussetzer führen zunächst zu
`DEGRADED`; der Stream wird automatisch neu gestartet. Erst nach wiederholtem Scheitern wird
eine Aufnahmeserie fortsetzbar unterbrochen. Die synchronisierte Aufnahme toleriert derzeit
eine Frame-Verspätung von 1,5 Sekunden und verwendet frische Frames statt einer wachsenden
Warteschlange.

## 8. UR16e

### Monitoring

Der UR wird bei 10 Hz über `rtde_receive` beobachtet. Angezeigt werden Robot/Safety Mode,
Speed Scaling, Gelenkpositionen und TCP-Pose. Der Dashboard-Server ergänzt Program State,
geladenes Programm, Local/Remote und PolyScope-Version.

Erlaubte Dashboard-Befehle sind ausschließlich:

- `robotmode`
- `safetymode`
- `programState`
- `get loaded program`
- `is in remote control`
- `PolyscopeVersion`

RTDE und Dashboard werden getrennt bewertet. Fällt nur einer aus, ist der UR `DEGRADED` und
nicht fälschlich vollständig getrennt.

### Sicherheitsgrenze

Die Anwendung verwendet kein RTDE-Control-Interface, kein frei erzeugtes URScript, keinen
Dashboard-`play`, kein Power-on und kein Brake-release. Bewegungen werden ausschließlich in
einem lokal geladenen und geprüften PolyScope-Programm ausgeführt.

### Feste Pose-IDs

Freigegebene Werte im aktuellen Code:

```text
155, 160, 170, 180, 190, 200, 210,
1155, 1170, 1185, 1200,
2155, 2170, 2185, 2200
```

Diese Werte sind Bezeichner und nicht zwingend Winkel in Grad. Das ursprüngliche
`BiBaZu_GUI.urp` enthält die geprüften Switch-Cases/Waypoints.

### Kontinuierlicher Winkel

`ur_program/BiBaZu_Continuous.urp` hält die am Teach Pendant eingelernte XYZ-Position und
ändert nur die definierte Orientierung. Zulässiger Pitchbereich: 15,5 bis 21,0 Grad,
Kodierung in Zehntelgrad (`155` bis `210`). Standard-Schrittweite: 0,5 Grad.

Registervertrag:

| Register | Richtung | Bedeutung |
| --- | --- | --- |
| Input Integer 42 | GUI → UR | Pose-ID oder Winkel in Zehntelgrad |
| Input Integer 43 | GUI → UR | eindeutige Befehlssequenz/Commit |
| Output Integer 41 | UR → GUI | zuletzt erreichter Wert |
| Output Integer 42 | UR → GUI | quittierte Sequenz |
| Output Integer 43 | UR → GUI | `1` bereit, `2` fährt, `3` erreicht, `-1` abgelehnt |

Der Wert wird zuerst geschrieben, die Sequenz danach. Alte Registerinhalte dürfen beim
Programmstart keine Bewegung auslösen. Der RTDE-Worker ist der einzige Besitzer der Register
42/43; kein zweiter RTDE-Client darf gleichzeitig schreiben.

Inbetriebnahme und Dateien sind zusätzlich in `ur_program/README.md` beschrieben. Das URP
wird von der Desktopanwendung weder hochgeladen noch gestartet.

## 9. Neewer RGB660 Pro II

Es werden zwei identische Panels gleichzeitig unterstützt. Jeder Adapter besitzt eine eigene
gespeicherte Adresse; bereits vom anderen Adapter verwendete Adressen werden bei der Auswahl
ausgeschlossen.

Verhalten:

- fünf Sekunden BLE-Scan,
- Auswahl nach gespeicherter Adresse, Neewer/RGB660-Name und RSSI,
- Power, Helligkeit 0–100 %, CCT und HSI,
- Slider-Entprellung,
- höchstens ein BLE-Befehl gleichzeitig pro Panel,
- veraltete Rampenzwischenwerte werden übersprungen,
- Befehls-Timeout derzeit drei Sekunden,
- begrenzte automatische Wiederverbindung,
- keine Änderung der Lichtwerte beim Verbinden oder Beenden.

Die angezeigten Werte sind der „letzte bestätigte Befehl“, nicht zwingend ein physisch
zurückgelesener Istwert. Die Smartphone-App muss getrennt sein. Unter Windows ist eine normale
Bluetooth-Geräteliste nicht zuverlässig für BLE-Diagnosen; entscheidend ist der Bleak-Scan.

Diagnose:

```powershell
uv run python scripts/probe_ble.py
uv run python scripts/probe_ble.py --connect <BLE-ADRESSE>
```

## 10. Förderband und TwinCAT ADS

### Voraussetzungen

- TwinCAT ADS Runtime/Router auf dem PC,
- lokale Netzwerkkarte im Netz der SPS,
- beidseitig passende AMS-Route,
- laufendes SPS-Programm mit den erwarteten `MAIN.*`-Symbolen,
- gültige SPS-Kalibrierung,
- keine gleichzeitige Steuerung durch `PressureControlGUI` oder `CSVSaver`.

Die ursprüngliche Förderbandlogik wurde aus
`C:\Users\Administrator\Documents\Dashas_ws\BiBaZu_Big_Boi\CSVSaver` beziehungsweise dessen
PressureControlGUI
abgeleitet. Dieser externe Ordner und das vollständige SPS-Projekt sind nicht Teil dieses
Repositories.

### Verwendete SPS-Symbole

Fahrbefehle:

- `MAIN.GuiCalibrationJogSteps`
- `MAIN.GuiCalibrationJogSpeedFullStepsPerSec`
- `MAIN.GuiCalibrationMoveLeft`
- `MAIN.GuiCalibrationMoveRight`
- `MAIN.GuiCalibrationStop`
- `MAIN.GuiConveyorCalibrationMode`
- `MAIN.GuiConveyorEnabled`

Status/Kalibrierung:

- `MAIN.CalibrationBusy`
- `MAIN.CalibrationError`
- `MAIN.CalibrationStatusCode`
- `MAIN.StepperPosReadyToExecute`
- `MAIN.StepperPosWarning`
- `MAIN.StepperInternalPosition`
- `MAIN.StepperControlEnable`
- `MAIN.StepperWcState`
- `MAIN.StepperInfoDataState`
- `MAIN.GuiConveyorMmPerFullStep`
- `MAIN.GuiConveyorCalibrationValid`
- `MAIN.GuiConveyorReverse`

In TwinCAT muss insbesondere `Term 19 → POS Status → Actual position` mit
`MAIN.StepperInternalPosition` verknüpft sein. Ohne echte Positionsrückmeldung darf die
synchronisierte Serie nicht starten.

### Bedien- und Sicherheitslogik

- Beim Verbinden erfolgen keine Schreibzugriffe und keine Bewegung.
- Vor der ersten Serie muss die Positionsrückmeldung durch eine manuelle Testfahrt geprüft
  werden. Der Fahrweg ist frei in Millimetern wählbar; Links/Rechts startet unmittelbar ohne
  zusätzliche Bestätigungsfrage.
- Der Bediener bestätigt einmal, ob Links oder Rechts „vorwärts“ ist.
- Das Bauteil wird hinten platziert und „Aktuelle Position = 0 mm“ gewählt.
- Die vorhandene SPS-Kalibrierung wird nur gelesen.
- Bei Stop, Fehler oder Beenden werden Stop, Kalibriermodus aus und Förderband aus geschrieben.
- Relative Ziele werden immer vom logischen Nullpunkt berechnet; Rundungsfehler summieren sich
  nicht über einzelne Stationen.
- `UDINT`-Positionsüberläufe werden wrap-sicher behandelt.

Eine Fahrt gilt erst als fertig, wenn sowohl die SPS-Fertigmeldung als auch
`StepperInternalPosition` zum berechneten Endpunkt passen. Die Toleranz beträgt drei
Vollschritte. Eine bis zu zwei Sekunden verzögerte Positionsrückmeldung wird bei 10 Hz
nachgeführt, ohne die Toleranz künstlich zu vergrößern.

## 11. Aufnahmemodi und Reihenfolge

### Gemeinsame Variationen

- feste Pose-IDs oder kontinuierlicher UR-Winkel,
- optional Förderbandposition,
- optional mehrere Exposure-Zeiten,
- zwei getrennte Panelhelligkeiten,
- exaktes Raster oder schnelle Rampe.

### Diskrete Förderbandstationen

Reihenfolge:

```text
UR-Ziel → Förderbandstation → Exposure → vollständige Lichtvariation
```

Standard-Bandweg:

```text
0 → 10 → 20 → 30 → 40 → 50 → 40 → 30 → 20 → 10 → 0 mm
```

Hin- und Rückweg sind getrennte Stationen und werden getrennt gespeichert. Der UR wird erst
weitergedreht, wenn das Band wieder bei 0 mm steht.

### Synchronisierte kontinuierliche Förderbandfahrt

Reihenfolge:

```text
UR-Ziel → Exposure → komplette Fahrt 0 → Maximum → 0
                        gleichzeitig Lichtvariation und Kameraaufnahme
```

Im Rampenmodus werden die Lichtperioden auf die Fahrtdauer skaliert. Im diskreten Lichtmodus
werden alle Paare zeitlich über die Rundfahrt verteilt; falls erforderlich wird das Band
automatisch langsamer, damit BLE-Kommandos nicht gestaut werden.

### Exaktes Lichtraster

Panel 1 läuft für jeden Panel-2-Wert vollständig durch, zum Beispiel:

```text
0/0, 10/0, 20/0, …, 100/0, 0/10, 10/10, …
```

### Schnelle Lichtrampe

Standardprofil ohne Band:

- 10 Sekunden je UR-Ziel/Exposure,
- 6 Bilder/s,
- 60 Bilder,
- Panel 1: Dreiecksperiode 2,4 s,
- Panel 2: Dreiecksperiode 10 s,
- nominelle Helligkeit 0 → 100 → 0.

Die Panelwerte sind reproduzierbare Sollwerte beziehungsweise bestätigte Befehle, keine
garantierten optischen Istwerte.

## 12. Aufnahme-Robustheit und Fortsetzen

Vor dem Start zeigt die GUI einen Preflight mit konkreten Gründen, warum eine Aufnahme nicht
freigegeben ist. Geprüft werden Kamera, UR-Programm/Handshake, beide Panels, Exposure-Fähigkeit
und bei Bedarf SPS, Kalibrierung, Antriebszustand, Nullpunkt, Vorwärtsrichtung und
Positionsfeedback.

Eine unterbrochene Sitzung speichert ihren Zustand atomar in `capture_session.json`.
„Aufnahme fortsetzen“ wird nur für eine unvollständige Sitzung aktiviert. Beim Fortsetzen:

- werden Verbindungen erneut geprüft,
- vorhandene vollständige PNG/YAML-Paare übersprungen,
- unvollständige Paare bewusst als Fehler gemeldet,
- wird der UR-Zielzustand erneut hergestellt,
- wird der gespeicherte SPS-Nullpunkt geladen,
- wird Soll-/Ist-Bandposition verglichen,
- erfolgt niemals ungefragt eine automatische Korrekturfahrt.

Stop verhindert neue Aufträge. Eine bereits laufende UR-Bewegung wird nicht durch einen
externen Bewegungsabbruch gestoppt. Das Förderband erhält dagegen einen definierten Stop und
gibt den Positioniermodus frei.

## 13. Aufnahme-Dateien und Metadaten

Jede Sitzung liegt in:

```text
capture_YYYYMMDD_HHMMSS/
```

Zu jedem Bild existiert eine gleichnamige YAML-Datei. Beispiel für einen synchronisierten
Dateinamen:

```text
img_000001_ura-0155_belt-003_pos-0300_out_ramp-000_p1-020_p2-030_auto.png
```

Wichtige Bestandteile:

- `ur...` für feste Pose-ID,
- `ura-0155` für 15,5 Grad,
- `belt-NNN` für eindeutige Bandstation/Sample-ID,
- `pos-NNNN` für die nominelle Position in Zehntelmillimeter,
- `out`/`back` für Fahrtrichtung,
- optionale `ramp-NNN`,
- `p1`/`p2` für nominelle Sollhelligkeiten,
- `auto` oder Exposure-Kodierung.

Die YAML-Datei enthält unter anderem:

- Dateiname und Zeitstempel,
- Kamera-/Pixel-/Exposure-Daten,
- UR-Modus, Ziel, Quittierung, Gelenke und TCP-Pose,
- Panel-Sollwerte, bestätigte Befehle und Bestätigungsalter,
- Rampen-Sample und Timingabweichung,
- nominelle, quantisierte und gemessene Förderbandposition,
- Station, Hin-/Rückrichtung und SPS-interne Position,
- Kalibrierung, Geschwindigkeit und Bewegungsquittierung.

Die gemessene Bandposition ist für das spätere OBB-Labeling besonders wichtig und wird pro
Bild beibehalten.

## 14. Automatische OBB-Erzeugung

Der Dialog „OBB-Labels …“ verwaltet eine Liste von Klassenquellen. Standardidee:

- Klasse 0: `Pose 1`
- Klasse 1: `Pose 2`
- negative Quelle: `Leere Rutsche`

Weitere Klassen können ergänzt werden. Jede Zeile besitzt einen eigenen Ordnerwähler. Genau
eine Quelle muss als leere Rutsche markiert sein.

### Paarung

Objekt- und Leerbilder müssen dasselbe Aufnahmeprofil besitzen. Der Schlüssel berücksichtigt:

- UR-Pose beziehungsweise Winkel,
- Förderbandstation und Fahrtrichtung,
- Exposure,
- Rampen-Sample-ID,
- Panel-1- und Panel-2-Sollwert.

Abweichende Raster-/Rampenprofile oder fehlende Gegenbilder erzeugen einen konkreten Fehler.

### Klassische Konsenslogik

Bei mehreren Lichtbildern derselben stationären Ansicht werden Leer- und Objektbild zunächst
subpixelgenau registriert. Differenzmasken stimmen anschließend über eine gemeinsame Maske und
OBB ab. Schwache Einzelübereinstimmung wird `REVIEW`.

### Positionsgestützte Förderbandbahn

Bei synchronisierten Serien existiert pro Bandstation häufig nur ein Bild. Besonders bei 0/0
Licht ist eine direkte Segmentierung unmöglich. Deshalb wird zuerst die vollständige Bahn eines
UR-Winkels ausgewertet:

1. Differenzsegmentierungen liefern zunächst OBB-Kandidaten.
2. Ein deterministischer RANSAC-Filter sucht die dominante gerade Mittelpunktbahn und verwirft
   Schatten, Rutschenkanten sowie geometrisch unplausible Boxen. Die Bewertung gewichtet
   Trefferzahl und abgedeckte Bandstrecke, damit eine kurze lokale Kandidatengruppe keine
   schwächer besetzte, aber vollständige Bahn verdrängt.
3. Größe und Orientierung werden ausschließlich aus den verbleibenden Ankern stabilisiert.
4. Falls Bildfortschritt und gemessene Millimeterposition nicht linear gekoppelt sind, wird der
   Fortschritt robust auf einer weiterhin räumlich geraden Bahn modelliert.
5. Schlecht gefüllte Rohmasken werden auf lange dünne Ausläufer geprüft. Eine
   auflösungsabhängige morphologische Öffnung wird nur übernommen, wenn sie den Boxfüllgrad
   deutlich verbessert und mindestens 70 % der Maskenfläche erhält.
6. Fehlende oder stark abweichende Einzelboxen werden aus der geraden Bandbahn ergänzt.
7. Vor dem Export zeigt die GUI je Klasse und UR-Winkel sechs Beispiele über die gesamte
   Bandstrecke; Grün kennzeichnet Anker, Orange das berechnete Bahnmodell. Ablehnen beendet den
   Lauf, bevor der Ausgabeordner angelegt wird.
8. Danach prüft eine Sichtbarkeitsstufe Helligkeit, Clipping, Dynamikumfang und lokalen
   OBB-Kontrast. Verdächtige Bilder erscheinen in einer klickbaren Galerie; nur eindeutig
   unbrauchbare Bilder sind vorausgewählt. Ausschlüsse bleiben im CSV-Bericht auditierbar,
   werden jedoch nicht in `images/` und `labels/` exportiert.
9. Ergänzte Bilder werden nicht stillschweigend als korrekt behandelt, sondern mit
   `quality=REVIEW` und einem verständlichen `quality_reason` protokolliert.

Ein Bild wird jetzt zur automatischen Nichtübernahme empfohlen, wenn es zugleich extrem dunkel,
kontrastarm und lokal schwach ist (`q99 < 30`, Dynamikumfang `< 18`, lokale Stärke `< 10`). Sind
für einen Aufnahmeschlüssel alle positiven Klassen ausgeschlossen, wird auch das dazugehörige
Leerbild nicht in den Datensatz übernommen. Damit lernt YOLO nicht versehentlich „dunkel = leer“.

Synchronisierte Bandserien schreiben im finalen `review/` nur noch ein Sechserblatt pro
Klasse/Winkel und eine Klassenübersicht. Die früheren Vollmasken und Einzelblätter pro
Bandstation waren redundant und konnten bei hochauflösenden Datensätzen mehrere Gigabyte RAM
belegen. Stationäre Rasterserien behalten ihre detaillierten Konsensmasken.

Registrierung, Segmentierung und Sichtbarkeitsbewertung werden über einen begrenzten
`ThreadPoolExecutor` parallelisiert. OpenCV gibt bei den rechenintensiven Operationen den GIL
frei; deshalb skalieren unabhängige Bildpaare auf diesem Pfad gut. Die automatische Workerzahl
ist `min(12, logische CPUs / 2, Anzahl Bilder)`. Ergebnisse werden wieder in Eingabereihenfolge
zusammengesetzt, Fortschrittssignale kommen aus dem Labeling-Worker, und ein Abbruch verwirft
noch wartende Futures. Bahnmodellierung, GUI-Entscheidungen und Export bleiben seriell.

Diese Änderung behebt den früheren Fehler:

```text
Pose 155: kein stabiler Bauteilkonsens gefunden.
```

### OBB-Ausgabe

```text
OBB/
  data.yaml
  label_report.csv
  label_summary.json
  images/train/
  images/val/
  labels/train/
  labels/val/
  review/
```

Labels verwenden das Ultralytics-OBB-Format:

```text
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

Koordinaten sind auf `[0,1]` normiert. Leerbilder besitzen leere `.txt`-Labels. Quelldaten
werden niemals verändert. Auf demselben Laufwerk werden nach Möglichkeit Hardlinks verwendet.

Für wiederholbare Läufe außerhalb der GUI steht `scripts/generate_obb_cli.py` bereit. Es nimmt
beliebig viele Klassenquellen, Leerbildquelle, Ausgabeordner, Konsensparameter,
Mindestdifferenz, Review-Snapshot-Verzeichnis und optional `--exclude-recommended` entgegen.
Anchor-, Sichtbarkeits- und Review-Entscheidungen werden dabei mit den Ergebnissen erhalten.

## 15. Review, Kuratierung und Split-Logik

Im YOLO-Trainingsdialog werden angezeigt:

- Qualität und Prüfgrund,
- Klasse,
- Train/Val/Test,
- UR-Winkel oder Pose-ID,
- gemessene Bandposition,
- Station und Fahrtrichtung,
- beide Panelwerte,
- OBB-Overlay.

Filter:

- auffällige zuerst,
- nur `REVIEW`,
- nur interpolierte OBBs,
- alle,
- ausgeschlossen.

Ein entferntes Häkchen schließt genau ein Bild aus. Es gibt keine automatische Massenlöschung
und keine Mehrheitsabstimmung, durch die eine falsche OBB von fünf richtigen „überstimmt“
würde. Die Entscheidung wird in `curation.json` gespeichert; Quellbilder und Quelllabels
bleiben unverändert.

Beim Dataset-Aufbau werden außerdem die im OBB-Audit mit `excluded_from_dataset=True`
markierten Bilder ausgelassen. Ein fehlendes Leerbild ist nur dann zulässig, wenn sämtliche
damit verknüpften Positivbilder ebenfalls ausgeschlossen wurden.

Der OBB-Ausgang liefert zunächst Train/Val nach vollständigen UR-Winkeln. Beim Kuratieren
werden diese Quell-Splits übernommen. Falls ein unabhängiger Testsatz fehlt, hält die Software
deterministisch einen vollständigen Trainingswinkel als `test` zurück. Niemals werden
Lichtvarianten desselben Winkels auf verschiedene Splits verteilt.

Der kuratierte, versionierte Datensatz enthält:

- `images/{train,val,test}`
- `labels/{train,val,test}`
- `data.yaml`
- `dataset_manifest.json`
- optional eine Kopie von `curation.json`

Das Manifest bewahrt Klasse, Quelle, Split, Reviewentscheidung, UR-Ziel, Lichtwerte,
Rampen-Sample sowie nominelle und gemessene Förderbandposition.

## 16. Ql1i: Quelldaten, OBBs und Kuratierung

Die drei Rohserien liegen außerhalb des Repositories auf Laufwerk `D:`:

```text
Pose 1: D:\pictures\Ql1i\capture_20260808_165705
Pose 2: D:\pictures\Ql1i\capture_20260808_170121
Leer:   D:\pictures\Ql1i\capture_20260808_170321
```

Jede Serie enthält 400 Mono8-Bilder mit 1920 × 1464 Pixeln und 5000 µs Belichtung. Verwendet
wurden die vier UR-Winkel `155`, `180`, `205` und `210`. Der ursprüngliche, teilweise
unvollständige Ordner `D:\pictures\Ql1i\OBB` wurde bewusst nicht überschrieben.

Erhaltene Vergleichsläufe:

```text
D:\pictures\Ql1i\OBB_codex_20260809_v1
D:\pictures\Ql1i\OBB_codex_20260809_v1_review_snapshots
D:\pictures\Ql1i\OBB_codex_20260809_v2_mindiff20
D:\pictures\Ql1i\OBB_codex_20260809_v2_mindiff20_review_snapshots
```

V1 mit Mindestdifferenz 80 war stabiler als V2 mit Mindestdifferenz 20. V1 erzeugte pro
Klasse/Winkel 20/12/17/10 beziehungsweise 23/19/15/11 sichere Anker, 674 Review-Hinweise und
333 positionsinterpolierte OBBs. Die empfindlichere V2-Konfiguration verschlechterte das
Ergebnis auf 702 Review-Hinweise und weniger stabile Pose-1-Anker. Deshalb verwendet der
finale Lauf wieder Mindestdifferenz 80:

```text
Finale OBBs:       D:\pictures\Ql1i\OBB_final_20260809
Review-Snapshots:  D:\pictures\Ql1i\OBB_final_20260809_review_snapshots
```

Finaler OBB-Stand:

- 766 positive Bilder und 387 passende Leerbilder exportiert,
- 34 unbrauchbar dunkle positive Bilder automatisch ausgeschlossen,
- 13 damit verknüpfte Leerbilder ebenfalls nicht als Negativbeispiele übernommen,
- 800 Positivbilder positionsgeführt, davon 333 aus der Bahn interpoliert,
- 799 Positionskorrekturen und 674 Review-Hinweise vor den Ausschlüssen,
- zwei Klassen und vier Winkel,
- keine Quelldateien verändert.

Die Ursache der problematischen Bilder ist nicht nur eine schwache Segmentierung: Trotz
gleicher nomineller Lichtbefehle unterscheiden sich Objekt- und Leeraufnahme optisch stark.
Beispielsweise liegt der Vordergrundmittelwert einer Pose-1-Aufnahme bei 40/40 ungefähr bei
39, der des passenden Leerbildes ungefähr bei 163. Zusätzlich wirft das Bauteil einen langen
Schlagschatten, der rohe OBBs nach links ziehen kann. Die BLE-Bestätigung bestätigt nur das
Kommando, nicht die tatsächlich erreichte optische Beleuchtung.

Der finale kuratierte Datensatz liegt hier:

```text
D:\pictures\Ql1i\YOLO_final_20260809\dataset_20260809_200522
```

Er enthält 1.153 Bilder: 380 Pose 1, 386 Pose 2 und 387 Leerbilder. Die winkelreine Aufteilung
ist:

| Split | Winkel | Pose 1 | Pose 2 | Leer | Gesamt |
| --- | --- | ---: | ---: | ---: | ---: |
| Train | 15,5° + 18,0° | 189 | 191 | 191 | 571 |
| Validation | 20,5° | 96 | 95 | 96 | 287 |
| Test | 21,0° | 95 | 100 | 100 | 295 |

Verlustfreie Resize-Caches wurden für 640 und 960 Pixel erhalten:

```text
D:\pictures\Ql1i\YOLO_final_20260809\dataset_20260809_200522_imgsz640
D:\pictures\Ql1i\YOLO_final_20260809\dataset_20260809_200522_imgsz960
```

## 17. Df1a: erhaltener Referenzlauf

Die ursprünglichen OBBs liegen unter:

```text
C:\Users\Administrator\Pictures\Df1a\OBB
```

Sie enthalten 2.420 Bilder: vier Klassen mit je 484 positiven Bildern sowie 484 Leerbilder.
Der kuratierte Datensatz wurde als
`C:\Users\Administrator\Pictures\Df1a\YOLO\dataset_20260809_184640` erhalten und besitzt
1.210 Train-, 605 Validation- und 605 Testbilder. Der 640-Pixel-Cache liegt daneben als
`dataset_20260809_184640_imgsz640`.

Der finale Trainingslauf liegt unter:

```text
C:\Users\Administrator\Pictures\Df1a\YOLO\runs\Df1a_yolo26n_obb_20260809
```

Er war auf 50 Epochen mit Patience 12 angesetzt, stoppte nach Epoche 16 und wählte Epoche 4.
Das stabile, gegen den Run-Checkpoint per SHA geprüfte Modell ist:

```text
C:\Users\Administrator\Pictures\Df1a\YOLO\Df1a_best.pt
```

Unabhängiger Test: Precision 0,93386, Recall 0,94725, mAP50 0,98628 und mAP50–95 0,89037.
Auf allen 121 Leerbildern trat kein Fehlalarm auf. Eine zusätzliche praktische Stichprobe mit
je einem Bild pro Klasse plus Leerbild wurde vollständig richtig klassifiziert.

## 18. Kk1a: finaler OBB- und YOLO-Lauf

Die fünf Rohserien liegen unter `C:\Users\Administrator\Pictures\AutomatedImageCapture`.
Chronologisch wurden sie wie folgt zugeordnet:

```text
Pose 1: capture_20260810_213525
Pose 2: capture_20260810_214724
Leer:   capture_20260810_215123
Pose 3: capture_20260810_215358
Pose 4: capture_20260810_220005
```

Jede Serie enthält 500 vollständige PNG/YAML-Paare: je 100 Bilder bei den Winkeln 15,5°,
17,0°, 18,5°, 20,0° und 21,0°. Der finale OBB-Lauf und seine separat erhaltenen
Review-Snapshots liegen hier:

```text
C:\Users\Administrator\Pictures\AutomatedImageCapture\Kk1a_OBB_20260810
C:\Users\Administrator\Pictures\AutomatedImageCapture\Kk1a_OBB_20260810_review_snapshots
```

Er enthält 2.000 positive Bilder, 500 Leerbilder, vier Klassen und fünf Winkel. Es wurden
keine Bilder automatisch ausgeschlossen. Alle 2.000 Positivbilder sind positionsgeführt,
1.980 wurden durch das Bahnmodell korrigiert und nur 16 vollständig interpoliert. Die
schwächste Ankergruppe ist Pose 4 bei 17° mit 30/100 Ankern; auch deren Review-Sheet wurde
visuell als plausibel geprüft.

Der kuratierte Datensatz lautet:

```text
C:\Users\Administrator\Pictures\AutomatedImageCapture\Kk1a_YOLO_20260810\dataset_20260810_221045
```

| Split | Winkel | Bilder | Je Pose | Leer |
| --- | --- | ---: | ---: | ---: |
| Train | 15,5° + 17,0° + 20,0° | 1.500 | 300 | 300 |
| Validation | 18,5° | 500 | 100 | 100 |
| Test | 21,0° | 500 | 100 | 100 |

Training: YOLO26n-OBB, 640 Pixel, Batch 16, acht Worker, maximal 75 Epochen und Patience 15.
Early Stopping beendete den Lauf nach 21 Epochen; beste Trainingsepoche war Epoche 6. Der
vollständige Run liegt unter:

```text
C:\Users\Administrator\Pictures\AutomatedImageCapture\Kk1a_YOLO_20260810\runs\Kk1a_yolo26n_obb_20260810
```

Empfohlenes Modell und unveränderliche Baseline:

```text
C:\Users\Administrator\Pictures\AutomatedImageCapture\Kk1a_YOLO_20260810\Kk1a_best.pt
C:\Users\Administrator\Pictures\AutomatedImageCapture\Kk1a_YOLO_20260810\Kk1a_best_baseline_20260810.pt
```

Beide Kopien und `weights\best.pt` besitzen SHA-256
`033660BA5181DA2DC79E4B1D0BB725DEC098F9A4A451FB8C9290D03DF1829FE5`.

Validation: Precision 0,9545, Recall 0,9493, mAP50 0,9910 und mAP50–95 0,8977.
Unabhängiger 21°-Test: Precision 0,9425, Recall 0,9473, mAP50 0,9882 und mAP50–95 0,8602.
Klassenweises Test-mAP50–95: Pose 1 0,6886, Pose 2 0,8895, Pose 3 0,9321 und Pose 4
0,9308.

Der Rohbild-Konfidenz-Sweep ist als `confidence_sweep.json` erhalten. Für die GUI wird `0,26`
empfohlen: 392/400 Positivbilder korrekt, vier Pose-2-Bilder verpasst, vier Pose-4-Bilder als
Pose 2 erkannt und 0/100 Fehlalarme auf Leerbildern. Bei 0,25 gab es noch einen Leerbild-
Fehlalarm; höhere Schwellen verwerfen zunehmend richtige Pose-2-Bilder.

## 19. Kl1i: finaler OBB- und YOLO-Lauf

Die drei Rohserien liegen unter `C:\Users\Administrator\Pictures\Kl1i`:

```text
Pose 1: capture_20260810_220517
Pose 2: capture_20260810_220814
Leer:   capture_20260810_221324
```

Jede Serie enthält 500 vollständige PNG/YAML-Paare, je 100 bei 15,5°, 17,0°, 18,5°,
20,0° und 21,0°. Finale OBBs und separat erhaltene Review-Snapshots:

```text
C:\Users\Administrator\Pictures\Kl1i\Kl1i_OBB_20260811
C:\Users\Administrator\Pictures\Kl1i\Kl1i_OBB_20260811_review_snapshots
```

Der OBB-Lauf enthält 1.000 positive Bilder und 500 Leerbilder. Es gab keine Ausschlüsse und
keine vollständig interpolierte OBB; alle Positivbilder sind positionsgeführt und 999 Boxen
wurden durch das Bahnmodell stabilisiert. Die schwächsten Gruppen besitzen 35/100 sichere
Einzelanker. Deren Review-Sheets wurden ebenso wie stärkere Referenzgruppen visuell geprüft.

Kuratierter Datensatz:

```text
C:\Users\Administrator\Pictures\Kl1i\Kl1i_YOLO_20260811\dataset_20260811_112136
```

| Split | Winkel | Bilder | Pose 1 | Pose 2 | Leer |
| --- | --- | ---: | ---: | ---: | ---: |
| Train | 15,5° + 17,0° + 20,0° | 900 | 300 | 300 | 300 |
| Validation | 18,5° | 300 | 100 | 100 | 100 |
| Test | 21,0° | 300 | 100 | 100 | 100 |

Das YOLO26n-OBB-Training verwendete 640 Pixel, Batch 16, acht Worker, maximal 75 Epochen und
Patience 15. Es stoppte nach Epoche 42; beste Epoche war 27. Vollständiger Run:

```text
C:\Users\Administrator\Pictures\Kl1i\Kl1i_YOLO_20260811\runs\Kl1i_yolo26n_obb_20260811
```

Empfohlenes Modell und unveränderliche Baseline:

```text
C:\Users\Administrator\Pictures\Kl1i\Kl1i_YOLO_20260811\Kl1i_best.pt
C:\Users\Administrator\Pictures\Kl1i\Kl1i_YOLO_20260811\Kl1i_best_baseline_20260811.pt
```

Beide Kopien und `weights\best.pt` besitzen SHA-256
`642F646E0F7560091F4837833AFC3E9756F1A38A26ABB6D5F5FC18B26CDD0A83`.

Validation: Precision 0,9982, Recall 0,9950, mAP50 0,9950 und mAP50–95 0,9769.
Unabhängiger 21°-Test: Precision 0,9909, Recall 0,9900, mAP50 0,9940 und mAP50–95 0,9603.
Klassenweises Test-mAP50–95: Pose 1 0,9949 und Pose 2 0,9258.

Der Rohbild-Sweep ist als `confidence_sweep.json` erhalten. Für die GUI wird Konfidenz `0,25`
empfohlen. Von 0,05 bis 0,32 bleibt das Ergebnis stabil: 199/200 Positivbilder korrekt, eine
Pose-2-Aufnahme verpasst, keine Klassenverwechslung und 0/100 Leerbild-Fehlalarme. Ab 0,35
wird ein weiteres positives Bild verworfen.

## 20. YOLO-Training und Performance

Das Training läuft als separater Prozess, damit Hardwareanzeige und GUI responsiv bleiben.
Beim Stoppen wird zuerst ein geordneter Abbruch versucht; Checkpoints und Logs bleiben erhalten.

Aktuelle Implementierung und Defaults:

- Modell `yolo26n-obb.pt`, maximal 200 Epochen, Patience 40,
- Bildgröße 640 Pixel,
- Batch 16 und acht DataLoader-Worker,
- Batch und Worker sind im Trainingsdialog einstellbar und werden an den Prozess übergeben,
- Gerät standardmäßig GPU `0`, AMP aktiv,
- Seed 42 und deterministische Ausführung,
- keine Spiegelungen, kein Mosaic, MixUp oder Copy-Paste und keine große Rotation,
- geringe Translation, Skalierung und Helligkeitsvariation,
- einmaliger verlustfreier Resize-Cache, um wiederholtes Dekodieren und Skalieren zu vermeiden.

Batch 16/Workers 8 wurde auf der RTX 2000 Ada mit 8 GB getestet. Bei Df1a sank die gemessene
Epochendauer gegenüber Batch 4/Workers 0 auf ungefähr 17–20 Sekunden; die GPU belegte dabei
nur etwa 2,5–2,9 GB VRAM. Mehr CPU-Auslastung allein ist daher kein sinnvolles Ziel: Training,
Datenladen, GPU-Kernel und Synchronisation begrenzen verschiedene Phasen.

### Ql1i-Modell

Alle finalen Ql1i-Ergebnisse liegen unter:

```text
D:\pictures\Ql1i\YOLO_final_20260809
```

Empfohlenes Modell und unveränderliche Sicherung:

```text
D:\pictures\Ql1i\YOLO_final_20260809\Ql1i_best.pt
D:\pictures\Ql1i\YOLO_final_20260809\Ql1i_best_baseline_20260809.pt
```

Der zugehörige Run `runs\Ql1i_yolo26n_obb_20260809` verwendete 640 Pixel, Batch 16 und acht
Worker. Er war auf 75 Epochen mit Patience 15 angesetzt, stoppte ungefähr nach Epoche 20 und
wählte Epoche 5. Validation: Precision 0,7003, Recall 0,8592, mAP50 0,86646 und mAP50–95
0,54012. Unabhängiger Test: Precision 0,6861, Recall 0,86395, mAP50 0,86166 und mAP50–95
0,60046. Klassenweise erreichte Pose 1 mAP50/mAP50–95 von 0,761/0,472, Pose 2 von
0,962/0,729; die 100 Leerbilder erzeugten keinen Fehlalarm.

Der Konfidenz-Sweep auf allen 295 Testbildern ergab:

| Konfidenz | Pose 1 korrekt | Pose 2 korrekt | Leer korrekt | Klassenverwechslungen |
| ---: | ---: | ---: | ---: | ---: |
| 0,10 | 95/95 | 96/100 | 100/100 | 0 |
| 0,15 | 89/95 | 96/100 | 100/100 | 0 |
| 0,20 | 70/95 | 96/100 | 100/100 | 0 |
| 0,25 | 47/95 | 94/100 | 100/100 | 0 |

Für den aktuellen Ql1i-Liveversuch ist deshalb Konfidenz 0,10 sinnvoll. Der erhaltene
960-Pixel-Vergleich `runs\Ql1i_yolo26n_obb_960_20260809` erzielte zwar auf Validation
mAP50–95 0,77186, fiel auf dem unabhängigen Testwinkel aber auf mAP50 0,6920 und mAP50–95
0,5189 zurück. Insbesondere Pose 1 erreichte dort nur 0,436/0,359. Dieser Lauf hat den
Validation-Winkel überangepasst und wird nicht empfohlen.

Nach jedem Training bleiben `best.pt`, Ultralytics-Plots, Validation-/Testmetriken,
klassenweise Ergebnisse, Confusion-Matrizen, Leerbild-Fehlalarmrate und Zusammenfassung
erhalten. `.pt`-Dateien sind absichtlich in `.gitignore` und müssen separat gesichert werden.

Klassennamen in der Trainingszusammenfassung werden aus `dataset_manifest.json` gelesen. Eine
frühere hart codierte Zwei-Klassen-Annahme wurde beim Kk1a-Lauf entfernt; damit erscheinen
auch Pose 3, Pose 4 und weitere Klassen korrekt in den klassenweisen JSON-Metriken.

## 21. YOLO-Live-Inferenz

Das Hauptfenster kann ein trainiertes OBB-Modell auf dem jeweils neuesten Kameraframe
ausführen. Die Inferenz läuft in einem eigenen Thread; alte Frames werden verworfen statt
aufgestaut. Einstellbar sind Modellpfad, Konfidenz, Eingangsgröße und maximale Inferenz-FPS.

Das Overlay zeigt OBB, Klasse und Konfidenz. Der erste GPU-Frame ist wegen CUDA-Initialisierung
typischerweise deutlich langsamer.

Automatische Modellsuche verwendet historisch den Ordner:

```text
%USERPROFILE%\Pictures\Kk1_pose12_yolo26_obb\runs
```

Für Ql1i sollte `D:\pictures\Ql1i\YOLO_final_20260809\Ql1i_best.pt` mit Konfidenz 0,10 im
Hauptfenster explizit gewählt werden, solange der historische Suchpfad nicht umgestellt wurde.

## 22. Was Git bewusst nicht enthält

Die `.gitignore` schließt lokale, große oder reproduzierbare Artefakte aus:

- `.diagnostics/`
- root-lokale `capture_*`, `captures/`
- `datasets/`, `images/`, `labels/`, `review/`, `runs/`
- `temp/`, `tmp/`, `*.tmp`, `*.temp`
- `.pt`-Modelle
- Logs, virtuelle Umgebung und Caches

Für den PC-Wechsel müssen daher separat gesichert werden:

1. `D:\pictures\Ql1i` mit Rohbildern, OBBs, Datensätzen, Caches und Trainingsläufen.
2. `C:\Users\Administrator\Pictures\Df1a` mit OBBs, Datensatz und Referenzmodell.
3. `C:\Users\Administrator\Pictures\AutomatedImageCapture` mit Kk1a-Rohbildern, finalen
   OBBs, Review-Snapshots, Datensatz, Cache, Trainingslauf und stabilen Modellen.
4. `C:\Users\Administrator\Pictures\Kl1i` mit Rohbildern, finalen OBBs, Review-Snapshots,
   Datensatz, Cache, Trainingslauf und stabilen Modellen.
5. Gewünschte `best.pt`-Modelle und komplette Trainings-Runordner.
6. Optional QSettings/Registry-Export.
7. Baumer Camera Explorer/GenTL-Producer-Installer.
8. BT540-/Bluetooth-Treiber.
9. TwinCAT ADS Runtime und die AMS-Routen.
10. Das externe Beckhoff-/SPS-Projekt beziehungsweise `CSVSaver`, falls es auf dem neuen PC
   bearbeitet werden soll.
11. Ein Backup der tatsächlich auf dem UR getesteten `.urp`-/Installationsdateien.

Die Dateien unter `ur_program/` und die GUI-Quellen selbst sind dagegen versioniert.

## 23. Checkliste für einen neuen PC

### Software

- [ ] Repository klonen und `uv sync --extra dev` ausführen.
- [ ] NVIDIA-Treiber/CUDA-Verfügbarkeit mit dem Diagnosekommando prüfen.
- [ ] Baumer Camera Explorer beziehungsweise CTI-Producer installieren.
- [ ] BT540/BLE-Treiber installieren.
- [ ] TwinCAT ADS Runtime/Router installieren.

### Netzwerk und Routen

- [ ] Kamera-NIC so konfigurieren, dass `169.254.117.70` erreichbar ist.
- [ ] UR-NIC so konfigurieren, dass `10.10.10.10` erreichbar ist.
- [ ] Beckhoff-NIC so konfigurieren, dass `192.168.10.23` erreichbar ist.
- [ ] Lokale und entfernte AMS-Route für `10.145.4.14.1.1` einrichten.
- [ ] ADS-Port 851 und ADS/TCP 48898 prüfen.

### Hardwareprogramme

- [ ] Camera Explorer vor GUI-Verbindung schließen.
- [ ] Beide Panels einschalten; Smartphone-App trennen.
- [ ] BLE-Geräte in der GUI neu suchen und getrennt speichern.
- [ ] Passendes URP manuell laden/starten.
- [ ] Kontinuierlichen UR-Wegpunkt am Teach Pendant prüfen.
- [ ] RTDE-Register 42/43 sowie Outputs 41–43 auf Konflikte prüfen.
- [ ] SPS-Programm starten und `StepperInternalPosition`-Verknüpfung prüfen.
- [ ] Sicherstellen, dass keine zweite Band-/RTDE-Steuerung aktiv ist.

### GUI-Inbetriebnahme

- [ ] Kamera, UR, Förderband und beide Panels einzeln verbinden.
- [ ] Status und Livebild prüfen.
- [ ] Förderbandkalibrierung muss gültig sein.
- [ ] Fahrweg eingeben und eine kurze Testfahrt links/rechts durchführen.
- [ ] Vorwärtsrichtung bestätigen.
- [ ] Bauteil hinten platzieren und logischen Nullpunkt setzen.
- [ ] Zunächst Pilotserie mit einem Winkel und kleiner Strecke aufnehmen.
- [ ] PNG/YAML-Paare und gemessene Bandposition prüfen.
- [ ] Erst danach eine vollständige Serie starten.

### Daten und Training

- [ ] Ql1i-, Df1a-, Kk1a- und Kl1i-Bilder, OBBs, Runordner und Modelle separat auf den neuen PC
  kopieren.
- [ ] Pfade in OBB- und YOLO-Dialog neu auswählen.
- [ ] „Bilder laden / aktualisieren“ ausführen.
- [ ] Interpolierte/REVIEW-Overlays prüfen.
- [ ] Kuratierten Datensatz erzeugen und Splitzahlen kontrollieren.
- [ ] Erst dann Training starten.

## 24. Tests und Abnahme

Standardprüfung:

```powershell
uv run pytest
uv run ruff check .
```

Stand bei Erstellung dieses Dokuments:

```text
132 Tests bestanden, 6 Hardwaretests standardmäßig übersprungen
```

Hardwaretests:

```powershell
$env:RUN_HARDWARE_TESTS = "1"
uv run pytest -m hardware tests/hardware
```

Vor einer echten Hardware-Abnahme sollten gleichzeitig geprüft werden:

- 100 Kameraframes ohne Bufferfehler,
- mindestens 30 Sekunden UR-Status ohne Bewegungs-/Schreibbefehl,
- zwei getrennt verbundene Panels und sichtbare manuelle Reaktion,
- Bandfahrt 0 → Maximum → 0 mit plausibler interner Position,
- zwei UR-Winkel mit vollständiger synchronisierter Lichtvariation,
- Stop, Geräteverlust und Fortsetzen ohne GUI-Blockade,
- erfolgreiche OBB-Erzeugung aus Objekt- und Leerbildserie,
- gültiger Train-/Val-/Test-Datensatz.

### Speicherbereinigung

Das Hauptfenster enthält **Speicher bereinigen …**. Der Dialog erkennt ausschließlich
abgeschlossene Capture-Sitzungen sowie gültige OBB- und kuratierte YOLO-Datensätze. Eine
Nur-Analyse zeigt logische und physische Belegung, Hardlink-Gruppen, rekonstruierbare Caches,
übersprungene Bilder und die geschätzte Einsparung. Die Standardkodierung ist Mono-PNG,
Originalauflösung, Kompressionsstufe 3. PNG, JPEG und WebP werden von Labeling, Kuratierung und
Trainingscache unterstützt.

Erst eine separate Bestätigung ersetzt Bilder und löscht validierte `_imgsz…`- beziehungsweise
Ultralytics-Caches endgültig. Zwischen-Checkpoints, Review-Bilder, YAML und Labels bleiben
erhalten. Ein Journal schützt unterbrochene Läufe; ein erneuter Analyse-/Ausführungslauf
repariert den konsistent abgeschlossenen Zwischenstand. Jeder erfolgreiche oder bewusst
abgebrochene Lauf schreibt `cleanup_report_*.json` in den gewählten Ordner.

## 25. Bekannte Grenzen und nächste sinnvolle Arbeiten

1. Die automatische OBB-Erzeugung ist bewusst konservativ. Interpolierte Bilder müssen im
   Review beurteilt werden; das System ist kein vollautomatischer Ground-Truth-Ersatz.
2. Ql1i besitzt nur vier Winkel und zeigt starke optische Abweichungen trotz nominell gleicher
   Lichtbefehle. Für neue Serien mindestens ein Panel mit etwa 40 % betreiben, diffuses
   frontales Fülllicht verwenden, 700–1000 ms Licht-Settle-Time vorsehen und Winkelabstände von
   höchstens 1° aufnehmen. Positive und leere Serien müssen dieselbe physische Lichtsequenz
   verwenden; 0/0 und 20/0 sollten vermieden werden.
3. Das Ql1i-Modell ist bei Pose 1 und höheren Konfidenzschwellen noch empfindlich. Konfidenz
   0,10 ist für den aktuellen Versuch empirisch besser als der frühere Default 0,25.
4. Der automatische Live-Modellsuchpfad enthält noch den historischen Namen `Kk1`.
5. QSettings, Netzwerk, ADS-Routen und BLE-Auswahl sind rechnerlokal und brauchen eine
   bewusstere Export-/Importfunktion, falls häufig zwischen PCs gewechselt wird.
6. Die GUI steuert keine SPS-Kalibrierung. Eine ungültige Kalibrierung muss in TwinCAT behoben
   werden.
7. Automatische Roboterbewegung bleibt absichtlich auf das Registerprotokoll und lokal geprüfte
   UR-Programme beschränkt.
8. Automatische semantische Annotation jenseits der aktuellen OBB-Differenz-/Bahnlogik ist ein
   möglicher späterer Meilenstein.

## 26. Leitprinzipien für die Weiterentwicklung

- Keine Quelldaten ohne separate Analyse, Integritätsprüfung und ausdrückliche Bestätigung
  überschreiben oder massenhaft löschen.
- Jede Aufnahme muss reproduzierbare Metadaten besitzen.
- Tatsächliche Messwerte klar von Sollwerten und bestätigten Befehlen unterscheiden.
- Licht-, Band-, Kamera- und UR-Fehler getrennt behandeln.
- Lange Hardware- und Trainingsoperationen niemals im GUI-Thread ausführen.
- Alte Sessions, Dateinamen und feste Pose-IDs weiterhin lesbar halten.
- Lichtvarianten eines physisch gleichen UR-Winkels nicht über Splits verteilen.
- Unsicherheit sichtbar machen (`REVIEW`) statt fehlerhafte Sicherheit vorzutäuschen.
- Keine Erweiterung darf die bestehenden UR-Sicherheitsgrenzen umgehen.
- Jede relevante Fehlerbehebung durch einen automatisierten Regressionstest absichern.

