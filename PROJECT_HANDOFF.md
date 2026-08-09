# Projektübergabe: Automated Image Capture

Stand: 8. August 2026  
Repository: `https://github.com/match-BiBaZu/automated_image_capture`  
Referenzstand bei Erstellung dieses Dokuments: Branch `main`, Commit `48ce931`

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
- PyTorch-CUDA-11.8-Wheels für die vorhandene RTX 3060

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
- Die GUI erhält nur das neueste Bild und standardmäßig maximal 15 FPS.
- Unterstützt werden Mono-, hochbitige Mono-, Bayer-, RGB- und BGR-Formate.
- Hochbitige Bilder werden nur für die Vorschau skaliert; gespeicherte Daten bleiben
  verlustfrei.
- Status: Modell, Seriennummer, IP, Auflösung, Pixelformat, Kamera-FPS, Vorschau-FPS und
  Belichtungszeit.
- Exposure kann in der Kamerakarte manuell gesetzt werden. Dazu darf `ExposureAuto` für die
  aktuelle Verbindung deaktiviert werden. Beim Trennen werden die beim Verbinden gelesenen
  Werte wiederhergestellt.

Der Baumer Camera Explorer muss geschlossen sein. Andernfalls ist die Kamera meist exklusiv
belegt.

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
`C:\Users\nunning\BiBaZu_Big_Boi\CSVSaver` beziehungsweise dessen PressureControlGUI
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
- Vor der ersten Serie müssen 1-mm-Testfahrten erfolgen.
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
   Schatten, Rutschenkanten sowie geometrisch unplausible Boxen.
3. Größe und Orientierung werden ausschließlich aus den verbleibenden Ankern stabilisiert.
4. Fehlende oder stark abweichende Einzelboxen werden aus der geraden Bandbahn ergänzt.
5. Vor dem Export zeigt die GUI je Klasse und UR-Winkel sechs Beispiele über die gesamte
   Bandstrecke; Grün kennzeichnet Anker, Orange das berechnete Bahnmodell. Ablehnen beendet den
   Lauf, bevor der Ausgabeordner angelegt wird.
6. Ergänzte Bilder werden nicht stillschweigend als korrekt behandelt, sondern mit
   `quality=REVIEW` und einem verständlichen `quality_reason` protokolliert.

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

## 16. Aktueller Ql1i-Datensatz

Die derzeitigen Quelldaten liegen außerhalb des Repositories:

```text
Pose 1: C:\Users\nunning\Pictures\Ql1i\capture_20260808_165705
Pose 2: C:\Users\nunning\Pictures\Ql1i\capture_20260808_170121
Leer:   C:\Users\nunning\Pictures\Ql1i\capture_20260808_170321
OBB:    C:\Users\nunning\Pictures\Ql1i\OBB
```

Geprüfter Stand:

- 400 Bilder Pose 1,
- 400 Bilder Pose 2,
- 400 Leerbilder,
- 1.200 Bilder und 1.200 passende Labeldateien,
- UR-Winkel `155`, `180`, `205`, `210`, also 15,5°, 18,0°, 20,5°, 21,0°,
- 800 positive Bilder positionsgeführt,
- 333 OBBs ohne sichere Einzelsegmentierung aus der Bandbahn interpoliert,
- 413 positive Bilder vorsichtshalber als `REVIEW` markiert,
- keine unlesbaren Bilder und keine ungültigen OBB-Koordinaten.

Der YOLO-Dialog erzeugt daraus aktuell folgende winkelreine Aufteilung:

| Split | Winkel | Bilder |
| --- | --- | ---: |
| Train | 15,5° und 18,0° | 600 |
| Validation | 20,5° | 300 |
| Test | 21,0° | 300 |

Jeder Winkel enthält beide Klassen und die passenden Leerbilder. Vor einem produktiven
Training sollten vor allem die 333 interpolierten Overlays stichprobenartig geprüft werden.

## 17. YOLO-Training

Das Training läuft als separater Prozess, damit Hardwareanzeige und GUI responsiv bleiben.
Beim Stoppen wird zuerst ein geordneter Abbruch versucht; Checkpoints und Logs bleiben erhalten.

Aktuelle Implementierung:

- Modell: `yolo26n-obb.pt`
- maximal 200 Epochen
- Early Stopping/Patience 40
- Bildgröße in der GUI derzeit 640 Pixel
- Batch derzeit 4
- Gerät standardmäßig GPU `0`
- AMP aktiv
- Seed 42, deterministisch
- keine horizontalen/vertikalen Spiegelungen
- kein Mosaic, MixUp oder Copy-Paste
- keine große Rotation
- nur geringe Translation, Skalierung und Helligkeitsvariation

Die ursprünglich formulierte Zielkonfiguration sah `imgsz=1024` und automatische Batchgröße
vor. Der aktuelle Code verwendet aus praktischen Gründen 640 und Batch 4. Diese bewusste
Abweichung sollte vor einem finalen Trainingslauf erneut bewertet werden.

Nach dem Training werden gespeichert:

- `best.pt`,
- Kurven und Ultralytics-Plots,
- Validation- und Testmetriken,
- OBB mAP-Werte,
- klassenweise Ergebnisse,
- Confusion-Matrizen,
- Leerbild-Fehlalarmrate,
- vollständige Trainingszusammenfassung.

Kommandos:

```powershell
uv run python -m automated_image_capture.training prepare `
  --source C:\Users\nunning\Pictures\Ql1i\OBB

uv run python -m automated_image_capture.training train `
  --dataset <KURATIERTER-DATENSATZ>

uv run python -m automated_image_capture.training diagnose
```

`.pt`-Dateien sind absichtlich in `.gitignore` und müssen separat übertragen werden.

## 18. YOLO-Live-Inferenz

Das Hauptfenster kann ein trainiertes OBB-Modell auf dem jeweils neuesten Kameraframe
ausführen. Die Inferenz läuft in einem eigenen Thread; alte Frames werden verworfen statt
aufgestaut. Einstellbar sind Modellpfad, Konfidenz, Eingangsgröße und maximale Inferenz-FPS.

Das Overlay zeigt OBB, Klasse und Konfidenz. Der erste GPU-Frame ist wegen CUDA-Initialisierung
typischerweise deutlich langsamer.

Automatische Modellsuche verwendet historisch den Ordner:

```text
%USERPROFILE%\Pictures\Kk1_pose12_yolo26_obb\runs
```

Für Ql1i sollte der gewünschte `best.pt` im Hauptfenster explizit gewählt werden, solange der
Suchpfad nicht auf den neuen Datasetnamen umgestellt wurde.

## 19. Was Git bewusst nicht enthält

Die `.gitignore` schließt lokale, große oder reproduzierbare Artefakte aus:

- `.diagnostics/`
- root-lokale `capture_*`, `captures/`
- `datasets/`, `images/`, `labels/`, `review/`, `runs/`
- `temp/`, `tmp/`, `*.tmp`, `*.temp`
- `.pt`-Modelle
- Logs, virtuelle Umgebung und Caches

Für den PC-Wechsel müssen daher separat gesichert werden:

1. `C:\Users\nunning\Pictures\Ql1i` mit Rohbildern und OBB-Datensatz, falls weiter benötigt.
2. Gewünschte `best.pt`-Modelle und komplette Trainings-Runordner.
3. Optional QSettings/Registry-Export.
4. Baumer Camera Explorer/GenTL-Producer-Installer.
5. BT540-/Bluetooth-Treiber.
6. TwinCAT ADS Runtime und die AMS-Routen.
7. Das externe Beckhoff-/SPS-Projekt beziehungsweise `CSVSaver`, falls es auf dem neuen PC
   bearbeitet werden soll.
8. Ein Backup der tatsächlich auf dem UR getesteten `.urp`-/Installationsdateien.

Die Dateien unter `ur_program/` und die GUI-Quellen selbst sind dagegen versioniert.

## 20. Checkliste für einen neuen PC

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
- [ ] 1-mm-Testfahrt links/rechts durchführen.
- [ ] Vorwärtsrichtung bestätigen.
- [ ] Bauteil hinten platzieren und logischen Nullpunkt setzen.
- [ ] Zunächst Pilotserie mit einem Winkel und kleiner Strecke aufnehmen.
- [ ] PNG/YAML-Paare und gemessene Bandposition prüfen.
- [ ] Erst danach eine vollständige Serie starten.

### Daten und Training

- [ ] Ql1i-Bilder/Modelle separat auf den neuen PC kopieren.
- [ ] Pfade in OBB- und YOLO-Dialog neu auswählen.
- [ ] „Bilder laden / aktualisieren“ ausführen.
- [ ] Interpolierte/REVIEW-Overlays prüfen.
- [ ] Kuratierten Datensatz erzeugen und Splitzahlen kontrollieren.
- [ ] Erst dann Training starten.

## 21. Tests und Abnahme

Standardprüfung:

```powershell
uv run pytest
uv run ruff check .
```

Stand bei Erstellung dieses Dokuments:

```text
118 Tests bestanden, 6 Hardwaretests standardmäßig übersprungen
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

## 22. Bekannte Grenzen und nächste sinnvolle Arbeiten

1. Die automatische OBB-Erzeugung ist bewusst konservativ. Interpolierte Bilder müssen im
   Review beurteilt werden; das System ist kein vollautomatischer Ground-Truth-Ersatz.
2. Der aktuelle Ql1i-Datensatz besitzt viele dunkle Bilder. Die Positionsbahn macht sie
   labelbar, aber ihre Eignung für das Training sollte empirisch bewertet werden.
3. Die Trainingsdefaults 640/Batch 4 weichen vom ursprünglichen Ziel 1024/Auto-Batch ab.
4. Der automatische Live-Modellsuchpfad enthält noch den historischen Namen `Kk1`.
5. QSettings, Netzwerk, ADS-Routen und BLE-Auswahl sind rechnerlokal und brauchen eine
   bewusstere Export-/Importfunktion, falls häufig zwischen PCs gewechselt wird.
6. Die GUI steuert keine SPS-Kalibrierung. Eine ungültige Kalibrierung muss in TwinCAT behoben
   werden.
7. Automatische Roboterbewegung bleibt absichtlich auf das Registerprotokoll und lokal geprüfte
   UR-Programme beschränkt.
8. Automatische semantische Annotation jenseits der aktuellen OBB-Differenz-/Bahnlogik ist ein
   möglicher späterer Meilenstein.

## 23. Leitprinzipien für die Weiterentwicklung

- Keine Quelldaten überschreiben oder automatisch massenhaft löschen.
- Jede Aufnahme muss reproduzierbare Metadaten besitzen.
- Tatsächliche Messwerte klar von Sollwerten und bestätigten Befehlen unterscheiden.
- Licht-, Band-, Kamera- und UR-Fehler getrennt behandeln.
- Lange Hardware- und Trainingsoperationen niemals im GUI-Thread ausführen.
- Alte Sessions, Dateinamen und feste Pose-IDs weiterhin lesbar halten.
- Lichtvarianten eines physisch gleichen UR-Winkels nicht über Splits verteilen.
- Unsicherheit sichtbar machen (`REVIEW`) statt fehlerhafte Sicherheit vorzutäuschen.
- Keine Erweiterung darf die bestehenden UR-Sicherheitsgrenzen umgehen.
- Jede relevante Fehlerbehebung durch einen automatisierten Regressionstest absichern.

