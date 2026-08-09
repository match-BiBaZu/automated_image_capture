# Automated Image Capture

Eine ausführliche fachliche und technische Projektübergabe einschließlich PC-Migration,
Hardwarevoraussetzungen, Datenformaten und offenen Punkten steht in
[`PROJECT_HANDOFF.md`](PROJECT_HANDOFF.md).

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
Konsensbasierte OBB-Annotation, kuratierte Zwei-Klassen-Datensätze und ein reproduzierbares
YOLO26n-OBB-Training sind ebenfalls enthalten.

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
   Die Belichtungszeit kann direkt in der Kamerakarte eingetragen und mit **Übernehmen**
   gesetzt werden. Falls nötig, wird `ExposureAuto` dabei für diese Verbindung deaktiviert.
   Beim Trennen werden Belichtungszeit und Automatikmodus vom Verbindungsaufbau
   wiederhergestellt.
5. Beim UR die RTDE-/Dashboard-Anzeigen und insbesondere den Safety Mode prüfen.
6. Für einen Pose-Auftrag das vorbereitete `BiBaZu`-Programm manuell starten, eine Ansicht
   auswählen und die einmalige Bewegungsfreigabe bestätigen.
7. Die Lichter erst nach erfolgreicher Verbindung über Ein/Aus, Helligkeit, CCT oder HSI ändern.

## Kontinuierlicher UR-Winkel und Förderband

Zusätzlich zu den festen Pose-IDs unterstützt die Aufnahme `BiBaZu_Continuous.urp`.
Das am Handbediengerät eingelernte XYZ bleibt erhalten; variiert wird der Winkel von
15,5 bis 21,0 Grad. Das fertige URP und seine Skriptdateien liegen unter `ur_program/`.

Das Förderband wird über TwinCAT ADS angesprochen. Vorgaben sind SPS `192.168.10.23`,
AMS-Net-ID `10.145.4.14.1.1` und PLC-Port 851. Windows benötigt dafür die TwinCAT-ADS-
Runtime, eine beidseitige AMS-Route und eine Netzwerkkonfiguration, welche die SPS erreicht.
`PressureControlGUI` darf während der Aufnahme nicht gleichzeitig das Band steuern.

Vor einer neuen Serie wird das Bauteil hinten platziert und in der Förderbandkarte
**Aktuelle Position = 0 mm** gewählt. Nach zwei 1-mm-Testfahrten wird Links oder Rechts als
Vorwärtsrichtung gespeichert. `Term 19 → POS Status → Actual position` muss in TwinCAT
mit `MAIN.StepperInternalPosition` verknüpft sein. Die GUI gibt den synchronisierten Modus erst
frei, nachdem eine Testfahrt eine echte Änderung dieses Werts gezeigt hat.

Der Modus **Diskrete Stationen** verwendet weiterhin die Reihenfolge UR-Ziel →
Förderbandstation → Belichtung → vollständige Lichtvariation. Standardmäßig fährt das
Band `0, 10, 20, 30, 40, 50, 40, 30, 20, 10, 0 mm`; Hin- und Rückweg sind getrennte
Stationen. Im Modus **Kontinuierliche synchronisierte Fahrt** fährt das Band pro UR-Ziel und
Belichtung ohne Zwischenstopps `0 → Maximum → 0 mm`. Kamera und beide Panels laufen
währenddessen weiter. Erst nach der bestätigten Rückkehr zu 0 mm wird der UR-Winkel geändert.
Eine SPS-Fertigmeldung wird dabei erst akzeptiert, wenn auch `StepperInternalPosition` den
berechneten Endpunkt erreicht hat. Eine bis zu zwei Sekunden verzögerte Positionsrückmeldung
wird mit 10 Hz nachgeführt, ohne die enge Endpunkttoleranz aufzuweichen.

Beim Stoppen sendet die Anwendung einen Förderband-Stop und gibt den SPS-Positioniermodus
frei. Eine UR-Fahrt wird weiterhin nicht extern abgebrochen. Beim Fortsetzen wird die interne
SPS-Position gegen den gespeicherten Nullpunkt geprüft. Abweichungen erfordern eine separat
bestätigte Korrekturfahrt.

## Kamera-Bildrate

Der Kamera-Worker leert den GenTL-Datenstrom stets mit der von Belichtung, Kamera und Netzwerk
erreichbaren Rohbildrate. Unter **Einstellungen** kann das kameraseitige
`AcquisitionFrameRate`-Limit abgeschaltet werden; dies ist standardmäßig aktiv. Die dort separat
konfigurierte **Maximale Vorschau-FPS** begrenzt nur die Darstellung in der GUI. Während einer
automatischen Aufnahme erhält der Aufnahmecontroller einen eigenen Framepfad mit der im
Aufnahmedialog gewählten Bildrate. Mono-Bilder bleiben dort einkanalig und werden nicht unnötig
zu RGB verdreifacht. Die
YOLO-Liveerkennung besitzt weiterhin ihr eigenes FPS-Limit.

Die Belichtungszeit begrenzt die physikalisch mögliche Bildrate unmittelbar: 250000 µs erlauben
höchstens etwa 4 FPS, 5000 µs belichtungsseitig bis zu 200 FPS. Die tatsächlich gemessenen Werte
zeigt die Kamerakarte als **Kamera / Rohabruf / Vorschau**. Die schnelle Rampenaufnahme kann mit
bis zu 240 Bildern/s konfiguriert werden, startet aber nur, wenn der gemessene Rohabruf die
gewählte Rate erreicht. Bei zeitgesteuerten Hochgeschwindigkeitsaufnahmen werden weiterhin
verlustfreie, aber unkomprimierte PNGs geschrieben. Damit wird CPU-Kompression vermieden; der
Datenträger muss bei 1920 × 1200 Mono8 ungefähr 2,3 MB pro Bild aufnehmen können. Die
Writer-Warteschlange bricht kontrolliert ab, statt Arbeitsspeicher unbegrenzt zu belegen.

Für eine isolierte Messung bei geschlossener GUI:

```powershell
uv run python scripts/benchmark_camera.py --frames 300
```

## YOLO im Kamera-Livebild

Oberhalb des Kamerabilds lässt sich **YOLO Live-Erkennung** zuschalten. Beim ersten Start
wählt die GUI automatisch das neueste `best.pt` unter
`Pictures\Kk1_pose12_yolo26_obb\runs`; über **Modell …** kann jederzeit ein anderes
Ultralytics-OBB-Modell gewählt werden. Konfidenzschwelle und maximale Inferenz-Bildrate sind
direkt einstellbar und werden gespeichert.

Die erkannten orientierten Bounding Boxes, Klassen (`Pose 1`/`Pose 2`) und Konfidenzen werden
in das Livebild eingezeichnet. Neben den Reglern stehen Objektzahl und Laufzeit des letzten
Frames. Die Inferenz läuft auf der GPU in einem eigenen Thread und verarbeitet nur das jeweils
neueste Kamerabild. Dadurch stauen sich bei hoher Kamera-Bildrate keine alten Bilder an und die
Hardware-Bedienung bleibt responsiv. Der erste Frame benötigt wegen der CUDA-Initialisierung
deutlich länger als die folgenden Frames.

## Automatische Aufnahme

1. Passend zum UR-Modus `BiBaZu_GUI.urp` oder `BiBaZu_Continuous.urp` am Teach Pendant
   laden und manuell starten.
2. Kamera, UR, beide Panels und bei aktivierter Positionsvariation das Förderband verbinden.
3. Unter **Aufnahme konfigurieren** Speicherort, UR-Modus und den Lichtmodus festlegen:
   **Exaktes Raster** verwendet feste Helligkeitsschritte; **Schnelle Rampe** nimmt während
   zeitbasierter Dreieckskurven beider Panels kontinuierlich Bilder auf.
4. Für eine Bandserie Vorwärtsrichtung bestätigen, das Bauteil hinten platzieren und die
   aktuelle Position ausdrücklich als logischen Nullpunkt übernehmen. Für den
   synchronisierten Modus zuvor per 1-mm-Testfahrt die Positionsrückmeldung prüfen.
5. Optional die Belichtungsvariation aktivieren, wenn die verbundene Kamera eine manuell
   beschreibbare `ExposureTime` meldet.
6. **Aufnahme starten** wählen und die Freigabe des Arbeitsraums bestätigen.

Die allgemeine Reihenfolge ist UR-Ziel → Bandstation → Belichtung → Lichtvariation. Im
Rastermodus läuft Panel 1 je Panel-2-Wert vollständig durch. Im Rampenmodus wird pro
Belichtung eine vollständige Rampe aufgenommen. Das Standardprofil
speichert 60 Bilder in 10 Sekunden bei Panelperioden von 2,4 und 10 Sekunden. Pro Panel läuft
höchstens ein BLE-Befehl gleichzeitig; veraltete Zwischenziele werden übersprungen. Die
Dateinamen enthalten `ramp-NNN`, und die YAML-Dateien dokumentieren Sollzeit, tatsächliche
Aufnahmeabweichung sowie den letzten bestätigten Panelbefehl.

Bei der **kontinuierlichen synchronisierten Fahrt** lautet die Reihenfolge UR-Ziel →
Belichtung → kompletter Bandlauf mit gleichzeitiger Lichtvariation. Die Rundfahrtdauer wird
aus Weg und Geschwindigkeit bestimmt. Im Rampenmodus werden die Lichtperioden proportional
auf diese Dauer skaliert. Im diskreten Modus werden die Lichtpaare über die Fahrt verteilt;
bei zu vielen Kombinationen wird die Bandgeschwindigkeit automatisch reduziert. Veraltete
BLE-Zwischenziele werden nicht aufgestaut. Jede YAML-Datei enthält geplante und gemessene
Bandposition, Richtung, Sollhelligkeiten, zuletzt bestätigte Panelwerte sowie Zeitabweichung.
Jede Sitzung erhält einen eigenen Ordner `capture_YYYYMMDD_HHMMSS`. Zu jeder verlustfreien
PNG-Datei wird eine gleichnamige YAML-Datei mit Kamera-, Roboter-, Licht- und Sequenzdaten
gespeichert. Eine optionale Belichtungsvariation wird beim Abschluss oder Stoppen auf den Wert
vom Kamera-Verbindungsaufbau zurückgesetzt.

**Aufnahme stoppen** verhindert weitere Aufträge und Bilder. Eine bereits begonnene Bewegung
wird aus Sicherheitsgründen nicht durch einen externen Stop-Befehl unterbrochen.

Wird eine Sitzung durch einen Gerätefehler, **Aufnahme stoppen** oder das Beenden der GUI
unterbrochen, wird **Aufnahme fortsetzen** aktiviert. Die Sitzung speichert ihren Fortschritt
atomar in `capture_session.json` und kann daher auch nach einem GUI-Neustart wiederhergestellt
werden. Vor dem Fortsetzen werden alle Hardwareverbindungen erneut geprüft und bereits
vollständige PNG-/YAML-Paare übersprungen. Ein unvollständiges Dateipaar führt bewusst zu einer
Fehlermeldung, damit keine vorhandene Aufnahme unbemerkt überschrieben wird. Der UR fährt beim
Fortsetzen die nächste benötigte freigegebene Pose erneut über den normalen Handshake an.

## Automatische OBB-Labels

Bei synchronen Förderbandaufnahmen liest das Labeling zusätzlich
`conveyor.measured_logical_offset_mm` aus jeder Bild-YAML. Die OBB-Mittelpunkte werden pro
UR-Orientierung als gerade Bildbahn über der gemessenen Bandposition modelliert. Ein robuster
Ankerfilter verwirft Schatten, Rutschenkanten sowie unplausibel große Boxen, bevor Größe und
Orientierung bestimmt werden. Fehlerhafte Einzelsegmentierungen werden durch die Bahn
stabilisiert und im `label_report.csv` weiterhin als `REVIEW` nachvollziehbar markiert.
Fehlen Positionsmetadaten, bleibt das bisherige Konsensverfahren unverändert aktiv.

Bei mehreren konkurrierenden Bahnhypothesen zählt nicht nur die Trefferzahl, sondern auch die
abgedeckte Förderbandstrecke. Falls der Bildfortschritt durch Abtastverzug oder Schlupf nicht
linear zur Millimeterposition ist, bleibt die Mittelpunktbahn räumlich gerade und nur der
Fortschritt auf dieser Geraden wird robust quadratisch modelliert. Lange, dünne Ausläufer einer
Differenzmaske (zum Beispiel eine angeschlossene Rutschenkante) werden nur bei auffällig
schlecht gefüllten Rohboxen entfernt; dabei müssen mindestens 70 % der Maskenfläche erhalten
bleiben. Kompakte Masken verwenden unverändert den bisherigen Pfad.

Nach der einmaligen Segmentierung pausiert die GUI vor dem Export. Sie zeigt für jede Klasse
und jeden UR-Winkel sechs über die gesamte Bandstrecke verteilte Beispiele: Grün bedeutet
`ANKER`, Orange bedeutet eine aus der Bahn berechnete Box. Erst **Anker übernehmen** schreibt
noch nicht sofort den YOLO-Datensatz: Anschließend bewertet eine zweite Stufe globale
Helligkeit, Clipping, Dynamikumfang und den lokalen Kontrast in der erwarteten OBB. Verdächtige
Bilder erscheinen als Thumbnail-Galerie. Eindeutig schwarze oder weiße Bilder sind zum
Ausschluss vorausgewählt, dunkle Grenzfälle bleiben eine bewusste Nutzerentscheidung. Erst
**Auswahl übernehmen** startet den Export; bei Abbruch bleibt der Ausgabeordner unangetastet.
Ausgeschlossene Bilder stehen mit Sichtbarkeitsscore und Begründung weiterhin im
`label_report.csv`, werden aber weder als Trainingsbild noch als Label geschrieben.

Für synchronisierte Bandserien enthält `review/` je Klasse und UR-Winkel ein kompaktes Blatt
mit sechs über die Strecke verteilten Bildern sowie eine Klassenübersicht. Dadurch werden bei
hochauflösenden Serien nicht mehr pro einzelner Bandposition redundante Vollmasken und
2×3-Blätter erzeugt. Klassische Rasterserien behalten ihre bisherigen Konsensmasken.

Über **OBB-Labels …** lässt sich aus beliebig vielen Bauteilserien und einer parametrisch
identischen Leerbildserie ein gemeinsamer YOLO-OBB-Datensatz erzeugen. Die Quellenliste startet
mit `Pose 1`, `Pose 2` und `Leere Rutsche`; weitere Posen lassen sich hinzufügen oder wieder
entfernen. Jede Pose erhält anhand ihrer Listenposition eine eigene Klassen-ID und in jeder
Zeile kann der zugehörige Aufnahmeordner gewählt werden. Das Tool paart die Aufnahmen anhand
von UR-Pose, beiden Panelhelligkeiten und Belichtung. Vor der Differenzbildung wird jedes
Leerbild subpixelgenau registriert, damit kleine Wiederholabweichungen auf strukturierten
Untergründen nicht als Objekt erscheinen.

Alle Beleuchtungsvarianten derselben Klasse und UR-Pose stimmen über eine gemeinsame
Segmentierungsmaske und eine gemeinsame orientierte Bounding Box ab. Einzelbilder mit
schwacher Übereinstimmung werden in `label_report.csv` als `REVIEW` markiert. Unter `review/`
entstehen repräsentative Overlays und Konsensmasken für jede Klasse und UR-Pose.

Der Ausgabeordner ist direkt als Ultralytics-YOLO-OBB-Datensatz aufgebaut:

- `images/train`, `images/val`
- `labels/train`, `labels/val` mit acht normierten OBB-Koordinaten
- `data.yaml`
- `label_summary.json` und `label_report.csv`

Der Train-/Val-Split erfolgt klassenübergreifend nach vollständigen UR-Posen statt zufällig
nach Bildern. Dadurch landen Beleuchtungsvarianten derselben Ansicht nicht zugleich in
Training und Validierung. Optional werden die Leerbilder genau einmal mit leeren Labeldateien
als negative Beispiele übernommen.
Auf demselben Laufwerk nutzt das Tool Hardlinks, sodass dabei keine zweite Kopie der großen
PNG-Dateien entsteht.

Die beiden BLE-Adressen werden getrennt gespeichert, sodass „Alle verbinden“ nicht beide
Adapter demselben Panel zuordnet. Die Lichtwerte sind als „bestätigter letzter Befehl“
gekennzeichnet. Das BLE-Protokoll liefert
nicht in jeder Firmware verlässliche physische Istwerte zurück. Beim Beenden wird das Panel
nicht automatisch ausgeschaltet oder umgestellt.

## Kuratierung und YOLO26-OBB-Training

Über **YOLO-Training …** wird ein gemeinsamer, vom OBB-Labeltool erzeugter Datensatzordner
geladen. Klassen und Namen werden aus `label_summary.json` übernommen, sodass neben `Pose 1`
und `Pose 2` auch weitere Klassen ohne zusätzliche Quellfelder unterstützt werden. Leerbilder
werden dabei nur einmal übernommen. Der Review zeigt
automatisch auffällige Aufnahmen zuerst; ein entferntes Häkchen schließt ausschließlich dieses
Bild aus. Die Entscheidung wird in `curation.json` gespeichert und verändert keine Quelldatei.

Der erzeugte, versionierte Datensatz besitzt getrennte Ordner für Train, Validation und Test.
Alle Lichtvarianten derselben UR-Pose bleiben im selben Split. `dataset_manifest.json`
dokumentiert Quelle, Klasse, Beleuchtung, Split und Review-Entscheidung jeder Aufnahme.

Nach der Datensatzprüfung startet **Training starten** `yolo26n-obb.pt` in einem separaten
Prozess. Standardmäßig werden 200 Epochen bei 640 Pixeln mit Early Stopping trainiert. Das
Training verwendet keine Spiegelungen oder starken Rotationen, da diese die beiden physisch
unterschiedlichen Bauteilposen verfälschen könnten. `best.pt`, Validierungs-/Testmetriken,
Confusion-Matrizen und die Fehlalarmrate auf leeren Testbildern liegen anschließend im
angezeigten Ergebnisordner.

Dieselben Schritte sind ohne GUI reproduzierbar:

```powershell
uv run python -m automated_image_capture.training prepare --source <OBB-DATENSATZORDNER>
uv run python -m automated_image_capture.training train --dataset <DATENSATZORDNER>
uv run python -m automated_image_capture.training diagnose
```

Der erste Modellaufruf lädt die vortrainierten Gewichte, sofern sie noch nicht lokal vorhanden
sind. Das Projekt verwendet für die vorhandene RTX 3060 PyTorch mit CUDA 11.8.

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
- **Kurzer Kameraaussetzer:** Die Karte wechselt zunächst auf „Eingeschränkt“ und startet den
  GenTL-Datenstrom automatisch neu. Erst wenn zwei Wiederherstellungsversuche scheitern, wird
  die laufende Aufnahmeserie unterbrochen und kann anschließend fortgesetzt werden.
- **Nur ein UR-Kanal:** RTDE kann in den Robot Security Settings deaktiviert sein; Dashboard und
  RTDE werden deshalb getrennt als „eingeschränkt“ dargestellt.
- **Kein Licht:** Bluetooth-Symbol am Panel aktivieren, Smartphone-App trennen und näher an den
  PC bringen.

Ein passiver BLE-Scan kann bei Bedarf mit `uv run python scripts/probe_ble.py` ausgeführt werden.
Mit `--connect <Adresse>` liest das Skript zusätzlich nur die angebotenen GATT-Service-UUIDs.

Rotierende Logs liegen unter
`%LOCALAPPDATA%\AutomatedImageCapture\logs\dashboard.log`. Es werden keine Bilder protokolliert.
