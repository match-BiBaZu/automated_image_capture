# Repository-local Windows launcher

Run `uv sync` in this repository first, then double-click
`Verknuepfungen-installieren.cmd`. Python, source code and the custom icon all
come from this repository; no sibling repository is required.

Installs **BiBaZu Automated Image Capture** on the Desktop and in Start Menu > BiBaZu.
Only this application's shortcut is created/replaced. Existing shortcuts for
other applications are left alone. No devices are connected by the installer.
This is a launcher, not a bundled executable: Python dependencies and camera/hardware drivers are still required.

```powershell
.\WindowsLaunchers\Install-BiBaZuShortcuts.ps1 -StartMenuOnly
.\WindowsLaunchers\Install-BiBaZuShortcuts.ps1 -DesktopOnly
.\WindowsLaunchers\Install-BiBaZuShortcuts.ps1 -CheckOnly
.\WindowsLaunchers\Uninstall-BiBaZuShortcuts.ps1 -StartMenuOnly
```

Installation resolves the current clone's absolute path. Re-run the installer
after moving/cloning the repository or recreating its `.venv`.
`-DestinationDirectory <folder>` permits a custom/test shortcut location.
Icons (PNG source and multi-resolution ICO) live in `icons/`.
