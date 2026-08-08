$ErrorActionPreference = "Stop"

$directory = Split-Path -Parent $MyInvocation.MyCommand.Path
$sourcePath = Join-Path $directory "BiBaZu_GUI_native.urp.xml"
$targetPath = Join-Path $directory "BiBaZu_Continuous.urp"
$document = [xml][IO.File]::ReadAllText($sourcePath)
$document.PreserveWhitespace = $true
$program = $document.DocumentElement
$program.SetAttribute("name", "BiBaZu_Continuous")

$beforeStart = $program.SelectSingleNode("children/SpecialSequence")
$beforeChildren = $beforeStart.SelectSingleNode("children")
$mainProgram = $program.SelectSingleNode("children/MainProgram")
$mainChildren = $mainProgram.SelectSingleNode("children")
$assignment = $mainChildren.SelectSingleNode("Assignment").CloneNode($true)
$scriptTemplate = $beforeChildren.SelectSingleNode("Script").CloneNode($true)
$moveTemplate = $mainChildren.SelectSingleNode("Switch/children/Case[@caseValue='180']/children/Move").CloneNode($true)

function New-ScriptNode([string]$fileName) {
    $node = $scriptTemplate.CloneNode($true)
    $scriptPath = Join-Path $directory $fileName
    $node.SelectSingleNode("cachedContents").InnerText = [IO.File]::ReadAllText($scriptPath)
    $node.SelectSingleNode("file").InnerText = $fileName
    return $node
}

while ($beforeChildren.HasChildNodes) { [void]$beforeChildren.RemoveChild($beforeChildren.FirstChild) }
while ($mainChildren.HasChildNodes) { [void]$mainChildren.RemoveChild($mainChildren.FirstChild) }

[void]$beforeChildren.AppendChild((New-ScriptNode "BiBaZu_Continuous_Init.script"))
$feature = $moveTemplate.SelectSingleNode("feature")
$feature.RemoveAllAttributes()
$feature.SetAttribute("class", "GeomFeatureReference")
$feature.SetAttribute("referencedName", "Joint_0_name")
$waypoint = $moveTemplate.SelectSingleNode("children/Waypoint")
$waypoint.SetAttribute("name", "Rotation_Position")
[void]$beforeChildren.AppendChild($moveTemplate)
[void]$beforeChildren.AppendChild((New-ScriptNode "BiBaZu_Continuous_CapturePose.script"))
[void]$mainChildren.AppendChild($assignment)
[void]$mainChildren.AppendChild((New-ScriptNode "BiBaZu_Continuous_Wait.script"))
[void]$mainChildren.AppendChild((New-ScriptNode "BiBaZu_Continuous_Move.script"))
[void]$mainChildren.AppendChild((New-ScriptNode "BiBaZu_Continuous_Ack.script"))

$utf8 = [Text.UTF8Encoding]::new($false)
$outputBytes = $utf8.GetBytes($document.OuterXml)
$targetStream = [IO.File]::Create($targetPath)
$gzip = [IO.Compression.GzipStream]::new($targetStream, [IO.Compression.CompressionLevel]::Optimal)
$gzip.Write($outputBytes, 0, $outputBytes.Length)
$gzip.Dispose()
$targetStream.Dispose()
Write-Output $targetPath
