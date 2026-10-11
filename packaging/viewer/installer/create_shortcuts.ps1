# Desktop + Start-menu "QuantUI Viewer" shortcuts -> cmd /c "<prefix>\Scripts\quantui.exe view || pause"
# Run by post_install.bat; errors go to <prefix>\post_install.log.
param([Parameter(Mandatory = $true)][string]$Prefix)
$ErrorActionPreference = 'Stop'
$shell = New-Object -ComObject WScript.Shell
foreach ($dir in @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs'))) {
    if (-not $dir) { continue }
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $lnk = $shell.CreateShortcut((Join-Path $dir 'QuantUI Viewer.lnk'))
    # Through cmd so a failure stays on screen ("|| pause") instead of the
    # console closing before the message can be read.
    $exe = Join-Path $Prefix 'Scripts\quantui.exe'
    $lnk.TargetPath = $env:ComSpec
    $lnk.Arguments = '/c ""' + $exe + '" view || pause"'
    $lnk.IconLocation = $exe
    $lnk.WorkingDirectory = $env:USERPROFILE
    $lnk.Description = 'Browse QuantUI results (History + Analysis)'
    $lnk.Save()
    Write-Output "shortcut: $(Join-Path $dir 'QuantUI Viewer.lnk')"
}
