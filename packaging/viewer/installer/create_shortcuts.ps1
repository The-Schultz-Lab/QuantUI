# Desktop + Start-menu "QuantUI Viewer" shortcuts -> <prefix>\Scripts\quantui.exe view
# Run by post_install.bat; errors go to <prefix>\post_install.log.
param([Parameter(Mandatory = $true)][string]$Prefix)
$ErrorActionPreference = 'Stop'
$shell = New-Object -ComObject WScript.Shell
foreach ($dir in @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs'))) {
    if (-not $dir) { continue }
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $lnk = $shell.CreateShortcut((Join-Path $dir 'QuantUI Viewer.lnk'))
    $lnk.TargetPath = Join-Path $Prefix 'Scripts\quantui.exe'
    $lnk.Arguments = 'view'
    $lnk.WorkingDirectory = $env:USERPROFILE
    $lnk.Description = 'Browse QuantUI results (History + Analysis)'
    $lnk.Save()
    Write-Output "shortcut: $(Join-Path $dir 'QuantUI Viewer.lnk')"
}
