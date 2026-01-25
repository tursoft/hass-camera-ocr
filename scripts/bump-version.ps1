# Version bump script for Camera OCR
# Usage: .\scripts\bump-version.ps1 [-Message "commit message"]

param(
    [string]$Message = "Version bump"
)

$ErrorActionPreference = "Stop"

# Read current version from config.yaml
$configPath = "$PSScriptRoot\..\hass_camera_ocr\config.yaml"
$configContent = Get-Content $configPath -Raw
$currentVersion = [regex]::Match($configContent, 'version:\s*(\d+\.\d+\.\d+)').Groups[1].Value

# Parse version parts
$parts = $currentVersion.Split('.')
$major = [int]$parts[0]
$minor = [int]$parts[1]
$patch = [int]$parts[2]

# Increment patch version
$patch++
$newVersion = "$major.$minor.$patch"

Write-Host "Bumping version: $currentVersion -> $newVersion" -ForegroundColor Cyan

# Update config.yaml
$configContent = $configContent -replace "version:\s*$currentVersion", "version: $newVersion"
Set-Content $configPath $configContent -NoNewline

# Update manifest.json
$manifestPath = "$PSScriptRoot\..\custom_components\hass_camera_ocr\manifest.json"
$manifestContent = Get-Content $manifestPath -Raw
$manifestContent = $manifestContent -replace "`"version`":\s*`"$currentVersion`"", "`"version`": `"$newVersion`""
Set-Content $manifestPath $manifestContent -NoNewline

# Update README.md badge
$readmePath = "$PSScriptRoot\..\README.md"
$readmeContent = Get-Content $readmePath -Raw
$readmeContent = $readmeContent -replace "version-$currentVersion-blue", "version-$newVersion-blue"
$readmeContent = $readmeContent -replace "Version $currentVersion", "Version $newVersion"
Set-Content $readmePath $readmeContent -NoNewline

# Add changelog entry
$changelogPath = "$PSScriptRoot\..\hass_camera_ocr\CHANGELOG.md"
$changelogContent = Get-Content $changelogPath -Raw
$date = Get-Date -Format "yyyy"
$newEntry = "# Changelog`n`n## [$newVersion] - $date`n`n### Changed`n- $Message`n"
$changelogContent = $changelogContent -replace "# Changelog`n`n", $newEntry
Set-Content $changelogPath $changelogContent -NoNewline

Write-Host "Updated files:" -ForegroundColor Green
Write-Host "  - hass_camera_ocr/config.yaml"
Write-Host "  - custom_components/hass_camera_ocr/manifest.json"
Write-Host "  - README.md"
Write-Host "  - hass_camera_ocr/CHANGELOG.md"

# Git operations
Write-Host "`nCommitting and pushing to git..." -ForegroundColor Cyan
Set-Location "$PSScriptRoot\.."
git add -A
git commit -m "$Message (v$newVersion)"
git push

Write-Host "`nDone! Version $newVersion pushed to git." -ForegroundColor Green
