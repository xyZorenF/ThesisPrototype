# PowerShell script to create a 500‑image per class subset of JMuBEN
# -------------------------------------------------
# Destination folder (will be created if missing)
$dstRoot = "C:\Users\Admin\Desktop\thesisPROTOTYPE\dataset\jmuben_subset\JMuBEN"

# Source folder containing the full JMuBEN class directories
$srcRoot = "C:\Users\Admin\Desktop\thesisPROTOTYPE\dataset\jmuben\JMuBEN"

# Classes present in the dataset (adjust if you have more/less)
$classes = @('Healthy','Cerscospora','Leaf rust','Miner','Phoma')

foreach ($cls in $classes) {
    $src = Join-Path $srcRoot $cls
    $dst = Join-Path $dstRoot $cls
    # Ensure destination exists
    New-Item -ItemType Directory -Force -Path $dst | Out-Null
    # Select 500 random JPG files (or all if fewer)
    $files = Get-ChildItem $src -Filter *.jpg | Sort-Object {Get-Random}
    $take = [Math]::Min(500, $files.Count)
    $files | Select-Object -First $take | ForEach-Object {
        Copy-Item $_.FullName (Join-Path $dst $_.Name) -Force
    }
    Write-Host "Copied $take images to $dst"
}

Write-Host "Subset creation complete."