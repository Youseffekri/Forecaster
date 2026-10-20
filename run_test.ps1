# Get today's date in yyyy_MM_dd format
$today = Get-Date -Format "yyyy_MM_dd"
$testName = "ForecasterPy_rev2.1"

# Set output log directory
$logDir = "log\test\$today\$testName"

# Create log directory if it doesn't exist
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# Option A: Explicit list of files from your image
$examples = @(
    "Example_Covid.py",
    "Example_ETTh1.py",
    "Example_FeatureSelection.py",
    "Example_GEFCom.py",
    "Example_ILI.py",
    "Example_LLM_NARX_SR.py"
)

# Option B: Or automatically pick up ALL .py files in Examples directory:
# $examples = Get-ChildItem -Path ".\Examples" -Filter "*.py" | Select-Object -ExpandProperty Name

foreach ($file in $examples) {
    # Remove extension for log filename (e.g., Example_Covid)
    $basename = [System.IO.Path]::GetFileNameWithoutExtension($file)
    $logPath = Join-Path -Path $logDir -ChildPath "${basename}.log"

    Write-Host "Running Examples\$file -> $logPath"
    python ".\Examples\$file" all > $logPath 2>&1
}

Write-Host "All Python example runs completed!"