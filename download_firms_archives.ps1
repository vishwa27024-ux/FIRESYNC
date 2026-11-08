# Download NASA FIRMS archive request IDs for FIRESYNC.
# Run this from the FIRESYNC project root in PowerShell.
# The script preserves each ZIP and validates that it is a readable archive.

$ErrorActionPreference = 'Stop'

$RequestIds = @('839652', '839653', '839654', '839655')
$OutputDir = Join-Path (Get-Location) 'data\archive_downloads\nasa_firms'

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
Add-Type -AssemblyName System.IO.Compression.FileSystem

$Results = foreach ($Id in $RequestIds) {
    $FileName = "DL_FIRE_SV-C2_$Id.zip"
    $FinalPath = Join-Path $OutputDir $FileName
    $TempPath = "$FinalPath.part"
    $Url = "https://firms.modaps.eosdis.nasa.gov/data/download/$FileName"

    Write-Host "`n[$Id] Checking archive link..." -ForegroundColor Cyan

    if (Test-Path -LiteralPath $FinalPath) {
        try {
            $Zip = [System.IO.Compression.ZipFile]::OpenRead($FinalPath)
            $EntryCount = $Zip.Entries.Count
            $Zip.Dispose()
            if ($EntryCount -gt 0) {
                Write-Host "Already downloaded and ZIP-verified: $FileName" -ForegroundColor Green
                [PSCustomObject]@{
                    RequestId = $Id
                    Status = 'already_present_valid_zip'
                    Bytes = (Get-Item -LiteralPath $FinalPath).Length
                    Entries = $EntryCount
                    Path = $FinalPath
                    Note = ''
                }
                continue
            }
        }
        catch {
            Write-Warning "Existing file is not a valid ZIP; preserving it and not overwriting: $FinalPath"
            [PSCustomObject]@{
                RequestId = $Id
                Status = 'existing_file_invalid_manual_review'
                Bytes = (Get-Item -LiteralPath $FinalPath).Length
                Entries = 0
                Path = $FinalPath
                Note = $_.Exception.Message
            }
            continue
        }
    }

    if (Test-Path -LiteralPath $TempPath) {
        Remove-Item -LiteralPath $TempPath -Force
    }

    try {
        Invoke-WebRequest -Uri $Url -OutFile $TempPath -TimeoutSec 900 -MaximumRedirection 10

        $Zip = [System.IO.Compression.ZipFile]::OpenRead($TempPath)
        $EntryCount = $Zip.Entries.Count
        $Zip.Dispose()

        if ($EntryCount -lt 1) {
            throw 'The downloaded ZIP contains no entries.'
        }

        Move-Item -LiteralPath $TempPath -Destination $FinalPath
        $Bytes = (Get-Item -LiteralPath $FinalPath).Length

        Write-Host "Downloaded and ZIP-verified: $FileName ($Bytes bytes, $EntryCount entries)" -ForegroundColor Green

        [PSCustomObject]@{
            RequestId = $Id
            Status = 'downloaded_valid_zip'
            Bytes = $Bytes
            Entries = $EntryCount
            Path = $FinalPath
            Note = ''
        }
    }
    catch {
        if (Test-Path -LiteralPath $TempPath) {
            Remove-Item -LiteralPath $TempPath -Force
        }

        $Message = $_.Exception.Message
        Write-Warning "Could not download request $Id. Check that NASA has finished processing it and that its link is still active. Details: $Message"

        [PSCustomObject]@{
            RequestId = $Id
            Status = 'download_failed_or_not_ready'
            Bytes = 0
            Entries = 0
            Path = ''
            Note = $Message
        }
    }
}

Write-Host "`n========== DOWNLOAD SUMMARY ==========" -ForegroundColor Cyan
$Results | Format-Table RequestId, Status, Bytes, Entries, Path -AutoSize

$SummaryPath = Join-Path $OutputDir 'download_summary.csv'
$Results | Export-Csv -LiteralPath $SummaryPath -NoTypeInformation -Encoding UTF8
Write-Host "`nSummary saved to: $SummaryPath"
Write-Host "Archive folder: $OutputDir"
