# Assemble the paper submission package (JPDC, single-anonymized review) under submit\ from the current paper\ build.
#
#   powershell -ExecutionPolicy Bypass -File src\revloop\make_submit.ps1 [-Build] [-Verify] [-NoArchive] [-OutDir <dir>]
#
#   -Build      run paper\build.bat first (otherwise the existing paper\build\ outputs are used)
#   -Verify     extract LaTeX_Source.zip into a temporary directory, compile its two folders standalone
#               and report pages, undefined references and errors
#   -NoArchive  replace the package in place instead of moving it to <package>\old\<timestamp>\
#   -OutDir     assemble the package in this directory instead of submit\ (e.g. for a test build)
#   -Artifact   folder of the code to archive (default codeocean\code; e.g. the code\ folder
#               written by capsule_build.py --out-dir on a machine without the capsule submodule)
#
# Layout (one folder per upload file type; the two LaTeX folders are also zipped together):
#   main.pdf, appendix.pdf                    compiled main paper and Supplementary Material
#   MainManuscript\                           main.tex, ref.bib, main.bbl, the document class and
#                                             bibliography style that main.tex names, figs\,
#                                             build\appendix_only.aux (resolves S-prefixed references)
#   LaTeXSupplementaryFile\                   appendix_only.tex (standalone wrapper), appendix.tex,
#                                             appendix_only.bbl, figs\, build\main.aux (M- references)
#   Image\                                    every figure file of the main paper and the supplement
#   SourceCode\                               the code and result files (codeocean\code as last built by
#                                             capsule_build.py, or the folder given with -Artifact)
#   SupplementaryMaterialForReview\           Supplementary_Material.pdf and Code_and_Results.zip
#                                             (an archive of SourceCode\)
#   LaTeX_Source.zip                          MainManuscript\ and LaTeXSupplementaryFile\ in one archive
#   Image.zip                                 the figures
#   cover_letter.txt, Title_Page.docx,        from docs\
#   Highlights.txt
# Figures: only the files named by \includegraphics commands that are not commented out in main.tex
# and appendix.tex. A figure in paper\figs\profile\ (author photos) or in paper\backup\, legacy\ or
# template\ (material of earlier venues) stops the script, and at the end every packaged figure is
# checked against the included ones. Nothing else from paper\ is copied.
# submit\ is git-ignored. Earlier packages are kept under submit\old\.
param([switch]$Build, [switch]$Verify, [switch]$NoArchive, [string]$OutDir = '', [string]$Artifact = '')
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$paper = Join-Path $Root 'paper'
if ($OutDir) { $sub = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutDir) }
else { $sub = Join-Path $Root 'submit' }
$managed = 'MainManuscript', 'LaTeXSupplementaryFile', 'Image', 'SupplementaryMaterialForReview', 'SourceCode', 'Author_Photos'
# folders of paper\ that must never reach an upload (author photos, earlier venues' material)
$forbidden = 'profile', 'backup', 'legacy', 'template'

if ($Build) {
  & cmd /c (Join-Path $paper 'build.bat')
  if ($LASTEXITCODE -ne 0) { throw 'paper\build.bat failed; see paper\build\*.log' }
}
foreach ($f in 'main.pdf', 'appendix.pdf', 'build\main.aux', 'build\main.bbl', 'build\appendix_only.aux',
               'build\appendix_only.bbl', 'build\appendix_only.tex') {
  if (-not (Test-Path (Join-Path $paper $f))) { throw "missing paper\${f}: run with -Build or run paper\build.bat" }
}
$builtAt = (Get-Item (Join-Path $paper 'main.pdf')).LastWriteTime
$stale = 'main.tex', 'appendix.tex', 'ref.bib' | ForEach-Object { Get-Item (Join-Path $paper $_) } |
  Where-Object { $_.LastWriteTime -gt $builtAt }
if ($stale) { Write-Warning ("sources newer than paper\main.pdf: " + ($stale.Name -join ', ') + "; consider -Build") }

# zip with forward-slash entry names (Compress-Archive of Windows PowerShell 5.1 writes backslashes,
# which other systems extract as flat file names)
Add-Type -AssemblyName System.IO.Compression, System.IO.Compression.FileSystem
# -AsFolder keeps each folder's name as the top-level entry, so several folders can share one archive
function New-PortableZip($dirs, $zip, [switch]$AsFolder) {
  if (Test-Path $zip) { Remove-Item $zip -Force }
  $fs = [IO.File]::Open($zip, [IO.FileMode]::CreateNew)
  $za = New-Object IO.Compression.ZipArchive($fs, [IO.Compression.ZipArchiveMode]::Create)
  try {
    foreach ($dir in @($dirs)) {
      $base = (Resolve-Path $dir).Path.TrimEnd('\')
      if ($AsFolder) { $base = (Split-Path $base -Parent).TrimEnd('\') }
      foreach ($f in Get-ChildItem $dir -Recurse -File) {
        $name = $f.FullName.Substring($base.Length + 1).Replace('\', '/')
        [void][IO.Compression.ZipFileExtensions]::CreateEntryFromFile($za, $f.FullName, $name,
          [IO.Compression.CompressionLevel]::Optimal)
      }
    }
  } finally { $za.Dispose(); $fs.Dispose() }
}

# LaTeX source without comments (text after an unescaped %)
function Get-Uncommented($tex) {
  ((Get-Content $tex) | ForEach-Object { $_ -replace '(?<!\\)%.*$', '' }) -join "`n"
}

# document class and bibliography style named in main.tex (packaged so that the zips compile standalone)
$mainSource = Get-Uncommented (Join-Path $paper 'main.tex')
$clsName = [regex]::Match($mainSource, '\\documentclass\s*(\[[^\]]*\])?\s*\{([^}]+)\}').Groups[2].Value.Trim()
$bstName = [regex]::Match($mainSource, '\\bibliographystyle\s*\{([^}]+)\}').Groups[1].Value.Trim()
if (-not $clsName) { throw 'no \documentclass in paper\main.tex' }
$styleFiles = @()
foreach ($f in @("$clsName.cls") + @(if ($bstName) { "$bstName.bst" })) {
  $p = & kpsewhich $f
  if ($p) { $styleFiles += $p } else { Write-Warning "kpsewhich could not find $f; it is not packaged" }
}
$styleNames = ($styleFiles | ForEach-Object { Split-Path $_ -Leaf }) -join ', '
if (-not $styleNames) { $styleNames = '(none packaged)' }

# figures: targets of the \includegraphics commands that are not commented out, as paths relative to paper\
function Get-Figures($tex) {
  $names = [regex]::Matches((Get-Uncommented $tex), '\\includegraphics\s*(\[[^\]]*\])?\s*\{([^}]*)\}') |
    ForEach-Object { $_.Groups[2].Value.Trim() } | Sort-Object -Unique
  foreach ($n in $names) {
    $file = $n
    if (-not [IO.Path]::GetExtension($file)) {  # extension left to LaTeX
      $ext = '.pdf', '.png', '.jpg', '.jpeg', '.eps' | Where-Object { Test-Path (Join-Path $paper ($n + $_)) } |
        Select-Object -First 1
      if (-not $ext) { throw "figure '$n' of $(Split-Path $tex -Leaf) not found" }
      $file = $n + $ext
    }
    $full = [IO.Path]::GetFullPath((Join-Path $paper $file))
    if (-not $full.StartsWith($paper + '\', [StringComparison]::OrdinalIgnoreCase)) {
      throw "figure '$n' of $(Split-Path $tex -Leaf) lies outside paper\"
    }
    $rel = $full.Substring($paper.Length + 1)
    $bad = @((Split-Path $rel -Parent) -split '\\' | Where-Object { $forbidden -contains $_.ToLower() })
    if ($bad) { throw "figure '$n' of $(Split-Path $tex -Leaf) is in paper\$(Split-Path $rel -Parent), which must not be uploaded" }
    if (-not (Test-Path $full -PathType Leaf)) { throw "figure '$n' of $(Split-Path $tex -Leaf) not found" }
    $rel
  }
}
$mainFigs = @(Get-Figures (Join-Path $paper 'main.tex'))
$suppFigs = @(Get-Figures (Join-Path $paper 'appendix.tex'))

function Copy-Figure($rel, $destRoot) {
  $dst = Join-Path $destRoot $rel
  New-Item -ItemType Directory -Force (Split-Path $dst -Parent) | Out-Null
  Copy-Item (Join-Path $paper $rel) $dst
}

# 1. archive (or clear) the previous package
New-Item -ItemType Directory -Force $sub | Out-Null
$previous = Get-ChildItem $sub | Where-Object { $_.Name -ne 'old' }
if ($previous) {
  if ($NoArchive) {
    $previous | Where-Object { $managed -contains $_.Name -or $_.Extension -in '.pdf', '.zip', '.txt', '.docx' } |
      Remove-Item -Recurse -Force -Confirm:$false
  } else {
    $arch = Join-Path $sub ('old\' + (Get-Date -Format 'yyyy-MM-dd_HHmmss'))
    New-Item -ItemType Directory -Force $arch | Out-Null
    $previous | Move-Item -Destination $arch
    Write-Host "previous package moved to $arch"
  }
}

# 2. Main Manuscript
$m = Join-Path $sub 'MainManuscript'
New-Item -ItemType Directory -Force (Join-Path $m 'figs'), (Join-Path $m 'build') | Out-Null
Copy-Item (Join-Path $paper 'main.tex'), (Join-Path $paper 'ref.bib'), (Join-Path $paper 'main.pdf') $m
foreach ($f in $styleFiles) { Copy-Item $f $m }
Copy-Item (Join-Path $paper 'build\main.bbl') (Join-Path $m 'main.bbl')
Copy-Item (Join-Path $paper 'build\appendix_only.aux') (Join-Path $m 'build\appendix_only.aux')
foreach ($f in $mainFigs) { Copy-Figure $f $m }

# 3. LaTeX Supplementary File
$s = Join-Path $sub 'LaTeXSupplementaryFile'
New-Item -ItemType Directory -Force (Join-Path $s 'figs'), (Join-Path $s 'build') | Out-Null
(Get-Content (Join-Path $paper 'build\appendix_only.tex') -Raw).Replace('\input{../appendix}', '\input{appendix}') |
  Out-File -Encoding ascii (Join-Path $s 'appendix_only.tex')
Copy-Item (Join-Path $paper 'appendix.tex'), (Join-Path $paper 'ref.bib') $s
foreach ($f in $styleFiles) { Copy-Item $f $s }
Copy-Item (Join-Path $paper 'build\appendix_only.bbl') (Join-Path $s 'appendix_only.bbl')
Copy-Item (Join-Path $paper 'appendix.pdf') (Join-Path $s 'appendix.pdf')
Copy-Item (Join-Path $paper 'build\main.aux') (Join-Path $s 'build\main.aux')
foreach ($f in $suppFigs) { Copy-Figure $f $s }

# 4. Images
$i = Join-Path $sub 'Image'
New-Item -ItemType Directory -Force $i | Out-Null
foreach ($f in ($mainFigs + $suppFigs | Sort-Object -Unique)) { Copy-Item (Join-Path $paper $f) $i }

# 5. Supplementary Material for Review
$r = Join-Path $sub 'SupplementaryMaterialForReview'
New-Item -ItemType Directory -Force $r | Out-Null
Copy-Item (Join-Path $paper 'appendix.pdf') (Join-Path $r 'Supplementary_Material.pdf')
if ($Artifact) { $artifact = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Artifact) }
else { $artifact = Join-Path $Root 'codeocean\code' }
if (Test-Path (Join-Path $artifact 'README.md')) {
  $c = Join-Path $sub 'SourceCode'
  New-Item -ItemType Directory -Force $c | Out-Null
  Copy-Item (Join-Path $artifact '*') $c -Recurse
  New-PortableZip $c (Join-Path $r 'Code_and_Results.zip')
  New-PortableZip $c (Join-Path $sub 'SourceCode.zip') -AsFolder
} else { Write-Warning 'codeocean\code not built: run python src\revloop\capsule_build.py first; no source code packaged' }

# 6. top-level PDFs, cover letter, title page
Copy-Item (Join-Path $paper 'main.pdf'), (Join-Path $paper 'appendix.pdf') $sub
foreach ($f in 'cover_letter.txt', 'Title_Page.docx', 'Highlights.txt', 'Author_Biographies.txt') {
  $src = Join-Path $Root "docs\$f"
  if (Test-Path $src) { Copy-Item $src $sub } else { Write-Warning "docs\$f not found" }
}

# author photographs for the separate biography upload (never part of the LaTeX sources)
$photos = Join-Path $paper 'figs\profile'
if (Test-Path $photos) {
  New-Item -ItemType Directory -Force (Join-Path $sub 'Author_Photos') | Out-Null
  Copy-Item (Join-Path $photos '*.jpg') (Join-Path $sub 'Author_Photos')
} else { Write-Warning 'paper\figs\profile not found: no author photographs packaged' }

# 7. READMEs
@"
MAIN MANUSCRIPT

main.pdf            compiled manuscript
main.tex, ref.bib   LaTeX source and bibliography; main.bbl is the prebuilt bibliography
$styleNames
                    document class and bibliography style
figs/               figures used by the main paper
build/appendix_only.aux
                    labels of the Supplementary Material, so that references to
                    supplementary sections, tables and figures (S1, Table S3, ...)
                    resolve when the main paper is compiled on its own

Compile: pdflatex main; bibtex main; pdflatex main; pdflatex main
"@ | Out-File -Encoding ascii (Join-Path $m 'README.txt')
@"
LATEX SUPPLEMENTARY FILE - Supplementary Material

appendix_only.tex   standalone wrapper (document class of the main paper, S-numbered
                    sections, tables, figures and equations)
appendix.tex        body of the Supplementary Material
appendix_only.bbl   prebuilt bibliography; ref.bib, $styleNames
appendix.pdf        compiled Supplementary Material
figs/               figures used by the Supplementary Material
build/main.aux      labels of the main paper, so that references to main-paper
                    sections, tables and equations resolve

Compile: pdflatex appendix_only; bibtex appendix_only; pdflatex appendix_only; pdflatex appendix_only
"@ | Out-File -Encoding ascii (Join-Path $s 'README.txt')

@"
UPLOAD MAP (Journal of Parallel and Distributed Computing, single-anonymized review)

Manuscript                             main.pdf; LaTeX_Source.zip if the system asks for source
Highlights (separate editable file)    Highlights.txt
Cover letter                           cover_letter.txt
Title page, if requested               Title_Page.docx
Author biographies and photographs     Author_Biographies.txt; Author_Photos\ (one jpg per author)
Supplementary material                 SupplementaryMaterialForReview\Supplementary_Material.pdf
Code and result files                  SupplementaryMaterialForReview\Code_and_Results.zip
                                       (the same files as SourceCode\ and SourceCode.zip)
Figures, if requested separately       Image.zip

Declarations (competing interests, funding, generative AI use, data availability) are at the
end of main.pdf; copy them into the submission form if it asks. Check the form against the
journal's current Guide for Authors before upload.

Never upload anything from old\ or the whole paper folder.
"@ | Out-File -Encoding ascii (Join-Path $sub 'UPLOAD.txt')

# 8. figure check: every packaged figure is one of the included ones; nothing from a forbidden folder
$included = @($mainFigs + $suppFigs | ForEach-Object { Split-Path $_ -Leaf })
foreach ($d in 'MainManuscript\figs', 'LaTeXSupplementaryFile\figs', 'Image') {
  foreach ($f in Get-ChildItem (Join-Path $sub $d) -Recurse -File) {
    if ($included -notcontains $f.Name) { throw "$d contains $($f.Name), which main.tex and appendix.tex do not include" }
  }
}
foreach ($f in Get-ChildItem $managed.ForEach({ Join-Path $sub $_ }) -Recurse -File) {
  $parts = $f.FullName.Substring($sub.Length + 1) -split '\\'
  if ($parts | Where-Object { $forbidden -contains $_.ToLower() }) { throw "packaged file from a forbidden folder: $($f.FullName)" }
}

# 9. zips for upload: both LaTeX folders in one archive (top-level folders kept), the figures in another
New-PortableZip @((Join-Path $sub 'MainManuscript'), (Join-Path $sub 'LaTeXSupplementaryFile')) `
  (Join-Path $sub 'LaTeX_Source.zip') -AsFolder
New-PortableZip (Join-Path $sub 'Image') (Join-Path $sub 'Image.zip')

# 10. checks: freshness and local traces (repository paths, host names, earlier venues)
$pairs = @(('main.tex', 'MainManuscript\main.tex'), ('appendix.tex', 'LaTeXSupplementaryFile\appendix.tex'),
           ('ref.bib', 'MainManuscript\ref.bib'), ('main.pdf', 'main.pdf'),
           ('appendix.pdf', 'SupplementaryMaterialForReview\Supplementary_Material.pdf'))
foreach ($p in $pairs) {
  if ((Get-FileHash (Join-Path $paper $p[0])).Hash -ne (Get-FileHash (Join-Path $sub $p[1])).Hash) { Write-Warning "$($p[1]) differs from paper\$($p[0])" }
}
$hits = Get-ChildItem $managed[0..1].ForEach({ Join-Path $sub $_ }) -Recurse -File -Include *.tex, *.txt |
  Select-String -Pattern 'bobsteam|autodl|/root/|C:\\source|\\Users\\|acmart|vgpu|nza1|figs[\\/]profile|jpdc-r\d|taco'
foreach ($h in $hits) { Write-Warning ("local trace: {0}:{1}: {2}" -f $h.Path.Substring($sub.Length + 1), $h.LineNumber, $h.Line.Trim()) }

# 11. optional standalone compile of the zips
if ($Verify) {
  $tmp = Join-Path ([IO.Path]::GetTempPath()) ('submit_verify_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
  Expand-Archive (Join-Path $sub 'LaTeX_Source.zip') $tmp
  foreach ($job in @(@('MainManuscript', 'main'), @('LaTeXSupplementaryFile', 'appendix_only'))) {
    Push-Location (Join-Path $tmp $job[0])
    pdflatex -interaction=nonstopmode "$($job[1]).tex" | Out-Null
    bibtex $job[1] | Out-Null
    pdflatex -interaction=nonstopmode "$($job[1]).tex" | Out-Null
    pdflatex -interaction=nonstopmode "$($job[1]).tex" | Out-Null
    $log = Get-Content "$($job[1]).log" -Raw
    Pop-Location
    $pages = [regex]::Match($log, '\((\d+)\s+pages').Groups[1].Value
    $undef = ([regex]::Matches($log, 'Warning: (Reference|Citation)[^\n]*undefined')).Count
    $errs = ([regex]::Matches($log, '(?m)^! ')).Count
    Write-Host ("verify {0}: pages={1} undefined={2} errors={3} ({4})" -f $job[1], $pages, $undef, $errs, $tmp)
  }
}

Get-ChildItem $sub -Recurse -File | Where-Object { $_.FullName -notlike "$sub\old\*" } |
  Select-Object @{ n = 'File'; e = { $_.FullName.Substring($sub.Length + 1) } }, Length | Format-Table -AutoSize
