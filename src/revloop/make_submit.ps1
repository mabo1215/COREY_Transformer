# Assemble the TPDS submission package under submit\ from the current paper\ build.
#
#   powershell -ExecutionPolicy Bypass -File src\revloop\make_submit.ps1 [-Build] [-Verify] [-NoArchive]
#
#   -Build      run paper\build.bat first (otherwise the existing paper\build\ outputs are used)
#   -Verify     extract the two LaTeX zips into a temporary directory, compile them standalone and
#               report pages, undefined references and errors
#   -NoArchive  replace the package in place instead of moving it to submit\old\<timestamp>\
#
# Layout (one folder per upload file type, plus a zip of each LaTeX folder):
#   submit\main.pdf, appendix.pdf             compiled main paper and Supplementary Material
#   submit\MainManuscript\                    main.tex, ref.bib, main.bbl, IEEEtran.cls/.bst, figs\,
#                                             build\appendix_only.aux (resolves S-prefixed references)
#   submit\LaTeXSupplementaryFile\            appendix_only.tex (standalone wrapper), appendix.tex,
#                                             appendix_only.bbl, figs\, build\main.aux (M- references)
#   submit\Image\                             every figure file of the main paper and the supplement
#   submit\SupplementaryMaterialForReview\    Supplementary_Material.pdf
#   submit\cover_letter.txt, Title_Page.docx  from docs\ (they carry author names: editor-only uploads)
# submit\ is git-ignored. Earlier packages are kept under submit\old\.
param([switch]$Build, [switch]$Verify, [switch]$NoArchive)
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$paper = Join-Path $Root 'paper'
$sub = Join-Path $Root 'submit'
$managed = 'MainManuscript', 'LaTeXSupplementaryFile', 'Image', 'SupplementaryMaterialForReview'

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

$cls = & kpsewhich IEEEtran.cls
$bst = & kpsewhich IEEEtran.bst
if (-not $cls -or -not $bst) { throw 'kpsewhich could not find IEEEtran.cls/.bst' }

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

function Get-Figures($tex) {
  Select-String -Path $tex -Pattern 'includegraphics(\[[^\]]*\])?\{([^}]*)\}' -AllMatches |
    ForEach-Object { $_.Matches | ForEach-Object { $_.Groups[2].Value } } | Sort-Object -Unique
}
$mainFigs = @(Get-Figures (Join-Path $paper 'main.tex'))
$suppFigs = @(Get-Figures (Join-Path $paper 'appendix.tex'))

# 2. Main Manuscript
$m = Join-Path $sub 'MainManuscript'
New-Item -ItemType Directory -Force (Join-Path $m 'figs'), (Join-Path $m 'build') | Out-Null
Copy-Item (Join-Path $paper 'main.tex'), (Join-Path $paper 'ref.bib'), (Join-Path $paper 'main.pdf'), $cls, $bst $m
Copy-Item (Join-Path $paper 'build\main.bbl') (Join-Path $m 'main.bbl')
Copy-Item (Join-Path $paper 'build\appendix_only.aux') (Join-Path $m 'build\appendix_only.aux')
foreach ($f in $mainFigs) { Copy-Item (Join-Path $paper $f) (Join-Path $m $f) }

# 3. LaTeX Supplementary File
$s = Join-Path $sub 'LaTeXSupplementaryFile'
New-Item -ItemType Directory -Force (Join-Path $s 'figs'), (Join-Path $s 'build') | Out-Null
(Get-Content (Join-Path $paper 'build\appendix_only.tex') -Raw).Replace('\input{../appendix}', '\input{appendix}') |
  Out-File -Encoding ascii (Join-Path $s 'appendix_only.tex')
Copy-Item (Join-Path $paper 'appendix.tex'), (Join-Path $paper 'ref.bib'), $cls, $bst $s
Copy-Item (Join-Path $paper 'build\appendix_only.bbl') (Join-Path $s 'appendix_only.bbl')
Copy-Item (Join-Path $paper 'appendix.pdf') (Join-Path $s 'appendix.pdf')
Copy-Item (Join-Path $paper 'build\main.aux') (Join-Path $s 'build\main.aux')
foreach ($f in $suppFigs) { Copy-Item (Join-Path $paper $f) (Join-Path $s $f) }

# 4. Images
$i = Join-Path $sub 'Image'
New-Item -ItemType Directory -Force $i | Out-Null
foreach ($f in ($mainFigs + $suppFigs | Sort-Object -Unique)) { Copy-Item (Join-Path $paper $f) $i }

# 5. Supplementary Material for Review
$r = Join-Path $sub 'SupplementaryMaterialForReview'
New-Item -ItemType Directory -Force $r | Out-Null
Copy-Item (Join-Path $paper 'appendix.pdf') (Join-Path $r 'Supplementary_Material.pdf')

# 6. top-level PDFs, cover letter, title page
Copy-Item (Join-Path $paper 'main.pdf'), (Join-Path $paper 'appendix.pdf') $sub
foreach ($f in 'cover_letter.txt', 'Title_Page.docx') {
  $src = Join-Path $Root "docs\$f"
  if (Test-Path $src) { Copy-Item $src $sub } else { Write-Warning "docs\$f not found" }
}

# 7. READMEs
@'
MAIN MANUSCRIPT - IEEE TPDS regular paper (double-anonymous review)

main.pdf            compiled manuscript
main.tex, ref.bib   LaTeX source and bibliography; main.bbl is the prebuilt bibliography
IEEEtran.cls/.bst   IEEE class and BibTeX style
figs/               figures used by the main paper
build/appendix_only.aux
                    labels of the Supplementary Material, so that references to
                    supplementary sections, tables and figures (S1, Table S3, ...)
                    resolve when the main paper is compiled on its own

Compile: pdflatex main; bibtex main; pdflatex main; pdflatex main
'@ | Out-File -Encoding ascii (Join-Path $m 'README.txt')
@'
LATEX SUPPLEMENTARY FILE - Supplementary Material (double-anonymous review)

appendix_only.tex   standalone wrapper (IEEEtran journal class, S-numbered sections,
                    tables, figures and equations)
appendix.tex        body of the Supplementary Material
appendix_only.bbl   prebuilt bibliography; ref.bib, IEEEtran.cls/.bst
appendix.pdf        compiled Supplementary Material
figs/               figures used by the Supplementary Material
build/main.aux      labels of the main paper, so that references to main-paper
                    sections, tables and equations resolve

Compile: pdflatex appendix_only; bibtex appendix_only; pdflatex appendix_only; pdflatex appendix_only
'@ | Out-File -Encoding ascii (Join-Path $s 'README.txt')

# 8. zips for upload
foreach ($d in 'MainManuscript', 'LaTeXSupplementaryFile', 'Image') {
  Compress-Archive -Path (Join-Path $sub "$d\*") -DestinationPath (Join-Path $sub "$d.zip") -Force
}

# 9. checks: freshness, identifying strings, PDF info dictionary
$pairs = @(('paper\main.tex', 'submit\MainManuscript\main.tex'), ('paper\appendix.tex', 'submit\LaTeXSupplementaryFile\appendix.tex'),
           ('paper\ref.bib', 'submit\MainManuscript\ref.bib'), ('paper\main.pdf', 'submit\main.pdf'),
           ('paper\appendix.pdf', 'submit\SupplementaryMaterialForReview\Supplementary_Material.pdf'))
foreach ($p in $pairs) {
  if ((Get-FileHash (Join-Path $Root $p[0])).Hash -ne (Get-FileHash (Join-Path $Root $p[1])).Hash) { Write-Warning "$($p[1]) differs from $($p[0])" }
}
$hits = Get-ChildItem $managed.ForEach({ Join-Path $sub $_ }) -Recurse -File -Include *.tex, *.bib, *.bbl, *.aux, *.txt |
  Select-String -Pattern 'corey|mabo|bobsteam|autodl|/root/|C:\\source|\\Users\\|codeocean'
foreach ($h in $hits) { Write-Warning ("identifying text: {0}:{1}: {2}" -f $h.Path.Substring($sub.Length + 1), $h.LineNumber, $h.Line.Trim()) }
foreach ($f in 'main.pdf', 'appendix.pdf') {
  $bytes = [IO.File]::ReadAllText((Join-Path $sub $f), [Text.Encoding]::GetEncoding(28591))
  foreach ($k in '/Author', '/Title', '/CreationDate') { if ($bytes.Contains($k)) { Write-Warning "$f carries $k in its info dictionary" } }
}

# 10. optional standalone compile of the zips
if ($Verify) {
  $tmp = Join-Path ([IO.Path]::GetTempPath()) ('submit_verify_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
  Expand-Archive (Join-Path $sub 'MainManuscript.zip') (Join-Path $tmp 'main')
  Expand-Archive (Join-Path $sub 'LaTeXSupplementaryFile.zip') (Join-Path $tmp 'supp')
  foreach ($job in @(@('main', 'main'), @('supp', 'appendix_only'))) {
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
