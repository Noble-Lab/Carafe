<#
.SYNOPSIS
    Repoints the compiled-in Osprey defaults that the Carafe installer ships, without
    forking the upstream pwiz tree.

.DESCRIPTION
    Osprey reads its tuning knobs from OSPREY_* environment variables at process start
    (Osprey.Core/OspreyEnvironment.cs). A shipped MSI is run by collaborators who never
    set those variables, so the compiled UNSET default is what actually takes effect.
    Upstream (ProteoWizard/pwiz#4446) deliberately landed the new pass-2 modes and the
    learned peak pick "off by default", because flipping either one is a coordinated
    re-baseline of pwiz's committed regression golden. Carafe wants them ON. This script
    rewrites exactly those two defaults in the checked-out source before dotnet publish.

    Patch 1 - 2nd-pass q-value: rewrite ONLY the IsNullOrWhiteSpace(raw) (unset) branch
    of NormalizePass2QValue from "percolator" (retrain the 2nd-pass SVM) to
    "protein-compact" (frozen 1st-pass weights + protein-stratum competition, no
    retrain). The separate unrecognized-token fallback is left as "percolator" so an
    explicit OSPREY_PASS2_QVALUE=percolator still resolves to percolator (it falls
    through to that fallback) and typos still warn.

    Patch 2 - peak pick: rewrite PickLda from IsSetAndNotZero (opt-IN, i.e. the pure
    product-form legacy pick unless requested) to IsNotZero (default ON, still
    overridable with OSPREY_PICK_LDA=0). IsNotZero is upstream's own idiom for a
    default-on flag -- UseFdrProjection uses it -- so this reuses an existing helper
    rather than introducing new logic. Before pwiz#4446 the learned pick WAS the
    default, so this keeps the MSI's behavior continuous with every installer shipped
    so far instead of silently reverting collaborators to the legacy pick.

    Tripwire 3 - frozen 2nd pass: assert OSPREY_PROTEIN_COMPACT_RETRAIN is still
    opt-in (Pass2ProteinCompactRetrain = IsSetAndNotZero(...)). That variable is a
    diagnostic A/B switch that makes protein-compact RETRAIN the 2nd-pass Percolator
    over the compacted pool instead of transferring the frozen 1st-pass model. Nothing
    in the shipped binary sets it, so the MSI gets frozen scores -- but if a future
    upstream commit makes retrain the default, patch 1 would silently start shipping a
    retrained 2nd pass. Fail loudly here instead.

    Every match is asserted to occur exactly once. Any mismatch (upstream refactor,
    already-patched source, changed default) exits non-zero so the installer build
    fails visibly rather than silently shipping the wrong behavior. The regexes are
    whitespace/newline flexible so they work on either LF or CRLF checkouts.

.PARAMETER OspreyEnvPath
    Path to Osprey.Core/OspreyEnvironment.cs in the checked-out pwiz tree.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$OspreyEnvPath
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $OspreyEnvPath)) {
    Write-Error "OspreyEnvironment.cs not found at '$OspreyEnvPath'. Osprey source layout changed?"
    exit 1
}

$src = [System.IO.File]::ReadAllText($OspreyEnvPath)

# --- Tripwire: the shipped binary must keep FROZEN 1st-pass weights in the 2nd pass. ---
# OSPREY_PROTEIN_COMPACT_RETRAIN stays opt-in, so unset -> no retrain.
$retrainPattern = 'Pass2ProteinCompactRetrain\s*=\s*IsSetAndNotZero\(\s*@?"OSPREY_PROTEIN_COMPACT_RETRAIN"\s*\)'
if ([regex]::Matches($src, $retrainPattern).Count -ne 1) {
    Write-Error ("OSPREY_PROTEIN_COMPACT_RETRAIN default guard failed: expected exactly one " +
        "'Pass2ProteinCompactRetrain = IsSetAndNotZero(""OSPREY_PROTEIN_COMPACT_RETRAIN"")'. " +
        "The 2nd-pass frozen-vs-retrain default may have changed; refusing to build an installer " +
        "that could ship a retrained 2nd-pass Percolator.")
    exit 1
}

# --- Patch 1: 2nd-pass q-value unset default -> protein-compact. ---
# Capture the 'if (...)' + whitespace + 'return ' prefix so its formatting/newline is
# preserved; only the returned constant is swapped. \s* spans the line break on LF or CRLF.
# Idempotent: upstream may already default to protein-compact (pwiz removed the percolator
# token outright and made protein-compact the unset default). Treat "already correct" as
# success rather than as a changed-upstream error - the point of this step is that the SHIPPED
# default is protein-compact, not that this script was the one to set it. Only a value that is
# neither the old form nor the wanted one is a real "re-verify before building".
$unsetPrefix = 'if\s*\(\s*string\.IsNullOrWhiteSpace\(\s*raw\s*\)\s*\)\s*return\s+'
$unsetDefault = "($unsetPrefix)PASS2_QVALUE_PERCOLATOR;"
$alreadyCompact = "($unsetPrefix)PASS2_QVALUE_PROTEIN_COMPACT;"
$pass2Hits = [regex]::Matches($src, $unsetDefault)
$compactHits = [regex]::Matches($src, $alreadyCompact)
if ($pass2Hits.Count -eq 1) {
    $src = [regex]::Replace($src, $unsetDefault, '${1}PASS2_QVALUE_PROTEIN_COMPACT;')
    $pass2Action = "patched: percolator -> protein-compact"
} elseif ($compactHits.Count -eq 1) {
    $pass2Action = "already protein-compact upstream, left unchanged"
} else {
    Write-Error ("Could not establish the unset (IsNullOrWhiteSpace) 2nd-pass q-value default in " +
        "NormalizePass2QValue: found $($pass2Hits.Count) 'return PASS2_QVALUE_PERCOLATOR;' and " +
        "$($compactHits.Count) 'return PASS2_QVALUE_PROTEIN_COMPACT;'. OspreyEnvironment.cs changed " +
        "upstream in a way this script does not recognize; re-verify the 2nd-pass q-value default " +
        "before building.")
    exit 1
}

# --- Patch 2: learned peak pick opt-in -> default on (OSPREY_PICK_LDA=0 still disables). ---
# Idempotent for the same reason as patch 1: upstream now ships IsNotZero (default ON).
$pickLda = '(PickLda\s*=\s*)IsSetAndNotZero(\(\s*@?"OSPREY_PICK_LDA"\s*\))'
$pickAlreadyOn = 'PickLda\s*=\s*IsNotZero\(\s*@?"OSPREY_PICK_LDA"\s*\)'
$pickHits = [regex]::Matches($src, $pickLda)
$pickOnHits = [regex]::Matches($src, $pickAlreadyOn)
if ($pickHits.Count -eq 1) {
    $src = [regex]::Replace($src, $pickLda, '${1}IsNotZero${2}')
    $pickAction = "patched: opt-in -> ON"
} elseif ($pickOnHits.Count -eq 1) {
    $pickAction = "already ON upstream, left unchanged"
} else {
    Write-Error ("Could not establish the OSPREY_PICK_LDA default: found $($pickHits.Count) " +
        "'PickLda = IsSetAndNotZero(...)' and $($pickOnHits.Count) 'PickLda = IsNotZero(...)'. " +
        "The peak-pick flag changed upstream (it was OSPREY_PICK_LEGACY before pwiz#4446); " +
        "re-verify which pick the installer should ship before building.")
    exit 1
}

# Write UTF-8 without BOM; the regex replaces preserved the original newlines.
[System.IO.File]::WriteAllText($OspreyEnvPath, $src, (New-Object System.Text.UTF8Encoding($false)))

Write-Host "OSPREY_PASS2_QVALUE unset default -> protein-compact ($pass2Action)"
Write-Host "  (frozen 1st-pass weights + protein-stratum competition, no retrain)"
Write-Host "OSPREY_PICK_LDA default -> ON, learned resolution-keyed pick ($pickAction)"
Write-Host "  (set OSPREY_PICK_LDA=0 at run time to fall back to the legacy product pick)"
Write-Host "Verified OSPREY_PROTEIN_COMPACT_RETRAIN still opt-in (2nd pass stays frozen)"
