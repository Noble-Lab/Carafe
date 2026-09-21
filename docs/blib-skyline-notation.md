# Carafe `.blib` writer — Skyline modified-sequence notation

**Status:** RESOLVED 2026-07-11 (branch `blib-decoy-pairs`). The signed-delta-mass fix and the
`DecoyPairs` table below are both implemented and unit-tested. See "Resolution" at the end.
**Found:** 2026-07-11, while building the Osprey PRM re-scoring tool
(`maccoss/skyline-osprey-tool`) against a Carafe-generated `.blib`.

## Summary

Carafe writes modified residues into the `.blib` `RefSpectra.peptideModSeq` column as an **unsigned,
full-precision** delta mass, e.g. Carbamidomethyl-Cys comes out as:

```
C[57.02146372057]LAVYQAGAR
```

Skyline's convention for a modified sequence is a **signed** (and usually rounded) delta mass:

```
C[+57]LAVYQAGAR          # Skyline chromatogram/report export
C[+57.0]LAVYQAGAR        # other Skyline contexts
```

The missing `+` sign is the crux (the full precision is a secondary nit).

## Location

`src/main/java/ai/AIGear.java` — method `format_skyline_residue(char residue, BigDecimal deltaMass)`
(around line 13756), used on the "Skyline (.blib) library generation" path.

```java
private String format_skyline_residue(char residue, BigDecimal deltaMass) {
    return residue + "[" + deltaMass.stripTrailingZeros().toPlainString() + "]";
}
```

`stripTrailingZeros().toPlainString()` yields `57.02146372057` — unsigned, unrounded.

## Impact

- **Skyline itself is fine.** It matches library entries to document peptides by computed *mass*, not by
  the `peptideModSeq` string, so the library loads and matches correctly (the test document's 314 targets
  all matched, `count_measured=1`).
- **Downstream string joins break.** Any tool that joins the blib's `peptideModSeq` against a Skyline
  `PeptideModifiedSequence` export (e.g. joining predicted RT / decoy pairing back to exported
  chromatograms) gets a **silent miss on every modified peptide**, because `C[57.02146372057]` != `C[+57]`
  as strings. In the Osprey re-scoring tool this dropped all 50 Cysteine-containing precursors from the
  predicted-RT lookup until a mass-rounding normalizer was added on the consumer side. Emitting the signed
  form would let those joins work natively (and match how Skyline round-trips its own libraries).

## Suggested fix

Add the sign (minimal fix):

```java
private String format_skyline_residue(char residue, BigDecimal deltaMass) {
    String sign = deltaMass.signum() >= 0 ? "+" : "";
    return residue + "[" + sign + deltaMass.stripTrailingZeros().toPlainString() + "]";
}
```

Optionally also round to Skyline's precision (the sign is the essential part; Skyline reads full
precision without complaint):

```java
return residue + "[" + String.format("%+.4f", deltaMass.doubleValue()) + "]";   // e.g. C[+57.0215]
```

Verify against the format Skyline emits for the same fixed mod (Carbamidomethyl → `C[+57]` in
chromatogram/report exports) and pick whichever rounding matches your target Skyline version.

## Related, non-blocking observations

- `RefSpectra.score` / `scoreType` are all `0` and the `ScoreTypes` table is empty. Fine for a purely
  predicted library — flagged only in case a real score is intended.
- **Target/decoy pairing** lives only in the side-car `osprey_library_db_pairing.tsv`, not the `.blib`.
  For a self-contained library, see the proposed `DecoyPairs` table spec
  (`skyline-osprey-tool/docs/blib-decoy-pairing-spec.md`); `PairingManifestReconciler.java` already holds
  the pairing needed to populate it. **(Now implemented — see Resolution.)**

## Resolution (2026-07-11)

1. **Signed delta mass.** `format_skyline_residue` now prefixes a sign, e.g. `C[+57.02146372057]`
   (`+` for >= 0, `-` otherwise). Full precision is kept — Skyline matches library entries by computed
   mass and reads full precision fine, and the signed full-precision form matches how Skyline round-trips
   its own libraries; a consumer joining against a *rounded* report export still normalizes by mass.
   Covered by `ConvertModificationTest` (`testSkylineResidueDeltaMassIsSigned` + the acetyl/oxidation
   cases now assert the `+` form).

2. **`DecoyPairs` table.** The Skyline `.blib` now carries the additive, Skyline-neutral `DecoyPairs`
   table (`RefSpectraID PRIMARY KEY, IsDecoy, IsEntrapment, PairID, Method` + a `PairID` index), written
   in the same transaction as the library:
   - `main.java.db.DecoyPairPlanner` joins the pairing manifest to the `RefSpectra` rows actually written
     (by I&rarr;L-normalized stripped sequence), pairing a target precursor with its decoy at the same
     charge and modification set; only pairs whose target **and** decoy are present are emitted.
     `IsDecoy` comes from the manifest `decoy` column; `Method` is `reverse`/`cycle` inferred from
     `EntrapmentFastaGear.reversePreservingCterm`. Unit-tested in `DecoyPairPlannerTest` and `SkylineIOTest`.
   - **Entrapment.** Osprey scores an entrapment target (`p_target`) as a target and its entrapment
     decoy (`p_decoy`) as a decoy, and carries the pairing forward — so every target, real or
     entrapment, has a paired decoy. `DecoyPairs` reflects this: it pairs both `target`&harr;`decoy` and
     `p_target`&harr;`p_decoy`, and the `IsEntrapment` column (orthogonal to `IsDecoy`) flags the
     entrapment peptides so follow-on tools can score performance against the entrapment set.
   - `SkylineIO.create_DecoyPairs()` creates/inserts the table; `AIGear` populates it when a
     `-pairing_manifest` is supplied (the Osprey Workflow 4/5 Skyline path passes the prelim manifest).
     It is regenerated whenever Carafe writes the `.blib`; Skyline does not round-trip it (see the spec's
     "Minimize library" caveat).
