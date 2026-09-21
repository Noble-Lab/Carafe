package test.java.db;

import main.java.db.DecoySimilarityGate;
import main.java.db.EntrapmentFastaGear;
import main.java.input.CParameter;
import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Coverage for {@link DecoySimilarityGate} and the generation paths that consume it.
 *
 * <p>The gate tests are ported from Osprey's Rust and C# suites so all three implementations
 * assert the same behaviour on the same fixtures.</p>
 *
 * <p>TestNG style (argument order {@code assertEquals(actual, expected, message)}) to match the
 * project's other tests.</p>
 */
public class DecoySimilarityGateTest {

    /** Every rung of this pair coincides: the reversal is isobaric because L and I have the same
     *  residue mass, so the overlap ratio is 1.0 against a 0.4 budget. */
    private static final String ISOBARIC_TARGET = "AILLAK";
    private static final String ISOBARIC_REVERSAL = "ALLIAK";

    /** The case from the field report: 17 alanines leave almost nothing to permute into. */
    private static final String POLY_ALANINE = "AAAAAAAAAAAAAAAAGATCLER";

    private int savedEnzyme, savedMissed, savedMinLen, savedMaxLen;
    private boolean savedClipM;

    @BeforeMethod
    public void saveCParameter() {
        savedEnzyme = CParameter.enzyme;
        savedMissed = CParameter.maxMissedCleavages;
        savedMinLen = CParameter.minPeptideLength;
        savedMaxLen = CParameter.maxPeptideLength;
        savedClipM = CParameter.clip_nTerm_M;
    }

    @AfterMethod
    public void restoreCParameter() {
        CParameter.enzyme = savedEnzyme;
        CParameter.maxMissedCleavages = savedMissed;
        CParameter.minPeptideLength = savedMinLen;
        CParameter.maxPeptideLength = savedMaxLen;
        CParameter.clip_nTerm_M = savedClipM;
    }

    @Test
    public void testOverlapGateRejectsAnIsobaricNearDuplicate() {
        Assert.assertFalse(
                DecoySimilarityGate.isCandidateAcceptable(ISOBARIC_TARGET, ISOBARIC_REVERSAL),
                "an isobaric reversal shares every rung and must be rejected");

        // Without this half the test would pass equally well with a gate that rejected
        // everything, which would silently drop every peptide from the library.
        Assert.assertTrue(
                DecoySimilarityGate.isCandidateAcceptable("PEPTIDEK", "EDITPEPK"),
                "an ordinary tryptic reversal must be accepted");
    }

    @Test
    public void testFragmentOverlapIsReportedNotJustThresholded() {
        Assert.assertEquals(
                DecoySimilarityGate.fragmentOverlap(ISOBARIC_TARGET, ISOBARIC_REVERSAL), 1.0, 1e-9,
                "isobaric reversal should overlap completely");
        Assert.assertTrue(
                DecoySimilarityGate.fragmentOverlap("PEPTIDEK", "EDITPEPK")
                        <= DecoySimilarityGate.MAX_FRAGMENT_OVERLAP,
                "ordinary reversal should sit under the threshold");
    }

    @Test
    public void testLadderSkipsUnknownResiduesWithoutEmptyingItself() {
        // U (selenocysteine) is absent from the residue table and does occur in UniProt-derived
        // libraries. Only ions actually spanning it may be dropped; a leading unknown must not
        // empty the ladder, because an empty ladder reads as "accept".
        double[] ladder = DecoySimilarityGate.theoreticalLadder("UPEPTIDEK");
        Assert.assertTrue(ladder.length > 0,
                "a leading unknown residue must not empty the ladder");
    }

    @Test
    public void testGenerationNeverEmitsARejectedDecoy() {
        Set<String> noTargets = Collections.emptySet();
        String decoy = EntrapmentFastaGear.generateReverseDecoy(ISOBARIC_TARGET, noTargets);
        Assert.assertNotEquals(decoy, ISOBARIC_REVERSAL,
                "the plain reversal fails the gate and must not be emitted");
        if (decoy != null) {
            Assert.assertTrue(DecoySimilarityGate.isCandidateAcceptable(ISOBARIC_TARGET, decoy),
                    "any decoy that IS emitted must pass the gate: " + decoy);
        }
    }

    @Test
    public void testGenerationNeverEmitsARejectedEntrapment() {
        // The invariant that matters: whatever comes back is acceptable, or nothing comes back.
        // Asserted over sequences chosen to stress the gate - low complexity, isobaric residues,
        // and a plain tryptic peptide as the control that must still succeed.
        List<String> hard = new ArrayList<>();
        hard.add(POLY_ALANINE);
        hard.add(ISOBARIC_TARGET);
        hard.add("GGGGGGGGGGGGK");
        hard.add("LLLLIIIILLLLK");
        hard.add("SAMPLERPEPTIDEK");

        int generated = 0;
        for (String seq : hard) {
            String entrapment =
                    EntrapmentFastaGear.generateShuffledEntrapment(seq, 42L, new HashSet<>());
            if (entrapment == null) {
                continue;
            }
            generated++;
            Assert.assertNotEquals(entrapment, seq, "entrapment must differ from its target");
            Assert.assertTrue(DecoySimilarityGate.isCandidateAcceptable(seq, entrapment),
                    "emitted entrapment must pass the gate: " + seq + " -> " + entrapment);
        }
        Assert.assertTrue(generated > 0,
                "the gate must not reject every candidate for every sequence");
    }

    @Test
    public void testEntrapmentRespectsExistingTargets() {
        // A candidate that collides with a real target is not entrapment at all.
        Set<String> targets = new HashSet<>();
        String first = EntrapmentFastaGear.generateShuffledEntrapment(
                "SAMPLERPEPTIDEK", 42L, new HashSet<>());
        Assert.assertNotNull(first, "control sequence should yield an entrapment");
        targets.add(first);
        String second = EntrapmentFastaGear.generateShuffledEntrapment(
                "SAMPLERPEPTIDEK", 42L, targets);
        Assert.assertNotEquals(second, first,
                "a candidate colliding with a known target must be retried past");
    }

    @Test
    public void testAttemptZeroKeepsTheOriginalSeedDerivation() {
        // Deliberate, and two things depend on it: -no_similarity_gate reproduces a pre-gate
        // library byte for byte (it runs exactly one attempt, so attempt 0's derivation IS the
        // reproduction), and a before/after FDP comparison stays nearly paired because only the
        // ~4% of entrapment the gate rejects changes. The first is a permanent regression oracle,
        // not a migration convenience. See EntrapmentFastaGear.derivePepSeed - if this assertion
        // is ever changed on purpose, both properties go with it.
        String withoutAttempt = EntrapmentFastaGear.shufflePreservingCterm("SAMPLERPEPTIDEK", 42L);
        String attemptZero = EntrapmentFastaGear.shufflePreservingCterm("SAMPLERPEPTIDEK", 42L, 0);
        Assert.assertEquals(attemptZero, withoutAttempt,
                "attempt 0 must reproduce the pre-retry shuffle exactly");

        String attemptOne = EntrapmentFastaGear.shufflePreservingCterm("SAMPLERPEPTIDEK", 42L, 1);
        Assert.assertNotEquals(attemptOne, attemptZero,
                "a retry must draw a different permutation");
    }

    @Test
    public void testShuffleIsDeterministicAcrossCalls() {
        for (int i = 0; i < 3; i++) {
            Assert.assertEquals(
                    EntrapmentFastaGear.generateShuffledEntrapment("LENGTHYPEPTIDEK", 42L, new HashSet<>()),
                    EntrapmentFastaGear.generateShuffledEntrapment("LENGTHYPEPTIDEK", 42L, new HashSet<>()),
                    "generation must be a pure function of (sequence, seed, target set)");
        }
    }

    // --- I/L-isobaric collision rejection -------------------------------------------------

    @Test
    public void testIlNormalizeFoldsIsoleucineOntoLeucine() {
        Assert.assertEquals(EntrapmentFastaGear.ilNormalize("PEPTIDEIK"), "PEPTLDELK");
        // No isoleucine means no change, which is what makes the normalised comparison a
        // superset of exact equality rather than a different test.
        Assert.assertEquals(EntrapmentFastaGear.ilNormalize("PEPTLDELK"), "PEPTLDELK");
        Assert.assertEquals(
                EntrapmentFastaGear.ilNormalize("LLLL"), EntrapmentFastaGear.ilNormalize("IIII"));
    }

    @Test
    public void testEntrapmentRejectsIlIsobaricMatchToADifferentTarget() {
        // The point of this gate, and why the fragment-overlap gate cannot do it: overlap compares
        // a candidate to its OWN paired target, while a collision is an exact isobaric match to a
        // DIFFERENT one. Here the shuffle of one target lands I/L-identical to an unrelated target.
        String target = "SAMPLERPEPTIDEK";
        Set<String> targets = new HashSet<>();
        targets.add(target);

        String unguarded =
                EntrapmentFastaGear.generateShuffledEntrapment(target, 42L, targets);
        Assert.assertNotNull(unguarded, "control: this target should yield an entrapment");

        // Now declare an unrelated target that is I/L-isobaric to that entrapment. Nothing about
        // the candidate's relationship to its OWN target has changed, so the overlap gate still
        // passes it - only the normalised collision index can reject it.
        Assert.assertTrue(DecoySimilarityGate.isCandidateAcceptable(target, unguarded),
                "the overlap gate does not object to this candidate");

        Set<String> withCollider = new HashSet<>(targets);
        withCollider.add(unguarded.replace('L', 'I'));
        String guarded =
                EntrapmentFastaGear.generateShuffledEntrapment(target, 42L, withCollider);
        Assert.assertNotEquals(guarded, unguarded,
                "an entrapment I/L-isobaric to any real target must be rejected");
        if (guarded != null) {
            Assert.assertFalse(
                    EntrapmentFastaGear.ilNormalizedSet(withCollider)
                            .contains(EntrapmentFastaGear.ilNormalize(guarded)),
                    "whatever is emitted must not collide under I/L normalisation: " + guarded);
        }
    }

    @Test
    public void testDecoyAlsoRejectsIlIsobaricCollision() {
        // A decoy indistinguishable from a real target is not a valid null either - it is detected
        // wherever its twin is, which inflates the decoy count rather than the entrapment count.
        String target = "PEPTLDEKPEPTLDEK";
        Set<String> targets = new HashSet<>();
        targets.add(target);
        String plain = EntrapmentFastaGear.generateReverseDecoy(target, targets);
        Assert.assertNotNull(plain, "control: this target should yield a decoy");

        Set<String> withCollider = new HashSet<>(targets);
        withCollider.add(plain.replace('L', 'I'));
        String guarded = EntrapmentFastaGear.generateReverseDecoy(target, withCollider);
        if (guarded != null) {
            Assert.assertFalse(
                    EntrapmentFastaGear.ilNormalizedSet(withCollider)
                            .contains(EntrapmentFastaGear.ilNormalize(guarded)),
                    "emitted decoy must not be I/L-isobaric to a target: " + guarded);
        }
    }

    /**
     * The audit switch must actually change generation, not merely be accepted. Asserted on the
     * one pair where the expected output is knowable in advance: {@code ALLIAK} is the plain
     * C-term-preserving reversal of {@code AILLAK} and overlaps its ladder completely (L/I are
     * isobaric), so the gate must reject it and the pre-fix path must emit it.
     *
     * <p>This is the assertion the byte-for-byte reproduction claim rests on. The 349 MB SHA-256
     * check that motivates {@code -no_similarity_gate} cannot live in a unit suite, so what is
     * pinned here is the property underneath it: with the gate off, generation is the unretried,
     * exact-collision-only behaviour it was before.</p>
     */
    @Test
    public void testAuditSwitchRestoresPreFixGeneration() {
        Set<String> noTargets = Collections.emptySet();
        Set<String> noTargetsIl = EntrapmentFastaGear.ilNormalizedSet(noTargets);

        String ungated = EntrapmentFastaGear.generateReverseDecoy(
                ISOBARIC_TARGET, noTargets, noTargetsIl, false);
        Assert.assertEquals(ungated, ISOBARIC_REVERSAL,
                "with the gate off, the pre-fix path must emit the plain reversal it always did");

        String gated = EntrapmentFastaGear.generateReverseDecoy(
                ISOBARIC_TARGET, noTargets, noTargetsIl, true);
        Assert.assertNotEquals(gated, ISOBARIC_REVERSAL,
                "with the gate on, that reversal must be rejected - otherwise the switch is inert");
    }

    @Test
    public void testAuditSwitchIsDeterministicAcrossRuns() throws IOException {
        // -no_similarity_gate exists to reproduce a pre-fix library byte for byte, which is the
        // regression oracle for "does this generator still match the delivered library?". It must
        // therefore skip I/L rejection too, not just the overlap gate - otherwise the oracle
        // silently stops reproducing and the check it exists for is lost.
        Path in = write(TARGET_FASTA, "il_target");
        List<String> gatedOff = null;
        for (boolean gate : new boolean[] { false, false }) {
            Path outFasta = Files.createTempFile("il_out", ".fasta");
            Path manifest = Files.createTempFile("il_manifest", ".tsv");
            EntrapmentFastaGear.Config cfg = foreignConfig(in, outFasta, manifest);
            cfg.similarityGate = gate;
            EntrapmentFastaGear.run(cfg);
            List<String> seqs = new ArrayList<>();
            for (String[] row : manifestRows(manifest, "p_target")) {
                seqs.add(row[1] + "=" + row[0]);
            }
            if (gatedOff == null) {
                gatedOff = seqs;
            } else {
                Assert.assertEquals(seqs, gatedOff,
                        "the audit switch must be deterministic across runs");
            }
        }
        Assert.assertTrue(gatedOff != null && !gatedOff.isEmpty(),
                "the audit path must still produce entrapment");
    }

    // --- Foreign-species entrapment -------------------------------------------------------

    private static final String TARGET_FASTA =
            ">sp|P00001|TEST1_HUMAN First test protein\n"
                    + "SAMPLERPEPTIDEKANOTHERPEPTIDERENDKLENGTHYPEPTIDEK\n"
                    + ">sp|P00002|TEST2_HUMAN Second test protein\n"
                    + "VLLENGTHYSEQUENCEKSHORTKTAILINGSEQR\n";

    /** Distinct sequences so no foreign peptide equals a target peptide. */
    private static final String FOREIGN_FASTA =
            ">sp|Q00001|TEST1_ARATH First foreign protein\n"
                    + "GGFVDSAWQTNCKMHYVTQDNAWCKFFSDGVQWTANCK\n"
                    + ">sp|Q00002|TEST2_ARATH Second foreign protein\n"
                    + "YVTDNQSAWFGCKHMWQVTDNASGFCKTVDNQWASGFMCK\n";

    private Path write(String content, String prefix) throws IOException {
        Path p = Files.createTempFile(prefix, ".fasta");
        Files.writeString(p, content, StandardCharsets.UTF_8);
        return p;
    }

    private EntrapmentFastaGear.Config foreignConfig(Path in, Path outFasta, Path manifest) {
        CParameter.enzyme = 1; // Trypsin
        CParameter.maxMissedCleavages = 1;
        CParameter.minPeptideLength = 6;
        CParameter.maxPeptideLength = 35;
        CParameter.clip_nTerm_M = false;

        EntrapmentFastaGear.Config cfg = new EntrapmentFastaGear.Config();
        cfg.inputFasta = in.toString();
        cfg.outputFasta = outFasta.toString();
        cfg.manifest = manifest.toString();
        cfg.addEntrapment = true;
        cfg.addDecoys = true;
        cfg.applyMzFilter = false;
        cfg.uniqueAccessions = true;
        return cfg;
    }

    /** Manifest rows for a peptide_type, as (sequence, pair_index). */
    private List<String[]> manifestRows(Path manifest, String peptideType) throws IOException {
        List<String[]> rows = new ArrayList<>();
        List<String> lines = Files.readAllLines(manifest, StandardCharsets.UTF_8);
        for (int i = 1; i < lines.size(); i++) {
            String[] f = lines.get(i).split("\t", -1);
            if (f.length >= 5 && f[3].equals(peptideType)) {
                rows.add(new String[]{f[0], f[4]});
            }
        }
        return rows;
    }

    @Test
    public void testForeignEntrapmentUsesForeignSequencesNotAnagrams() throws IOException {
        Path in = write(TARGET_FASTA, "fe_target");
        Path foreign = write(FOREIGN_FASTA, "fe_foreign");
        Path outFasta = Files.createTempFile("fe_out", ".fasta");
        Path manifest = Files.createTempFile("fe_manifest", ".tsv");

        EntrapmentFastaGear.Config cfg = foreignConfig(in, outFasta, manifest);
        cfg.entrapmentSourceFasta = foreign.toString();
        EntrapmentFastaGear.Result r = EntrapmentFastaGear.run(cfg);

        Assert.assertTrue(r.entrapmentFromForeign > 0,
                "at least one target should have drawn a foreign entrapment peptide");

        List<String[]> targets = manifestRows(manifest, "target");
        List<String[]> pTargets = manifestRows(manifest, "p_target");
        Assert.assertTrue(pTargets.size() > 0, "manifest should carry p_target rows");

        Set<String> foreignSeqs = new HashSet<>();
        for (String[] row : pTargets) {
            foreignSeqs.add(row[0]);
        }
        Assert.assertEquals(foreignSeqs.size(), pTargets.size(),
                "every target must draw a UNIQUE foreign peptide");

        // The whole point: an entrapment peptide must NOT be an anagram of its target.
        for (String[] pt : pTargets) {
            for (String[] t : targets) {
                if (!t[1].equals(pt[1])) {
                    continue;
                }
                char[] a = t[0].toCharArray();
                char[] b = pt[0].toCharArray();
                java.util.Arrays.sort(a);
                java.util.Arrays.sort(b);
                Assert.assertFalse(java.util.Arrays.equals(a, b),
                        "foreign entrapment must not be an anagram of its target: "
                                + t[0] + " / " + pt[0]);
            }
        }
    }

    @Test
    public void testForeignEntrapmentIsDeterministic() throws IOException {
        Path in = write(TARGET_FASTA, "fe_target");
        Path foreign = write(FOREIGN_FASTA, "fe_foreign");

        List<String> first = null;
        for (int run = 0; run < 2; run++) {
            Path outFasta = Files.createTempFile("fe_out", ".fasta");
            Path manifest = Files.createTempFile("fe_manifest", ".tsv");
            EntrapmentFastaGear.Config cfg = foreignConfig(in, outFasta, manifest);
            cfg.entrapmentSourceFasta = foreign.toString();
            EntrapmentFastaGear.run(cfg);

            List<String> seqs = new ArrayList<>();
            for (String[] row : manifestRows(manifest, "p_target")) {
                seqs.add(row[1] + "=" + row[0]);
            }
            if (first == null) {
                first = seqs;
            } else {
                Assert.assertEquals(seqs, first,
                        "the same inputs must produce the same foreign assignment");
            }
        }
    }

    @Test
    public void testEntrapmentRatioRejectsUntestedValues() throws IOException {
        // A deliberate floor on an UNTESTED region rather than a claim that lower cannot work:
        // the ratio sweep covered r = 1, 0.5, 0.25 and 0.1, and at 0.1 only ~32 entrapment
        // peptides were accepted, so below that the estimate is counting noise. The error has to
        // say so, otherwise a user hitting it cannot tell a measured limit from an arbitrary one.
        Path in = write(TARGET_FASTA, "fe_target");
        Path outFasta = Files.createTempFile("fe_out", ".fasta");
        Path manifest = Files.createTempFile("fe_manifest", ".tsv");

        EntrapmentFastaGear.Config cfg = foreignConfig(in, outFasta, manifest);
        cfg.entrapmentRatio = 0.05;
        try {
            EntrapmentFastaGear.run(cfg);
            Assert.fail("a ratio below the tested minimum must be rejected");
        } catch (IllegalArgumentException e) {
            Assert.assertTrue(e.getMessage().contains("tested minimum"),
                    "the error must explain the bound is empirical, not arbitrary: "
                            + e.getMessage());
        }

        cfg.entrapmentRatio = 1.5;
        try {
            EntrapmentFastaGear.run(cfg);
            Assert.fail("a ratio above 1.0 must be rejected");
        } catch (IllegalArgumentException e) {
            Assert.assertTrue(e.getMessage().contains("cannot exceed 1.0"), e.getMessage());
        }

        // The boundary itself must be usable - an exclusive bound here would make the lowest
        // characterised ratio unreachable.
        cfg.entrapmentRatio = EntrapmentFastaGear.MIN_ENTRAPMENT_RATIO;
        EntrapmentFastaGear.Result r = EntrapmentFastaGear.run(cfg);
        Assert.assertTrue(r.keptQuartets > 0, "the minimum tested ratio must be accepted");
    }

    @Test
    public void testEntrapmentRatioLeavesSomeTargetsWithoutEntrapment() throws IOException {
        Path in = write(TARGET_FASTA, "fe_target");
        Path outFasta = Files.createTempFile("fe_out", ".fasta");
        Path manifest = Files.createTempFile("fe_manifest", ".tsv");

        EntrapmentFastaGear.Config cfg = foreignConfig(in, outFasta, manifest);
        cfg.entrapmentRatio = 0.5;
        EntrapmentFastaGear.Result r = EntrapmentFastaGear.run(cfg);

        Assert.assertTrue(r.entrapmentNotSelected > 0,
                "a ratio below 1 must leave some targets without entrapment");
        // Those targets are KEPT (as bare target+decoy), not dropped - dropping them would
        // silently shrink the target set, which is not what a ratio means.
        Assert.assertTrue(r.keptQuartets > 0, "unselected targets must still be retained");
        Assert.assertTrue(manifestRows(manifest, "p_target").size()
                        < manifestRows(manifest, "target").size(),
                "fewer entrapment rows than target rows at ratio 0.5");
    }
}
