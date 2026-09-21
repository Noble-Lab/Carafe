package test.java.db;

import main.java.db.DecoyPairPlanner;
import main.java.db.DecoyPairPlanner.DecoyPair;
import main.java.db.DecoyPairPlanner.ManifestEntry;
import main.java.db.DecoyPairPlanner.Precursor;
import main.java.db.EntrapmentFastaGear;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Coverage for {@link DecoyPairPlanner}: the target/decoy pairing written into a Skyline .blib as the
 * additive {@code DecoyPairs} table.
 */
public class DecoyPairPlannerTest {

    private static Precursor prec(int id, String seq, int charge, String mods) {
        return new Precursor(id, seq, charge, DecoyPairPlanner.modKey(mods));
    }

    private static ManifestEntry row(String seq, String type, int pair) {
        boolean isDecoy = type.equals("decoy") || type.equals("p_decoy");
        return new ManifestEntry(seq, isDecoy, type, pair);
    }

    /** Index the planned rows by RefSpectraID for assertions. */
    private static Map<Integer, DecoyPair> byId(List<DecoyPair> pairs) {
        Map<Integer, DecoyPair> m = new HashMap<>();
        for (DecoyPair p : pairs) {
            Assert.assertNull(m.put(p.refSpectraId(), p), "each RefSpectra appears at most once: " + p.refSpectraId());
        }
        return m;
    }

    @Test
    public void pairsTargetWithReverseDecoyAtSameCharge() {
        String t = "PEPTIDEK";
        String d = EntrapmentFastaGear.reversePreservingCterm(t); // "EDITPEPK"
        List<Precursor> precs = List.of(prec(1, t, 2, ""), prec(2, d, 2, ""));
        List<ManifestEntry> man = List.of(row(t, "target", 0), row(d, "decoy", 0));

        List<DecoyPair> pairs = DecoyPairPlanner.plan(precs, man);

        Assert.assertEquals(pairs.size(), 2);
        Map<Integer, DecoyPair> m = byId(pairs);
        Assert.assertEquals(m.get(1).isDecoy(), 0, "target row");
        Assert.assertNull(m.get(1).method(), "target row has no method");
        Assert.assertEquals(m.get(2).isDecoy(), 1, "decoy row");
        Assert.assertEquals(m.get(2).method(), "reverse", "decoy is the C-term-preserved reverse");
        Assert.assertEquals(m.get(1).pairId(), m.get(2).pairId(), "target and decoy share a PairID");
        Assert.assertEquals(m.get(1).isEntrapment(), 0, "a real target/decoy pair is not entrapment");
        Assert.assertEquals(m.get(2).isEntrapment(), 0);
    }

    @Test
    public void oneDistinctPairPerCharge() {
        String t = "PEPTIDEK";
        String d = EntrapmentFastaGear.reversePreservingCterm(t);
        List<Precursor> precs = List.of(
                prec(1, t, 2, ""), prec(2, d, 2, ""),
                prec(3, t, 3, ""), prec(4, d, 3, ""));
        List<ManifestEntry> man = List.of(row(t, "target", 0), row(d, "decoy", 0));

        List<DecoyPair> pairs = DecoyPairPlanner.plan(precs, man);
        Map<Integer, DecoyPair> m = byId(pairs);

        Assert.assertEquals(pairs.size(), 4);
        Assert.assertEquals(m.get(1).pairId(), m.get(2).pairId(), "z2 pair shares a PairID");
        Assert.assertEquals(m.get(3).pairId(), m.get(4).pairId(), "z3 pair shares a PairID");
        Assert.assertNotEquals(m.get(1).pairId(), m.get(3).pairId(), "different charges are different pairs");
    }

    @Test
    public void skipsChargePresentOnOnlyOneSide() {
        String t = "PEPTIDEK";
        String d = EntrapmentFastaGear.reversePreservingCterm(t);
        // target at z2 and z3, decoy only at z2 -> only the z2 pair is emitted.
        List<Precursor> precs = List.of(prec(1, t, 2, ""), prec(2, d, 2, ""), prec(3, t, 3, ""));
        List<ManifestEntry> man = List.of(row(t, "target", 0), row(d, "decoy", 0));

        List<DecoyPair> pairs = DecoyPairPlanner.plan(precs, man);

        Assert.assertEquals(pairs.size(), 2, "the unpaired z3 target is not emitted");
        Assert.assertFalse(byId(pairs).containsKey(3), "z3 target has no decoy, so no row");
    }

    @Test
    public void normalizesIsoleucineToLeucineWhenMatching() {
        // Manifest uses L; the predicted library wrote I (or vice-versa) - they must still pair.
        String manTarget = "PEPTLDEK";
        String manDecoy = EntrapmentFastaGear.reversePreservingCterm(manTarget);
        List<Precursor> precs = List.of(prec(1, "PEPTIDEK", 2, ""), prec(2, manDecoy, 2, ""));
        List<ManifestEntry> man = List.of(row(manTarget, "target", 0), row(manDecoy, "decoy", 0));

        List<DecoyPair> pairs = DecoyPairPlanner.plan(precs, man);

        Assert.assertEquals(pairs.size(), 2, "I/L difference does not prevent pairing");
    }

    @Test
    public void labelsNonReverseDecoyAsCycle() {
        String t = "PEPTIDEK";
        String d = "AAAAAAAK"; // not the reverse -> collision fallback
        List<Precursor> precs = List.of(prec(1, t, 2, ""), prec(2, d, 2, ""));
        List<ManifestEntry> man = List.of(row(t, "target", 0), row(d, "decoy", 0));

        List<DecoyPair> pairs = DecoyPairPlanner.plan(precs, man);

        Assert.assertEquals(byId(pairs).get(2).method(), "cycle");
    }

    @Test
    public void pairsEntrapmentQuartetOnBothSides() {
        String t = "PEPTIDEK", pt = "SAMPLEDK";
        String d = EntrapmentFastaGear.reversePreservingCterm(t);
        String pd = EntrapmentFastaGear.reversePreservingCterm(pt);
        List<Precursor> precs = List.of(
                prec(1, t, 2, ""), prec(2, d, 2, ""),
                prec(3, pt, 2, ""), prec(4, pd, 2, ""));
        List<ManifestEntry> man = List.of(
                row(t, "target", 0), row(pt, "p_target", 0),
                row(d, "decoy", 0), row(pd, "p_decoy", 0));

        List<DecoyPair> pairs = DecoyPairPlanner.plan(precs, man);
        Map<Integer, DecoyPair> m = byId(pairs);

        Assert.assertEquals(pairs.size(), 4, "target/decoy and p_target/p_decoy each pair");
        Assert.assertEquals(m.get(1).pairId(), m.get(2).pairId());
        Assert.assertEquals(m.get(3).pairId(), m.get(4).pairId());
        Assert.assertNotEquals(m.get(1).pairId(), m.get(3).pairId(), "the two side pairs are distinct");
        Assert.assertEquals(m.get(3).isDecoy(), 0, "p_target is a target-side (decoy=No) row");
        Assert.assertEquals(m.get(4).isDecoy(), 1, "p_decoy is a decoy-side row");
        // IsEntrapment is orthogonal to IsDecoy: the real pair is not entrapment, the p_ pair is.
        Assert.assertEquals(m.get(1).isEntrapment(), 0, "real target is not entrapment");
        Assert.assertEquals(m.get(2).isEntrapment(), 0, "real decoy is not entrapment");
        Assert.assertEquals(m.get(3).isEntrapment(), 1, "p_target is an entrapment peptide");
        Assert.assertEquals(m.get(4).isEntrapment(), 1, "p_decoy is an entrapment peptide");
    }

    @Test
    public void pairsModifiedIsoformsByModificationSet() {
        String t = "PEPTMK";
        String d = EntrapmentFastaGear.reversePreservingCterm(t);
        // Each side has an unmodified form and an Oxidation-M form; they must pair like-with-like.
        List<Precursor> precs = List.of(
                prec(1, t, 2, ""), prec(2, d, 2, ""),
                prec(3, t, 2, "Oxidation@M"), prec(4, d, 2, "Oxidation@M"));
        List<ManifestEntry> man = List.of(row(t, "target", 0), row(d, "decoy", 0));

        List<DecoyPair> pairs = DecoyPairPlanner.plan(precs, man);
        Map<Integer, DecoyPair> m = byId(pairs);

        Assert.assertEquals(pairs.size(), 4);
        Assert.assertEquals(m.get(1).pairId(), m.get(2).pairId(), "unmodified forms pair together");
        Assert.assertEquals(m.get(3).pairId(), m.get(4).pairId(), "oxidised forms pair together");
        Assert.assertNotEquals(m.get(1).pairId(), m.get(3).pairId());
    }

    @Test
    public void modKeyIsOrderInsensitiveAndDropsPositions() {
        Assert.assertEquals(DecoyPairPlanner.modKey(""), "");
        Assert.assertEquals(DecoyPairPlanner.modKey("Oxidation@M;Carbamidomethyl@C"),
                DecoyPairPlanner.modKey("Carbamidomethyl@C;Oxidation@M"), "sorted, so order does not matter");
    }

    @Test
    public void parseManifestReadsRowsAndToleratesColumnReorder() throws IOException {
        Path f = Files.createTempFile("pairing", ".tsv");
        try {
            Files.writeString(f, String.join("\n",
                    "peptide_pair_index\tpeptide_type\tproteins\tdecoy\tsequence",
                    "0\ttarget\tP1\tNo\tPEPTIDEK",
                    "0\tdecoy\tP1\tYes\tEDITPEPK") + "\n", StandardCharsets.UTF_8);
            List<ManifestEntry> rows = DecoyPairPlanner.parseManifest(f.toFile());
            Assert.assertEquals(rows.size(), 2);
            Assert.assertEquals(rows.get(0).sequence(), "PEPTIDEK");
            Assert.assertFalse(rows.get(0).isDecoy());
            Assert.assertEquals(rows.get(0).peptideType(), "target");
            Assert.assertEquals(rows.get(0).pairIndex(), 0);
            Assert.assertTrue(rows.get(1).isDecoy(), "decoy=Yes -> isDecoy true");
        } finally {
            Files.deleteIfExists(f);
        }
    }

    @Test(expectedExceptions = IOException.class)
    public void parseManifestThrowsOnMissingColumn() throws IOException {
        Path f = Files.createTempFile("pairing_bad", ".tsv");
        try {
            Files.writeString(f, "sequence\tdecoy\tproteins\n" + "PEPTIDEK\tNo\tP1\n", StandardCharsets.UTF_8);
            DecoyPairPlanner.parseManifest(f.toFile());
        } finally {
            Files.deleteIfExists(f);
        }
    }
}
