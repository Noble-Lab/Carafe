package test.java.db;

import main.java.db.EntrapmentFastaGear;
import main.java.input.CParameter;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Coverage for {@link EntrapmentFastaGear}: digestion-driven peptide FASTA generation, decoy and
 * entrapment quartet building, collision-drop, the FDRBench pairing manifest, shared-peptide
 * accession joining, the m/z filter, and determinism.
 *
 * <p>TestNG style (argument order {@code assertEquals(actual, expected, message)}) to match the
 * project's other tests so these run under {@code mvn test}.</p>
 */
public class EntrapmentFastaGearTest {

    /** Two small proteins with a couple of clean tryptic peptides each. */
    private static final String FASTA =
            ">sp|P00001|TEST1_HUMAN First test protein\n"
                    + "SAMPLERPEPTIDEKANOTHERPEPTIDERENDK\n"
                    + ">sp|P00002|TEST2_HUMAN Second test protein\n"
                    + "VLLENGTHYPEPTIDEKSHORTKTAILINGSEQR\n";

    private Path writeFasta() throws IOException {
        Path in = Files.createTempFile("entrap_in", ".fasta");
        Files.writeString(in, FASTA, StandardCharsets.UTF_8);
        return in;
    }

    private Path writeFastaFrom(String content) throws IOException {
        Path in = Files.createTempFile("entrap_custom", ".fasta");
        Files.writeString(in, content, StandardCharsets.UTF_8);
        return in;
    }

    private EntrapmentFastaGear.Config baseConfig(Path in, Path outFasta, Path manifest) {
        // Use a permissive digest so the tiny sequences yield peptides deterministically.
        CParameter.enzyme = 1; // Trypsin
        CParameter.maxMissedCleavages = 1;
        CParameter.minPeptideLength = 6;
        CParameter.maxPeptideLength = 35;
        CParameter.clip_nTerm_M = false;

        EntrapmentFastaGear.Config cfg = new EntrapmentFastaGear.Config();
        cfg.inputFasta = in.toString();
        cfg.outputFasta = outFasta.toString();
        cfg.manifest = manifest == null ? null : manifest.toString();
        cfg.applyMzFilter = false;
        cfg.uniqueAccessions = true;
        return cfg;
    }

    private List<String[]> readFastaEntries(Path fasta) throws IOException {
        List<String[]> entries = new ArrayList<>();
        List<String> lines = Files.readAllLines(fasta, StandardCharsets.UTF_8);
        String header = null;
        StringBuilder seq = new StringBuilder();
        for (String line : lines) {
            if (line.startsWith(">")) {
                if (header != null) {
                    entries.add(new String[]{header, seq.toString()});
                }
                header = line.substring(1);
                seq.setLength(0);
            } else {
                seq.append(line.trim());
            }
        }
        if (header != null) {
            entries.add(new String[]{header, seq.toString()});
        }
        return entries;
    }

    /** Parse a written manifest into rows of column->value keyed by header. */
    private List<Map<String, String>> readManifest(Path manifest) throws IOException {
        List<String> lines = Files.readAllLines(manifest, StandardCharsets.UTF_8);
        String[] header = lines.get(0).split("\t", -1);
        List<Map<String, String>> rows = new ArrayList<>();
        for (int i = 1; i < lines.size(); i++) {
            String[] c = lines.get(i).split("\t", -1);
            Map<String, String> row = new java.util.HashMap<>();
            for (int j = 0; j < header.length && j < c.length; j++) {
                row.put(header[j], c[j]);
            }
            rows.add(row);
        }
        return rows;
    }

    @Test
    public void targetDecoyOnly_emitsTwoEntriesPerPeptide() throws IOException {
        Path in = writeFasta();
        Path outFasta = Files.createTempFile("entrap_out", ".fasta");
        EntrapmentFastaGear.Config cfg = baseConfig(in, outFasta, null);
        cfg.addEntrapment = false;
        cfg.addDecoys = true;

        EntrapmentFastaGear.Result r = EntrapmentFastaGear.run(cfg);

        Assert.assertTrue(r.keptQuartets > 0, "expected some target peptides");
        Assert.assertEquals(r.targetEntries, r.keptQuartets, "target entries == kept quartets");
        Assert.assertEquals(r.decoyEntries, r.keptQuartets, "decoy entries == kept quartets");
        Assert.assertEquals(r.pTargetEntries, 0, "no entrapment entries");
        Assert.assertEquals(r.pDecoyEntries, 0, "no p_decoy entries");

        List<String[]> entries = readFastaEntries(outFasta);
        Assert.assertEquals(entries.size(), r.targetEntries + r.decoyEntries);
    }

    @Test
    public void reverseDecoy_matchesOspreyReverseAndCycle() {
        // Reverse keeps the C-terminus and reverses the rest (Osprey DecoyGenerator.ReverseSequence).
        Assert.assertEquals(EntrapmentFastaGear.reversePreservingCterm("ABCDEK"), "EDCBAK");
        // Cycle rotates the internal residues by N, keeping the C-terminus (CycleSequence).
        Assert.assertEquals(EntrapmentFastaGear.cyclePreservingCterm("ABCDEK", 1), "BCDEAK");
        Assert.assertEquals(EntrapmentFastaGear.cyclePreservingCterm("ABCDEK", 2), "CDEABK");
        // Length <= 2 is returned unchanged.
        Assert.assertEquals(EntrapmentFastaGear.reversePreservingCterm("AK"), "AK");

        // No collision: the decoy is just the reverse.
        Assert.assertEquals(EntrapmentFastaGear.generateReverseDecoy("ABCDEK", new HashSet<>()),
                "EDCBAK");
        // Reverse collides with a real target -> cycle to the next unique form.
        Set<String> targets = new HashSet<>();
        targets.add("EDCBAK");
        Assert.assertEquals(EntrapmentFastaGear.generateReverseDecoy("ABCDEK", targets), "BCDEAK");
    }

    @Test
    public void entrapmentOn_emitsFullQuartet() throws IOException {
        Path in = writeFasta();
        Path outFasta = Files.createTempFile("entrap_out", ".fasta");
        EntrapmentFastaGear.Config cfg = baseConfig(in, outFasta, null);
        cfg.addEntrapment = true;
        cfg.addDecoys = true;

        EntrapmentFastaGear.Result r = EntrapmentFastaGear.run(cfg);

        Assert.assertEquals(r.targetEntries, r.keptQuartets);
        Assert.assertEquals(r.pTargetEntries, r.keptQuartets);
        Assert.assertEquals(r.decoyEntries, r.keptQuartets);
        Assert.assertEquals(r.pDecoyEntries, r.keptQuartets);
    }

    @Test
    public void perPeptideCounterSuffixIsUnique() throws IOException {
        Path in = writeFasta();
        Path outFasta = Files.createTempFile("entrap_out", ".fasta");
        EntrapmentFastaGear.Config cfg = baseConfig(in, outFasta, null);
        cfg.addEntrapment = false;
        cfg.addDecoys = true;
        cfg.uniqueAccessions = true;

        EntrapmentFastaGear.run(cfg);

        List<String[]> entries = readFastaEntries(outFasta);
        Set<String> headers = new HashSet<>();
        for (String[] e : entries) {
            Assert.assertTrue(headers.add(e[0]), "duplicate FASTA header: " + e[0]);
            // Every accession field should carry a _pepNNNNN counter.
            Assert.assertTrue(e[0].matches(".*_pep\\d+.*"), "missing _pep counter in: " + e[0]);
        }
    }

    @Test
    public void manifestHasCleanAccessionsAndFiveColumns() throws IOException {
        Path in = writeFasta();
        Path outFasta = Files.createTempFile("entrap_out", ".fasta");
        Path manifest = Files.createTempFile("entrap_man", ".tsv");
        EntrapmentFastaGear.Config cfg = baseConfig(in, outFasta, manifest);
        cfg.addEntrapment = true;
        cfg.addDecoys = true;

        EntrapmentFastaGear.run(cfg);

        List<String> lines = Files.readAllLines(manifest, StandardCharsets.UTF_8);
        Assert.assertEquals(lines.get(0),
                "sequence\tdecoy\tproteins\tpeptide_type\tpeptide_pair_index");
        boolean sawTarget = false;
        boolean sawDecoy = false;
        for (int i = 1; i < lines.size(); i++) {
            String[] c = lines.get(i).split("\t", -1);
            Assert.assertEquals(c.length, 5, "manifest row must have 5 columns: " + lines.get(i));
            // proteins column must NOT carry the per-peptide counter suffix.
            Assert.assertFalse(c[2].matches(".*_pep\\d+.*"), "manifest proteins must be clean: " + c[2]);
            if (c[3].equals("target")) {
                sawTarget = true;
                Assert.assertEquals(c[1], "No");
            }
            if (c[3].equals("decoy")) {
                sawDecoy = true;
                Assert.assertEquals(c[1], "Yes");
            }
        }
        Assert.assertTrue(sawTarget);
        Assert.assertTrue(sawDecoy);
    }

    @Test
    public void shuffleIsDeterministicAndPreservesCterm() {
        String pep = "SAMPLEPEPTIDEK";
        String s1 = EntrapmentFastaGear.shufflePreservingCterm(pep, 24);
        String s2 = EntrapmentFastaGear.shufflePreservingCterm(pep, 24);
        Assert.assertEquals(s2, s1, "same seed must give same shuffle");
        Assert.assertEquals(s1.charAt(s1.length() - 1), pep.charAt(pep.length() - 1),
                "C-terminal residue preserved");
        Assert.assertFalse(s1.equals(EntrapmentFastaGear.shufflePreservingCterm(pep, 42)),
                "different master seed should differ");
        // A permutation: same multiset of residues.
        char[] a = pep.toCharArray();
        char[] b = s1.toCharArray();
        java.util.Arrays.sort(a);
        java.util.Arrays.sort(b);
        Assert.assertEquals(new String(b), new String(a), "shuffle must be a permutation");
    }

    @Test
    public void mzRangeFilterRespectsCharges() {
        // A peptide whose 2+/3+ m/z lands inside a typical DIA window.
        Double mass = EntrapmentFastaGear.peptideNeutralMass("SAMPLEPEPTIDEK");
        Assert.assertNotNull(mass);
        Assert.assertTrue(EntrapmentFastaGear.fitsMzRange(mass, new int[]{2, 3}, 400.0, 900.0));
        Assert.assertFalse(EntrapmentFastaGear.fitsMzRange(mass, new int[]{2, 3}, 2000.0, 3000.0));
        // Unknown residue -> null mass.
        Assert.assertNull(EntrapmentFastaGear.peptideNeutralMass("SAMPLEX"));
    }

    @Test
    public void sharedPeptideAppearsOnceWithJoinedProteins() throws IOException {
        // The peptide SHAREDPEPTIDEK is tryptic in both proteins.
        String fasta =
                ">sp|P00001|A_HUMAN protein A\nSHAREDPEPTIDEKAAAAAAK\n"
                        + ">sp|P00002|B_HUMAN protein B\nMMMMMMKSHAREDPEPTIDEKCCCCK\n";
        Path in = writeFastaFrom(fasta);
        Path outFasta = Files.createTempFile("entrap_out", ".fasta");
        Path manifest = Files.createTempFile("entrap_man", ".tsv");
        EntrapmentFastaGear.Config cfg = baseConfig(in, outFasta, manifest);
        cfg.addEntrapment = false;
        cfg.addDecoys = true;

        EntrapmentFastaGear.Result r = EntrapmentFastaGear.run(cfg);
        Assert.assertTrue(r.sharedEntries >= 1, "at least one peptide shared across two proteins");

        // The shared target FASTA header must join both source accessions with ';'.
        List<String[]> entries = readFastaEntries(outFasta);
        boolean sawJoinedShared = false;
        for (String[] e : entries) {
            if (e[1].equals("SHAREDPEPTIDEK") && e[0].contains(";")
                    && e[0].contains("P00001") && e[0].contains("P00002")) {
                sawJoinedShared = true;
            }
        }
        Assert.assertTrue(sawJoinedShared, "shared peptide header should join P00001;P00002");

        // The manifest proteins column for the shared target lists both clean accessions.
        boolean sawManifestShared = false;
        for (Map<String, String> row : readManifest(manifest)) {
            if (row.get("sequence").equals("SHAREDPEPTIDEK") && row.get("peptide_type").equals("target")) {
                Assert.assertTrue(row.get("proteins").contains("P00001"));
                Assert.assertTrue(row.get("proteins").contains("P00002"));
                sawManifestShared = true;
            }
        }
        Assert.assertTrue(sawManifestShared);
    }

    @Test
    public void noSyntheticPeptideCollidesWithATarget() throws IOException {
        // Collision-drop invariant: no decoy/p_target/p_decoy in the output may equal a target.
        Path in = writeFasta();
        Path outFasta = Files.createTempFile("entrap_out", ".fasta");
        Path manifest = Files.createTempFile("entrap_man", ".tsv");
        EntrapmentFastaGear.Config cfg = baseConfig(in, outFasta, manifest);
        cfg.addEntrapment = true;
        cfg.addDecoys = true;
        EntrapmentFastaGear.run(cfg);

        Set<String> targets = new HashSet<>();
        List<String[]> synthetics = new ArrayList<>();
        for (Map<String, String> row : readManifest(manifest)) {
            if (row.get("peptide_type").equals("target")) {
                targets.add(row.get("sequence"));
            } else {
                synthetics.add(new String[] { row.get("peptide_type"), row.get("sequence") });
            }
        }
        for (String[] s : synthetics) {
            Assert.assertFalse(targets.contains(s[1]), s[0] + " collides with a target: " + s[1]);
        }
    }

    @Test
    public void manifestPairIndicesAreContiguousAndGroupTarget() throws IOException {
        Path in = writeFasta();
        Path outFasta = Files.createTempFile("entrap_out", ".fasta");
        Path manifest = Files.createTempFile("entrap_man", ".tsv");
        EntrapmentFastaGear.Config cfg = baseConfig(in, outFasta, manifest);
        cfg.addEntrapment = true;
        cfg.addDecoys = true;
        EntrapmentFastaGear.Result r = EntrapmentFastaGear.run(cfg);

        // Group rows by pair_index; every group must contain exactly one target.
        Map<Integer, Integer> targetsPerGroup = new java.util.TreeMap<>();
        int maxIdx = -1;
        for (Map<String, String> row : readManifest(manifest)) {
            int idx = Integer.parseInt(row.get("peptide_pair_index"));
            maxIdx = Math.max(maxIdx, idx);
            if (row.get("peptide_type").equals("target")) {
                targetsPerGroup.merge(idx, 1, Integer::sum);
            }
        }
        Assert.assertEquals(maxIdx + 1, r.keptQuartets, "one pair_index per kept quartet");
        Assert.assertEquals(targetsPerGroup.size(), r.keptQuartets);
        for (int v : targetsPerGroup.values()) {
            Assert.assertEquals(v, 1, "exactly one target per pair_index group");
        }
    }

    @Test
    public void mzFilterReducesRetainedPeptides() throws IOException {
        Path in = writeFasta();

        Path outNoFilter = Files.createTempFile("entrap_nofilter", ".fasta");
        EntrapmentFastaGear.Config noFilter = baseConfig(in, outNoFilter, null);
        noFilter.applyMzFilter = false;
        int retainedNoFilter = EntrapmentFastaGear.run(noFilter).uniqueTargets;

        Path outFilter = Files.createTempFile("entrap_filter", ".fasta");
        EntrapmentFastaGear.Config filtered = baseConfig(in, outFilter, null);
        filtered.applyMzFilter = true;
        // A window no precursor can satisfy -> everything dropped on m/z.
        filtered.minMz = 5000.0;
        filtered.maxMz = 6000.0;
        EntrapmentFastaGear.Result fr = EntrapmentFastaGear.run(filtered);

        Assert.assertTrue(fr.droppedOutOfMz > 0, "m/z filter must drop some peptides");
        Assert.assertTrue(fr.uniqueTargets < retainedNoFilter);
    }

    @Test
    public void outputIsDeterministicAcrossRuns() throws IOException {
        Path in = writeFasta();
        Path outA = Files.createTempFile("entrap_a", ".fasta");
        Path manA = Files.createTempFile("entrap_a_man", ".tsv");
        Path outB = Files.createTempFile("entrap_b", ".fasta");
        Path manB = Files.createTempFile("entrap_b_man", ".tsv");

        EntrapmentFastaGear.Config a = baseConfig(in, outA, manA);
        a.addEntrapment = true;
        EntrapmentFastaGear.run(a);
        EntrapmentFastaGear.Config b = baseConfig(in, outB, manB);
        b.addEntrapment = true;
        EntrapmentFastaGear.run(b);

        Assert.assertEquals(Files.readString(outB), Files.readString(outA), "FASTA must be deterministic");
        Assert.assertEquals(Files.readString(manB), Files.readString(manA), "manifest must be deterministic");
    }

    @Test
    public void uniqueAccessionsToggleControlsPepSuffix() throws IOException {
        Path in = writeFasta();
        Path out = Files.createTempFile("entrap_nouniq", ".fasta");
        EntrapmentFastaGear.Config cfg = baseConfig(in, out, null);
        cfg.addEntrapment = false;
        cfg.addDecoys = false; // targets only, simplest to inspect
        cfg.uniqueAccessions = false;
        EntrapmentFastaGear.run(cfg);

        for (String[] e : readFastaEntries(out)) {
            Assert.assertFalse(e[0].matches(".*_pep\\d+.*"),
                    "no _pep suffix when uniqueAccessions is off: " + e[0]);
        }
    }
}
