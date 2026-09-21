package test.java.db;

import main.java.db.PairingManifestReconciler;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Coverage for {@link PairingManifestReconciler}: pruning a pairing manifest to the peptides actually
 * present in the predicted library, dropping a pair group whose target did not survive (so no
 * entrapment is left without a target), keeping a target+decoy when only the entrapment was dropped,
 * renumbering pair indices contiguously, and I&harr;L normalization.
 */
public class PairingManifestReconcilerTest {

    private Path writeTsv(String prefix, String content) throws IOException {
        Path p = Files.createTempFile(prefix, ".tsv");
        Files.writeString(p, content, StandardCharsets.UTF_8);
        return p;
    }

    private List<String[]> readManifest(Path manifest) throws IOException {
        List<String> lines = Files.readAllLines(manifest, StandardCharsets.UTF_8);
        List<String[]> rows = new ArrayList<>();
        for (int i = 1; i < lines.size(); i++) {
            rows.add(lines.get(i).split("\t", -1));
        }
        return rows;
    }

    @Test
    public void prunesToLibrary_dropsGroupWithoutTarget_keepsPartialGroup_renumbers() throws IOException {
        // Prelim manifest: three quartet-ish groups plus an I/L group.
        String manifest = "sequence\tdecoy\tproteins\tpeptide_type\tpeptide_pair_index\n"
                // group 0: everything survives
                + "TARGETONER\tNo\tsp|P1\ttarget\t0\n"
                + "ENTRAPONER\tNo\tsp|P1_p_target\tp_target\t0\n"
                + "DECOYONER\tYes\tdecoy_sp|P1\tdecoy\t0\n"
                + "PDECOYONER\tYes\tdecoy_sp|P1_p_target\tp_decoy\t0\n"
                // group 1: target + decoy survive, entrapment dropped in prediction
                + "TARGETTWOR\tNo\tsp|P2\ttarget\t1\n"
                + "ENTRAPTWOR\tNo\tsp|P2_p_target\tp_target\t1\n"
                + "DECOYTWOR\tYes\tdecoy_sp|P2\tdecoy\t1\n"
                // group 2: target dropped -> whole group must go (no orphan entrapment/decoy)
                + "TARGETTHREER\tNo\tsp|P3\ttarget\t2\n"
                + "ENTRAPTHREER\tNo\tsp|P3_p_target\tp_target\t2\n"
                + "DECOYTHREER\tYes\tdecoy_sp|P3\tdecoy\t2\n"
                // group 3: manifest uses I, library uses L -> must match under I<->L normalization
                + "PEPTIDER\tNo\tsp|P4\ttarget\t3\n"
                + "DECOYFOURK\tYes\tdecoy_sp|P4\tdecoy\t3\n";

        // Library contains only the surviving peptides (group 2's target absent; group 1's entrapment
        // absent; group 3's target spelled with L instead of I).
        String library = "ModifiedPeptide\tStrippedPeptide\tPrecursorMz\tPrecursorCharge\n"
                + "_TARGETONER_\tTARGETONER\t500\t2\n"
                + "_ENTRAPONER_\tENTRAPONER\t500\t2\n"
                + "_DECOYONER_\tDECOYONER\t500\t2\n"
                + "_PDECOYONER_\tPDECOYONER\t500\t2\n"
                + "_TARGETTWOR_\tTARGETTWOR\t500\t2\n"
                + "_DECOYTWOR_\tDECOYTWOR\t500\t2\n"
                + "_ENTRAPTHREER_\tENTRAPTHREER\t500\t2\n" // present but its target is not -> dropped anyway
                + "_PEPTLDER_\tPEPTLDER\t500\t2\n"          // L where the manifest has I
                + "_DECOYFOURK_\tDECOYFOURK\t500\t2\n";

        Path manIn = writeTsv("recon_man", manifest);
        Path lib = writeTsv("recon_lib", library);
        Path manOut = Files.createTempFile("recon_out", ".tsv");

        PairingManifestReconciler.Result r = PairingManifestReconciler.run(
                manIn.toString(), lib.toString(), manOut.toString());

        List<String[]> rows = readManifest(manOut);
        Set<String> keptSeqs = new HashSet<>();
        Set<String> keptPairIdx = new HashSet<>();
        for (String[] row : rows) {
            keptSeqs.add(row[0]);
            keptPairIdx.add(row[4]);
        }

        // Group 0 fully retained.
        Assert.assertTrue(keptSeqs.contains("TARGETONER"));
        Assert.assertTrue(keptSeqs.contains("ENTRAPONER"));
        Assert.assertTrue(keptSeqs.contains("DECOYONER"));
        Assert.assertTrue(keptSeqs.contains("PDECOYONER"));
        // Group 1: target + decoy kept, entrapment dropped (not in library).
        Assert.assertTrue(keptSeqs.contains("TARGETTWOR"));
        Assert.assertTrue(keptSeqs.contains("DECOYTWOR"));
        Assert.assertFalse(keptSeqs.contains("ENTRAPTWOR"));
        // Group 2: target absent -> entire group dropped, including a library-present entrapment.
        Assert.assertFalse(keptSeqs.contains("TARGETTHREER"));
        Assert.assertFalse(keptSeqs.contains("ENTRAPTHREER"),
                "an entrapment whose target did not survive must not be left orphaned");
        Assert.assertFalse(keptSeqs.contains("DECOYTHREER"));
        // Group 3: kept via I<->L normalization (manifest PEPTIDER == library PEPTLDER).
        Assert.assertTrue(keptSeqs.contains("PEPTIDER"), "I/L-equivalent target must be kept");
        Assert.assertTrue(keptSeqs.contains("DECOYFOURK"));

        // Pair indices renumbered contiguously from 0 (3 kept groups -> {0,1,2}).
        Assert.assertEquals(keptPairIdx, new HashSet<>(List.of("0", "1", "2")));

        // No p_target row without a target in its (new) group.
        for (String[] row : rows) {
            if (row[3].equals("p_target")) {
                boolean hasTarget = false;
                for (String[] other : rows) {
                    if (other[4].equals(row[4]) && other[3].equals("target")) {
                        hasTarget = true;
                        break;
                    }
                }
                Assert.assertTrue(hasTarget, "p_target " + row[0] + " must share a group with a target");
            }
        }

        Assert.assertEquals(r.groupsKept, 3);
        Assert.assertEquals(r.groupsDropped, 1);
        Assert.assertEquals(r.keptTargetsWithoutDecoy, 0, "every kept group here retained its decoy");
        Assert.assertEquals(r.libraryPeptidesNotInManifest, 0,
                "the test library contains only peptides present in the manifest");
    }

    @Test
    public void reconcilesAgainstABlibLibrary() throws Exception {
        // Workflow 4 + Skyline emits only a .blib; the reconciler must read its peptide set from
        // RefSpectra.peptideSeq (targets and decoys) instead of a TSV.
        String manifest = "sequence\tdecoy\tproteins\tpeptide_type\tpeptide_pair_index\n"
                + "TARGETONER\tNo\tsp|P1\ttarget\t0\n"
                + "DECOYONER\tYes\tdecoy_sp|P1\tdecoy\t0\n"
                + "TARGETTWOR\tNo\tsp|P2\ttarget\t1\n"   // target absent from blib -> group dropped
                + "DECOYTWOR\tYes\tdecoy_sp|P2\tdecoy\t1\n";
        Path manIn = writeTsv("recon_man_blib", manifest);

        Path blib = Files.createTempFile("recon_lib", ".blib");
        Files.deleteIfExists(blib);
        try (Connection c = DriverManager.getConnection("jdbc:sqlite:" + blib);
             Statement st = c.createStatement()) {
            st.executeUpdate("CREATE TABLE RefSpectra ("
                    + "id INTEGER PRIMARY KEY AUTOINCREMENT, peptideSeq VARCHAR, peptideModSeq VARCHAR);");
            // library has group 0's target+decoy only; group 1's target TARGETTWOR is absent.
            st.executeUpdate("INSERT INTO RefSpectra (peptideSeq, peptideModSeq) VALUES ('TARGETONER','TARGETONER');");
            st.executeUpdate("INSERT INTO RefSpectra (peptideSeq, peptideModSeq) VALUES ('DECOYONER','DECOYONER');");
        }

        Path manOut = Files.createTempFile("recon_out_blib", ".tsv");
        PairingManifestReconciler.Result r = PairingManifestReconciler.run(
                manIn.toString(), blib.toString(), manOut.toString());

        Assert.assertEquals(r.libraryPeptides, 2, "two peptides read from the blib");
        Assert.assertEquals(r.groupsKept, 1);
        Assert.assertEquals(r.groupsDropped, 1, "group whose target is not in the blib is dropped");

        Set<String> keptSeqs = new HashSet<>();
        for (String[] row : readManifest(manOut)) {
            keptSeqs.add(row[0]);
        }
        Assert.assertTrue(keptSeqs.contains("TARGETONER"));
        Assert.assertTrue(keptSeqs.contains("DECOYONER"));
        Assert.assertFalse(keptSeqs.contains("TARGETTWOR"));
        Assert.assertFalse(keptSeqs.contains("DECOYTWOR"));
    }

    @Test(expectedExceptions = IOException.class)
    public void failsFastOnMalformedManifestRow() throws IOException {
        // A truncated manifest row must fail loudly, not silently drop the peptide/group.
        String manifest = "sequence\tdecoy\tproteins\tpeptide_type\tpeptide_pair_index\n"
                + "TARGETONER\tNo\tsp|P1\ttarget\t0\n"
                + "TRUNCATED_ROW\n";
        String library = "ModifiedPeptide\tStrippedPeptide\n_TARGETONER_\tTARGETONER\n";
        Path manIn = writeTsv("recon_bad_man", manifest);
        Path lib = writeTsv("recon_ok_lib", library);
        Path manOut = Files.createTempFile("recon_out_badman", ".tsv");
        PairingManifestReconciler.run(manIn.toString(), lib.toString(), manOut.toString());
    }

    @Test(expectedExceptions = IOException.class)
    public void failsFastOnNullLibraryPath() throws IOException {
        // Direct API misuse with a null library path must fail clearly, not NPE.
        String manifest = "sequence\tdecoy\tproteins\tpeptide_type\tpeptide_pair_index\n"
                + "TARGETONER\tNo\tsp|P1\ttarget\t0\n";
        Path manIn = writeTsv("recon_null_lib_man", manifest);
        Path manOut = Files.createTempFile("recon_out_nulllib", ".tsv");
        PairingManifestReconciler.run(manIn.toString(), null, manOut.toString());
    }

    @Test(expectedExceptions = IOException.class)
    public void failsFastOnMalformedLibraryRow() throws IOException {
        // A library row missing the StrippedPeptide column would undercount the library; fail fast.
        String manifest = "sequence\tdecoy\tproteins\tpeptide_type\tpeptide_pair_index\n"
                + "TARGETONER\tNo\tsp|P1\ttarget\t0\n";
        String library = "ModifiedPeptide\tStrippedPeptide\n_ONLY_ONE_COLUMN_\n";
        Path manIn = writeTsv("recon_ok_man", manifest);
        Path lib = writeTsv("recon_bad_lib", library);
        Path manOut = Files.createTempFile("recon_out_badlib", ".tsv");
        PairingManifestReconciler.run(manIn.toString(), lib.toString(), manOut.toString());
    }

    @Test
    public void handlesReorderedHeader() throws IOException {
        // Columns are resolved by name, so peptide_pair_index need not be last. A valid file with a
        // reordered header must still reconcile correctly.
        String manifest = "peptide_pair_index\tpeptide_type\tdecoy\tproteins\tsequence\n"
                + "0\ttarget\tNo\tsp|P1\tTARGETONER\n"
                + "0\tdecoy\tYes\tdecoy_sp|P1\tDECOYONER\n";
        String library = "ModifiedPeptide\tStrippedPeptide\n_TARGETONER_\tTARGETONER\n_DECOYONER_\tDECOYONER\n";
        Path manIn = writeTsv("recon_reorder_man", manifest);
        Path lib = writeTsv("recon_reorder_lib", library);
        Path manOut = Files.createTempFile("recon_out_reorder", ".tsv");

        PairingManifestReconciler.Result r = PairingManifestReconciler.run(
                manIn.toString(), lib.toString(), manOut.toString());
        Assert.assertEquals(r.groupsKept, 1);
        Set<String> kept = new HashSet<>();
        for (String[] row : readManifest(manOut)) {
            kept.add(row[0]);
        }
        Assert.assertTrue(kept.contains("TARGETONER"));
        Assert.assertTrue(kept.contains("DECOYONER"));
    }

    @Test(expectedExceptions = IOException.class)
    public void failsFastOnShortRowWithReorderedHeader() throws IOException {
        // A row long enough for peptide_pair_index (index 0 here) but too short for 'sequence'
        // (index 4) must still be rejected — the guard uses the max column index, not just iPair.
        String manifest = "peptide_pair_index\tpeptide_type\tdecoy\tproteins\tsequence\n"
                + "0\ttarget\tNo\tsp|P1\tTARGETONER\n"
                + "0\ttarget\tNo\n"; // 3 columns: has pair_index but not the sequence column
        String library = "ModifiedPeptide\tStrippedPeptide\n_TARGETONER_\tTARGETONER\n";
        Path manIn = writeTsv("recon_reorder_short_man", manifest);
        Path lib = writeTsv("recon_reorder_short_lib", library);
        Path manOut = Files.createTempFile("recon_out_reorder_short", ".tsv");
        PairingManifestReconciler.run(manIn.toString(), lib.toString(), manOut.toString());
    }
}
