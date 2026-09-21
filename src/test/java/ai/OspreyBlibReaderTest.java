package test.java.ai;

import main.java.ai.OspreyBlibReader;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Tests for {@link OspreyBlibReader}: builds a minimal BiblioSpec-schema SQLite blib (the same
 * tables Osprey's BlibWriter produces) and verifies the conversion to a DIA-NN-style
 * identification TSV, including reconstruction of DIA-NN UniMod modified sequences from the
 * structured Modifications table.
 *
 * <p>TestNG style (argument order {@code assertEquals(actual, expected, message)}) to match the
 * project's other tests so these run under {@code mvn test}.</p>
 */
public class OspreyBlibReaderTest {

    /** Build a small blib at {@code path} with one unmodified, one carbamidomethyl-C, and one
     *  oxidation-M peptide, all from a single source file. */
    private void buildBlib(Path path) throws Exception {
        String url = "jdbc:sqlite:" + path.toString();
        try (Connection c = DriverManager.getConnection(url); Statement st = c.createStatement()) {
            st.executeUpdate("CREATE TABLE SpectrumSourceFiles (id INTEGER PRIMARY KEY AUTOINCREMENT, fileName VARCHAR);");
            st.executeUpdate("CREATE TABLE RefSpectra ("
                    + "id INTEGER PRIMARY KEY AUTOINCREMENT, peptideSeq VARCHAR, precursorMZ REAL, "
                    + "precursorCharge INTEGER, peptideModSeq VARCHAR, retentionTime REAL, "
                    + "startTime REAL, endTime REAL, ionMobility REAL, fileID INTEGER);");
            st.executeUpdate("CREATE TABLE Modifications ("
                    + "id INTEGER PRIMARY KEY AUTOINCREMENT, RefSpectraID INTEGER, position INTEGER, mass REAL);");

            st.executeUpdate("INSERT INTO SpectrumSourceFiles (id, fileName) VALUES (1, 'sample1.mzML');");

            // 1: unmodified PEPTIDEK
            st.executeUpdate("INSERT INTO RefSpectra (id, peptideSeq, precursorMZ, precursorCharge, "
                    + "peptideModSeq, retentionTime, startTime, endTime, ionMobility, fileID) "
                    + "VALUES (1, 'PEPTIDEK', 460.75, 2, 'PEPTIDEK', 10.5, 10.2, 10.8, 0.91, 1);");
            // 2: carbamidomethyl C at position 5 (PEPTCDEK)
            st.executeUpdate("INSERT INTO RefSpectra (id, peptideSeq, precursorMZ, precursorCharge, "
                    + "peptideModSeq, retentionTime, startTime, endTime, ionMobility, fileID) "
                    + "VALUES (2, 'PEPTCDEK', 482.20, 2, 'PEPTC[+57.0]DEK', 20.0, 19.5, 20.5, 0.0, 1);");
            st.executeUpdate("INSERT INTO Modifications (RefSpectraID, position, mass) VALUES (2, 5, 57.021464);");
            // 3: oxidation M at position 3 (PEMTIDEK)
            st.executeUpdate("INSERT INTO RefSpectra (id, peptideSeq, precursorMZ, precursorCharge, "
                    + "peptideModSeq, retentionTime, startTime, endTime, ionMobility, fileID) "
                    + "VALUES (3, 'PEMTIDEK', 475.22, 3, 'PEM[+16.0]TIDEK', 30.0, 29.0, 31.0, 0.0, 1);");
            st.executeUpdate("INSERT INTO Modifications (RefSpectraID, position, mass) VALUES (3, 3, 15.994915);");
        }
    }

    /** Parse the converter's TSV into a list of column maps keyed by header. */
    private List<Map<String, String>> readTsv(String tsvPath) throws Exception {
        List<String> lines = Files.readAllLines(Path.of(tsvPath), StandardCharsets.UTF_8);
        String[] header = lines.get(0).split("\t", -1);
        List<Map<String, String>> rows = new ArrayList<>();
        for (int i = 1; i < lines.size(); i++) {
            if (lines.get(i).isEmpty()) {
                continue;
            }
            String[] c = lines.get(i).split("\t", -1);
            Map<String, String> row = new HashMap<>();
            for (int j = 0; j < header.length && j < c.length; j++) {
                row.put(header[j], c[j]);
            }
            rows.add(row);
        }
        return rows;
    }

    private Map<String, Map<String, String>> byPeptide(List<Map<String, String>> rows) {
        Map<String, Map<String, String>> m = new HashMap<>();
        for (Map<String, String> r : rows) {
            m.put(r.get("Stripped.Sequence"), r);
        }
        return m;
    }

    @Test
    public void convertsBlibToDiannTsvWithExpectedColumns() throws Exception {
        Path blib = Files.createTempFile("osprey_test", ".blib");
        Files.deleteIfExists(blib);
        buildBlib(blib);
        Path outDir = Files.createTempDirectory("osprey_out");

        String tsv = OspreyBlibReader.convertBlibToDiannTsv(blib.toString(), outDir.toString());
        List<Map<String, String>> rows = readTsv(tsv);

        Assert.assertEquals(rows.size(), 3, "three identifications expected");
        // DIA-NN-style columns must be present. Precursor.Id is required by the multi-run merge
        // (AIGear.get_ms_file2psm_diann_multiple_ms_runs) that folder training goes through.
        Map<String, String> any = rows.get(0);
        for (String col : new String[] { "Precursor.Id", "File.Name", "Stripped.Sequence",
                "Modified.Sequence", "Precursor.Charge", "Precursor.MZ", "RT", "RT.Start",
                "RT.Stop", "Q.Value" }) {
            Assert.assertTrue(any.containsKey(col), "missing column " + col);
        }
    }

    @Test
    public void emitsPrecursorIdAsModifiedSequencePlusCharge() throws Exception {
        // Regression: the Osprey report initially omitted Precursor.Id, so multi-mzML (folder)
        // finetuning NPE'd in the "best" merge (hIndex.get("Precursor.Id") -> null). Precursor.Id
        // must be present and equal Modified.Sequence + Precursor.Charge, matching the key
        // AIGear's Skyline->DIA-NN converter builds so both report sources merge identically.
        Path blib = Files.createTempFile("osprey_test", ".blib");
        Files.deleteIfExists(blib);
        buildBlib(blib);
        Path outDir = Files.createTempDirectory("osprey_out");

        String tsv = OspreyBlibReader.convertBlibToDiannTsv(blib.toString(), outDir.toString());
        Map<String, Map<String, String>> byPep = byPeptide(readTsv(tsv));

        // Unmodified, charge 2.
        Assert.assertEquals(byPep.get("PEPTIDEK").get("Precursor.Id"), "PEPTIDEK2");
        // Carbamidomethyl C, charge 2: the UniMod token is carried into the id.
        Assert.assertEquals(byPep.get("PEPTCDEK").get("Precursor.Id"), "PEPTC(UniMod:4)DEK2");
        // Oxidation M, charge 3.
        Assert.assertEquals(byPep.get("PEMTIDEK").get("Precursor.Id"), "PEM(UniMod:35)TIDEK3");

        // And the id is exactly Modified.Sequence + Precursor.Charge for every row.
        for (Map<String, String> r : byPep.values()) {
            Assert.assertEquals(r.get("Precursor.Id"),
                    r.get("Modified.Sequence") + r.get("Precursor.Charge"),
                    "Precursor.Id must be Modified.Sequence + Precursor.Charge");
        }
    }

    @Test
    public void reconstructsUnimodModifiedSequences() throws Exception {
        Path blib = Files.createTempFile("osprey_test", ".blib");
        Files.deleteIfExists(blib);
        buildBlib(blib);
        Path outDir = Files.createTempDirectory("osprey_out");

        String tsv = OspreyBlibReader.convertBlibToDiannTsv(blib.toString(), outDir.toString());
        Map<String, Map<String, String>> byPep = byPeptide(readTsv(tsv));

        // Unmodified: modified sequence equals stripped sequence.
        Assert.assertEquals(byPep.get("PEPTIDEK").get("Modified.Sequence"), "PEPTIDEK");
        // Carbamidomethyl C at position 5 -> C(UniMod:4).
        Assert.assertEquals(byPep.get("PEPTCDEK").get("Modified.Sequence"), "PEPTC(UniMod:4)DEK");
        // Oxidation M at position 3 -> M(UniMod:35).
        Assert.assertEquals(byPep.get("PEMTIDEK").get("Modified.Sequence"), "PEM(UniMod:35)TIDEK");
    }

    @Test
    public void carriesSourceFileChargeAndRt() throws Exception {
        Path blib = Files.createTempFile("osprey_test", ".blib");
        Files.deleteIfExists(blib);
        buildBlib(blib);
        Path outDir = Files.createTempDirectory("osprey_out");

        String tsv = OspreyBlibReader.convertBlibToDiannTsv(blib.toString(), outDir.toString());
        Map<String, Map<String, String>> byPep = byPeptide(readTsv(tsv));

        Map<String, String> r1 = byPep.get("PEPTIDEK");
        Assert.assertEquals(r1.get("File.Name"), "sample1.mzML");
        Assert.assertEquals(r1.get("Precursor.Charge"), "2");
        Assert.assertEquals(Double.parseDouble(r1.get("RT")), 10.5, 1e-6);
        Assert.assertEquals(Double.parseDouble(r1.get("RT.Start")), 10.2, 1e-6);
        Assert.assertEquals(Double.parseDouble(r1.get("RT.Stop")), 10.8, 1e-6);

        Assert.assertEquals(byPep.get("PEMTIDEK").get("Precursor.Charge"), "3");
    }

    @Test
    public void skipsIdentificationsWithUnrecognizedModifications() throws Exception {
        // A peptide whose modification mass isn't recognized would otherwise be emitted with a
        // dropped mod (wrong precursor mass). It must be skipped entirely, not partially
        // de-modified, so only the clean peptide survives the conversion.
        Path blib = Files.createTempFile("osprey_unrecmod", ".blib");
        Files.deleteIfExists(blib);
        String url = "jdbc:sqlite:" + blib;
        try (Connection c = DriverManager.getConnection(url); Statement st = c.createStatement()) {
            st.executeUpdate("CREATE TABLE SpectrumSourceFiles (id INTEGER PRIMARY KEY AUTOINCREMENT, fileName VARCHAR);");
            st.executeUpdate("CREATE TABLE RefSpectra ("
                    + "id INTEGER PRIMARY KEY AUTOINCREMENT, peptideSeq VARCHAR, precursorMZ REAL, "
                    + "precursorCharge INTEGER, peptideModSeq VARCHAR, retentionTime REAL, "
                    + "startTime REAL, endTime REAL, ionMobility REAL, fileID INTEGER);");
            st.executeUpdate("CREATE TABLE Modifications ("
                    + "id INTEGER PRIMARY KEY AUTOINCREMENT, RefSpectraID INTEGER, position INTEGER, mass REAL);");
            st.executeUpdate("INSERT INTO SpectrumSourceFiles (id, fileName) VALUES (1, 'run.mzML');");
            // 1: clean peptide -> kept.
            st.executeUpdate("INSERT INTO RefSpectra (id, peptideSeq, precursorMZ, precursorCharge, "
                    + "peptideModSeq, retentionTime, startTime, endTime, ionMobility, fileID) "
                    + "VALUES (1, 'PEPTIDEK', 460.75, 2, 'PEPTIDEK', 10.0, 9.8, 10.2, 0.0, 1);");
            // 2: SAMPLEK with an unrecognized +123.456 mod on M (pos 3, non-terminal) -> skipped.
            st.executeUpdate("INSERT INTO RefSpectra (id, peptideSeq, precursorMZ, precursorCharge, "
                    + "peptideModSeq, retentionTime, startTime, endTime, ionMobility, fileID) "
                    + "VALUES (2, 'SAMPLEK', 400.0, 2, 'SAM[+123.5]PLEK', 12.0, 11.8, 12.2, 0.0, 1);");
            st.executeUpdate("INSERT INTO Modifications (RefSpectraID, position, mass) VALUES (2, 3, 123.456);");
        }
        Path outDir = Files.createTempDirectory("osprey_out");
        String tsv = OspreyBlibReader.convertBlibToDiannTsv(blib.toString(), outDir.toString());
        Map<String, Map<String, String>> byPep = byPeptide(readTsv(tsv));

        Assert.assertEquals(byPep.size(), 1, "the unrecognized-mod identification must be skipped");
        Assert.assertTrue(byPep.containsKey("PEPTIDEK"), "the clean peptide must survive");
        Assert.assertFalse(byPep.containsKey("SAMPLEK"),
                "must not emit a partially de-modified (wrong-mass) sequence");
    }

    @Test
    public void handlesBlibWithoutModificationsTable() throws Exception {
        // A blib that only has RefSpectra + SpectrumSourceFiles (no Modifications table) must
        // still convert, treating every peptide as unmodified.
        Path blib = Files.createTempFile("osprey_nomods", ".blib");
        Files.deleteIfExists(blib);
        String url = "jdbc:sqlite:" + blib;
        try (Connection c = DriverManager.getConnection(url); Statement st = c.createStatement()) {
            st.executeUpdate("CREATE TABLE SpectrumSourceFiles (id INTEGER PRIMARY KEY AUTOINCREMENT, fileName VARCHAR);");
            st.executeUpdate("CREATE TABLE RefSpectra (id INTEGER PRIMARY KEY AUTOINCREMENT, peptideSeq VARCHAR, "
                    + "precursorMZ REAL, precursorCharge INTEGER, peptideModSeq VARCHAR, retentionTime REAL, fileID INTEGER);");
            st.executeUpdate("INSERT INTO SpectrumSourceFiles (id, fileName) VALUES (1, 'run.mzML');");
            st.executeUpdate("INSERT INTO RefSpectra (id, peptideSeq, precursorMZ, precursorCharge, peptideModSeq, "
                    + "retentionTime, fileID) VALUES (1, 'SAMPLEPEPTIDEK', 500.0, 2, 'SAMPLEPEPTIDEK', 5.0, 1);");
        }
        Path outDir = Files.createTempDirectory("osprey_out");
        String tsv = OspreyBlibReader.convertBlibToDiannTsv(blib.toString(), outDir.toString());
        Map<String, Map<String, String>> byPep = byPeptide(readTsv(tsv));
        Assert.assertEquals(byPep.size(), 1);
        Assert.assertEquals(byPep.get("SAMPLEPEPTIDEK").get("Modified.Sequence"), "SAMPLEPEPTIDEK");
        Assert.assertEquals(byPep.get("SAMPLEPEPTIDEK").get("File.Name"), "run.mzML");
    }
}
