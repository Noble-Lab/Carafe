package test.java.ai;

import main.java.ai.SkylineIO;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.Statement;
import java.sql.Types;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.HashMap;

/**
 * TestNG (argument order {@code assertEquals(actual, expected)}) so these run under
 * {@code mvn test}; the project's Surefire uses the TestNG provider and skips JUnit tests.
 */
public class SkylineIOTest {

    @Test
    public void testGet_unimod_from_peptide() throws NoSuchMethodException, InvocationTargetException, IllegalAccessException {
        // The parser's regex matches the DIA-NN "UniMod:" casing; the modified sequences below
        // use that casing accordingly. (This test previously used lowercase "unimod:" and never
        // ran under Surefire, so the mismatch went unnoticed.)
        String mod_pep = "SSSFSC(UniMod:4)PE";
        Method method = SkylineIO.class.getDeclaredMethod("get_unimod_from_peptide", String.class);
        method.setAccessible(true);
        HashMap<Integer, String> result = (HashMap<Integer, String>) method.invoke(null, mod_pep);
        Assert.assertEquals(result.get(6), "C(UniMod:4)");

        mod_pep = "SSSFSC(UniMod:4)PE(UniMod:35)";
        result = (HashMap<Integer, String>) method.invoke(null, mod_pep);
        Assert.assertEquals(result.get(6), "C(UniMod:4)");
        Assert.assertEquals(result.get(8), "E(UniMod:35)");

        mod_pep = "SSSFSC(UniMod:4)PC";
        result = (HashMap<Integer, String>) method.invoke(null, mod_pep);
        Assert.assertEquals(result.get(6), "C(UniMod:4)");

        mod_pep = "SSSFSCPC(UniMod:4)";
        result = (HashMap<Integer, String>) method.invoke(null, mod_pep);
        Assert.assertEquals(result.get(8), "C(UniMod:4)");

        mod_pep = "(UniMod:41)SSSFSCPC(UniMod:4)";
        result = (HashMap<Integer, String>) method.invoke(null, mod_pep);
        Assert.assertEquals(result.get(0), "UniMod:41");
        Assert.assertEquals(result.get(8), "C(UniMod:4)");
    }

    @Test
    public void testSkylineBlibIonMobilitySchemaAndValues() throws Exception {
        Path tempFile = Files.createTempFile("skyline-ion-mobility", ".blib");
        SkylineIO skylineIO = new SkylineIO(tempFile.toString());
        try {
            skylineIO.add_SpectrumSourceFiles();
            skylineIO.add_ScoreTypes();
            skylineIO.add_IonMobilityTypes();
            skylineIO.create_RefSpectra();
            skylineIO.create_RetentionTimes();

            skylineIO.pStatementRefSpectra.setString(1, "PEPTIDE");
            skylineIO.pStatementRefSpectra.setDouble(2, 500.25);
            skylineIO.pStatementRefSpectra.setInt(3, 2);
            skylineIO.pStatementRefSpectra.setString(4, "PEPTIDE");
            skylineIO.pStatementRefSpectra.setNull(5, Types.CHAR);
            skylineIO.pStatementRefSpectra.setNull(6, Types.CHAR);
            skylineIO.pStatementRefSpectra.setInt(7, 5);
            skylineIO.pStatementRefSpectra.setDouble(8, 1.2345);
            skylineIO.pStatementRefSpectra.setNull(9, Types.DOUBLE);
            skylineIO.pStatementRefSpectra.setNull(10, Types.DOUBLE);
            skylineIO.pStatementRefSpectra.setInt(11, SkylineIO.ION_MOBILITY_TYPE_INVERSE_K0);
            skylineIO.pStatementRefSpectra.setDouble(12, 10.5);
            skylineIO.pStatementRefSpectra.setNull(13, Types.VARCHAR);
            skylineIO.pStatementRefSpectra.executeUpdate();

            skylineIO.pStatementRetentionTimes.setInt(1, 1);
            skylineIO.pStatementRetentionTimes.setDouble(2, 1.2345);
            skylineIO.pStatementRetentionTimes.setNull(3, Types.DOUBLE);
            skylineIO.pStatementRetentionTimes.setNull(4, Types.DOUBLE);
            skylineIO.pStatementRetentionTimes.setInt(5, SkylineIO.ION_MOBILITY_TYPE_INVERSE_K0);
            skylineIO.pStatementRetentionTimes.setDouble(6, 10.5);
            skylineIO.pStatementRetentionTimes.executeUpdate();

            assertRefSpectraIonMobility(skylineIO.connection);
            assertRetentionTimesIonMobility(skylineIO.connection);
            assertIonMobilityTypes(skylineIO.connection);
        } finally {
            skylineIO.close();
            Files.deleteIfExists(tempFile);
        }
    }

    private void assertRefSpectraIonMobility(Connection connection) throws Exception {
        try (PreparedStatement statement = connection.prepareStatement(
                "SELECT ionMobility, collisionalCrossSectionSqA, ionMobilityType FROM RefSpectra WHERE id = 1");
             ResultSet rs = statement.executeQuery()) {
            Assert.assertTrue(rs.next());
            Assert.assertEquals(rs.getDouble("ionMobility"), 1.2345, 0.0001);
            Assert.assertNull(rs.getObject("collisionalCrossSectionSqA"));
            Assert.assertEquals(rs.getInt("ionMobilityType"), SkylineIO.ION_MOBILITY_TYPE_INVERSE_K0);
        }
    }

    private void assertRetentionTimesIonMobility(Connection connection) throws Exception {
        try (PreparedStatement statement = connection.prepareStatement(
                "SELECT ionMobility, collisionalCrossSectionSqA, ionMobilityType FROM RetentionTimes WHERE RefSpectraID = 1");
             ResultSet rs = statement.executeQuery()) {
            Assert.assertTrue(rs.next());
            Assert.assertEquals(rs.getDouble("ionMobility"), 1.2345, 0.0001);
            Assert.assertNull(rs.getObject("collisionalCrossSectionSqA"));
            Assert.assertEquals(rs.getInt("ionMobilityType"), SkylineIO.ION_MOBILITY_TYPE_INVERSE_K0);
        }
    }

    private void assertIonMobilityTypes(Connection connection) throws Exception {
        try (PreparedStatement statement = connection.prepareStatement(
                "SELECT ionMobilityType FROM IonMobilityTypes WHERE id = ?")) {
            statement.setInt(1, SkylineIO.ION_MOBILITY_TYPE_INVERSE_K0);
            try (ResultSet rs = statement.executeQuery()) {
                Assert.assertTrue(rs.next());
                Assert.assertEquals(rs.getString("ionMobilityType"), "inverseK0(Vsec/cm^2)");
            }
        }
    }

    /**
     * End-to-end blib output: populate a library the way
     * {@code AIGear.generate_spectral_library_parquet_skyline} does (RefSpectra + peaks + mods + RT
     * in one transaction), then read it back and verify the BiblioSpec schema and encodings Skyline
     * relies on — the mass-delta {@code peptideModSeq}, the little-endian m/z(double)/intensity(float)
     * peak BLOBs, the structured Modifications rows, and LibInfo counts.
     */
    @Test
    public void writesReadableBlibWithModsPeaksAndRt() throws Exception {
        Path tmp = Files.createTempFile("skyline-roundtrip", ".blib");
        Files.deleteIfExists(tmp);
        double[] mz = { 175.119, 288.203, 401.287 };
        float[] inten = { 1.0f, 0.5f, 0.25f };

        SkylineIO sky = new SkylineIO(tmp.toString());
        try {
            sky.add_SpectrumSourceFiles();
            sky.add_ScoreTypes();
            sky.add_IonMobilityTypes();
            sky.add_RefSpectraPeakAnnotations();
            sky.create_RefSpectra();
            sky.create_Modifications();
            sky.create_RetentionTimes();
            sky.pStatementRefSpectra.setNull(5, Types.CHAR);
            sky.pStatementRefSpectra.setNull(6, Types.CHAR);
            sky.pStatementRefSpectra.setNull(13, Types.VARCHAR);
            sky.create_RefSpectraPeaks();
            sky.connection.setAutoCommit(false);

            writeRef(sky, "LGGNEQVTR", 487.2567, 2, "LGGNEQVTR", mz.length, 18.5);
            writePeaks(sky, 1, mz, inten);
            writeRt(sky, 1, 18.5);
            // Carbamidomethyl C at position 5, mass-delta peptideModSeq (SkylineIO's current style).
            writeRef(sky, "PEPTCIDER", 558.7531, 2, "PEPTC[57.021464]IDER", mz.length, 30.7);
            writePeaks(sky, 2, mz, inten);
            writeRt(sky, 2, 30.7);
            sky.pStatementModifications.setInt(1, 2);
            sky.pStatementModifications.setInt(2, 5);
            sky.pStatementModifications.setDouble(3, 57.021464);
            sky.pStatementModifications.addBatch();

            sky.pStatementRefSpectra.executeBatch();
            sky.pStatementRefSpectraPeaks.executeBatch();
            sky.pStatementRetentionTimes.executeBatch();
            sky.pStatementModifications.executeBatch();
            sky.numSpectra = 2;
            sky.add_LibInfo();
            sky.add_index();
            sky.connection.commit();
        } finally {
            sky.close();
        }

        try (Connection c = DriverManager.getConnection("jdbc:sqlite:" + tmp);
             Statement st = c.createStatement()) {
            try (ResultSet rs = st.executeQuery("SELECT numSpecs FROM LibInfo")) {
                Assert.assertTrue(rs.next());
                Assert.assertEquals(rs.getInt("numSpecs"), 2);
            }
            try (ResultSet rs = st.executeQuery("SELECT peptideSeq, peptideModSeq, precursorCharge, "
                    + "numPeaks, retentionTime FROM RefSpectra WHERE id = 2")) {
                Assert.assertTrue(rs.next());
                Assert.assertEquals(rs.getString("peptideSeq"), "PEPTCIDER");
                Assert.assertEquals(rs.getString("peptideModSeq"), "PEPTC[57.021464]IDER");
                Assert.assertEquals(rs.getInt("precursorCharge"), 2);
                Assert.assertEquals(rs.getInt("numPeaks"), 3);
                Assert.assertEquals(rs.getDouble("retentionTime"), 30.7, 1e-6);
            }
            try (ResultSet rs = st.executeQuery("SELECT peakMZ, peakIntensity FROM RefSpectraPeaks "
                    + "WHERE RefSpectraID = 1")) {
                Assert.assertTrue(rs.next());
                double[] gotMz = leDoubles(rs.getBytes("peakMZ"));
                float[] gotInt = leFloats(rs.getBytes("peakIntensity"));
                Assert.assertEquals(gotMz.length, 3, "three m/z peaks round-trip");
                Assert.assertEquals(gotMz[0], 175.119, 1e-6);
                Assert.assertEquals(gotMz[2], 401.287, 1e-6);
                Assert.assertEquals(gotInt.length, 3);
                Assert.assertEquals(gotInt[0], 1.0f, 1e-6f);
                Assert.assertEquals(gotInt[2], 0.25f, 1e-6f);
            }
            try (ResultSet rs = st.executeQuery("SELECT position, mass FROM Modifications "
                    + "WHERE RefSpectraID = 2")) {
                Assert.assertTrue(rs.next());
                Assert.assertEquals(rs.getInt("position"), 5, "1-based modification position");
                Assert.assertEquals(rs.getDouble("mass"), 57.021464, 1e-6);
            }
        } finally {
            Files.deleteIfExists(tmp);
        }
    }

    private void writeRef(SkylineIO sky, String seq, double mz, int z, String modSeq, int numPeaks,
            double rt) throws Exception {
        sky.pStatementRefSpectra.setString(1, seq);
        sky.pStatementRefSpectra.setDouble(2, mz);
        sky.pStatementRefSpectra.setInt(3, z);
        sky.pStatementRefSpectra.setString(4, modSeq);
        sky.pStatementRefSpectra.setInt(7, numPeaks);
        sky.pStatementRefSpectra.setNull(8, Types.DOUBLE);
        sky.pStatementRefSpectra.setNull(9, Types.DOUBLE);
        sky.pStatementRefSpectra.setNull(10, Types.DOUBLE);
        sky.pStatementRefSpectra.setInt(11, SkylineIO.ION_MOBILITY_TYPE_NONE);
        sky.pStatementRefSpectra.setDouble(12, rt);
        sky.pStatementRefSpectra.addBatch();
    }

    private void writePeaks(SkylineIO sky, int refId, double[] mz, float[] inten) throws Exception {
        sky.pStatementRefSpectraPeaks.setInt(1, refId);
        sky.pStatementRefSpectraPeaks.setBytes(2, SkylineIO.doublesToLittleEndianBytes(mz));
        sky.pStatementRefSpectraPeaks.setBytes(3, SkylineIO.floatsToLittleEndianBytes(inten));
        sky.pStatementRefSpectraPeaks.addBatch();
    }

    private void writeRt(SkylineIO sky, int refId, double rt) throws Exception {
        sky.pStatementRetentionTimes.setInt(1, refId);
        sky.pStatementRetentionTimes.setNull(2, Types.DOUBLE);
        sky.pStatementRetentionTimes.setNull(3, Types.DOUBLE);
        sky.pStatementRetentionTimes.setNull(4, Types.DOUBLE);
        sky.pStatementRetentionTimes.setInt(5, SkylineIO.ION_MOBILITY_TYPE_NONE);
        sky.pStatementRetentionTimes.setDouble(6, rt);
        sky.pStatementRetentionTimes.addBatch();
    }

    private static double[] leDoubles(byte[] b) {
        ByteBuffer buf = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN);
        double[] d = new double[b.length / Double.BYTES];
        for (int i = 0; i < d.length; i++) {
            d[i] = buf.getDouble();
        }
        return d;
    }

    private static float[] leFloats(byte[] b) {
        ByteBuffer buf = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN);
        float[] f = new float[b.length / Float.BYTES];
        for (int i = 0; i < f.length; i++) {
            f[i] = buf.getFloat();
        }
        return f;
    }

    /**
     * The additive DecoyPairs table round-trips: an entrapment target row (NULL Method, IsEntrapment=1)
     * and its entrapment decoy row (Method set) sharing a PairID, plus the PairID index. This is the
     * self-describing pairing + entrapment annotation Skyline ignores but follow-on tools consume.
     */
    @Test
    public void testDecoyPairsTableRoundTrip() throws Exception {
        Path dbPath = Files.createTempFile("decoypairs", ".blib");
        Files.deleteIfExists(dbPath); // SkylineIO creates the file fresh
        try {
            SkylineIO io = new SkylineIO(dbPath.toString());
            io.create_DecoyPairs();
            // entrapment target: IsDecoy=0, IsEntrapment=1, PairID=1, Method NULL
            io.pStatementDecoyPairs.setInt(1, 10);
            io.pStatementDecoyPairs.setInt(2, 0);
            io.pStatementDecoyPairs.setInt(3, 1);
            io.pStatementDecoyPairs.setInt(4, 1);
            io.pStatementDecoyPairs.setNull(5, Types.VARCHAR);
            io.pStatementDecoyPairs.addBatch();
            // entrapment decoy: IsDecoy=1, IsEntrapment=1, PairID=1, Method reverse
            io.pStatementDecoyPairs.setInt(1, 11);
            io.pStatementDecoyPairs.setInt(2, 1);
            io.pStatementDecoyPairs.setInt(3, 1);
            io.pStatementDecoyPairs.setInt(4, 1);
            io.pStatementDecoyPairs.setString(5, "reverse");
            io.pStatementDecoyPairs.addBatch();
            io.pStatementDecoyPairs.executeBatch();
            io.close();

            try (Connection c = DriverManager.getConnection("jdbc:sqlite:" + dbPath);
                 Statement st = c.createStatement()) {
                ResultSet rs = st.executeQuery(
                        "SELECT RefSpectraID, IsDecoy, IsEntrapment, PairID, Method FROM DecoyPairs "
                                + "ORDER BY RefSpectraID");
                Assert.assertTrue(rs.next());
                Assert.assertEquals(rs.getInt("RefSpectraID"), 10);
                Assert.assertEquals(rs.getInt("IsDecoy"), 0);
                Assert.assertEquals(rs.getInt("IsEntrapment"), 1, "entrapment target");
                Assert.assertEquals(rs.getInt("PairID"), 1);
                Assert.assertNull(rs.getString("Method"), "target row has NULL Method");
                Assert.assertTrue(rs.next());
                Assert.assertEquals(rs.getInt("RefSpectraID"), 11);
                Assert.assertEquals(rs.getInt("IsDecoy"), 1);
                Assert.assertEquals(rs.getInt("IsEntrapment"), 1, "entrapment decoy");
                Assert.assertEquals(rs.getInt("PairID"), 1, "the decoy shares the target's PairID");
                Assert.assertEquals(rs.getString("Method"), "reverse");
                Assert.assertFalse(rs.next(), "exactly two rows");

                ResultSet idx = st.executeQuery(
                        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_DecoyPairs_PairID'");
                Assert.assertTrue(idx.next(), "the PairID index is created");
            }
        } finally {
            Files.deleteIfExists(dbPath);
        }
    }
}
