package test.java.ai;

import main.java.ai.SkylineIO;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
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
}
