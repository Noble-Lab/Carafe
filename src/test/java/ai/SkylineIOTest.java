package test.java.ai;

import main.java.ai.SkylineIO;
import org.junit.Assert;
import org.junit.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.Types;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.HashMap;

public class SkylineIOTest {

    @Test
    public void testGet_unimod_from_peptide() throws NoSuchMethodException, InvocationTargetException, IllegalAccessException {
        String mod_pep = "SSSFSC(unimod:4)PE";
        Method method = SkylineIO.class.getDeclaredMethod("get_unimod_from_peptide", String.class);
        method.setAccessible(true);
        HashMap<Integer, String> result = (HashMap<Integer, String>) method.invoke(null, mod_pep);
        org.junit.Assert.assertEquals(result.get(6), "C(unimod:4)");

        mod_pep = "SSSFSC(unimod:4)PE(unimod:35)";
        result = (HashMap<Integer, String>) method.invoke(null, mod_pep);
        org.junit.Assert.assertEquals(result.get(6), "C(unimod:4)");
        org.junit.Assert.assertEquals(result.get(8), "E(unimod:35)");

        mod_pep = "SSSFSC(unimod:4)PC";
        result = (HashMap<Integer, String>) method.invoke(null, mod_pep);
        org.junit.Assert.assertEquals(result.get(6), "C(unimod:4)");

        mod_pep = "SSSFSCPC(unimod:4)";
        result = (HashMap<Integer, String>) method.invoke(null, mod_pep);
        org.junit.Assert.assertEquals(result.get(8), "C(unimod:4)");

        mod_pep = "(unimod:41)SSSFSCPC(unimod:4)";
        result = (HashMap<Integer, String>) method.invoke(null, mod_pep);
        org.junit.Assert.assertEquals(result.get(0), "unimod:41");
        org.junit.Assert.assertEquals(result.get(8), "C(unimod:4)");
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
            Assert.assertEquals(1.2345, rs.getDouble("ionMobility"), 0.0001);
            Assert.assertNull(rs.getObject("collisionalCrossSectionSqA"));
            Assert.assertEquals(SkylineIO.ION_MOBILITY_TYPE_INVERSE_K0, rs.getInt("ionMobilityType"));
        }
    }

    private void assertRetentionTimesIonMobility(Connection connection) throws Exception {
        try (PreparedStatement statement = connection.prepareStatement(
                "SELECT ionMobility, collisionalCrossSectionSqA, ionMobilityType FROM RetentionTimes WHERE RefSpectraID = 1");
             ResultSet rs = statement.executeQuery()) {
            Assert.assertTrue(rs.next());
            Assert.assertEquals(1.2345, rs.getDouble("ionMobility"), 0.0001);
            Assert.assertNull(rs.getObject("collisionalCrossSectionSqA"));
            Assert.assertEquals(SkylineIO.ION_MOBILITY_TYPE_INVERSE_K0, rs.getInt("ionMobilityType"));
        }
    }

    private void assertIonMobilityTypes(Connection connection) throws Exception {
        try (PreparedStatement statement = connection.prepareStatement(
                "SELECT ionMobilityType FROM IonMobilityTypes WHERE id = ?")) {
            statement.setInt(1, SkylineIO.ION_MOBILITY_TYPE_INVERSE_K0);
            try (ResultSet rs = statement.executeQuery()) {
                Assert.assertTrue(rs.next());
                Assert.assertEquals("inverseK0(Vsec/cm^2)", rs.getString("ionMobilityType"));
            }
        }
    }
}
