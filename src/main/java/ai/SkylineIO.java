package main.java.ai;

import com.compomics.util.experiment.biology.proteins.Peptide;
import com.compomics.util.experiment.identification.matches.ModificationMatch;
import main.java.input.CPTM;
import main.java.input.CParameter;
import main.java.input.PeptideUtils;
import main.java.util.Cloger;
import org.sqlite.SQLiteConfig;
import tech.tablesaw.api.Row;
import tech.tablesaw.api.Table;
import tech.tablesaw.io.csv.CsvReadOptions;
import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.sql.*;
import java.util.ArrayList;
import java.util.Date;
import java.text.SimpleDateFormat;
import java.util.HashMap;
import java.util.List;
import java.util.zip.Deflater;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class SkylineIO {


    public Connection connection = null;
    private Statement statement = null;
    public PreparedStatement pStatementRefSpectra = null;
    public PreparedStatement pStatementModifications = null;
    public PreparedStatement pStatementRefSpectraPeaks = null;
    public PreparedStatement pStatementRetentionTimes = null;

    private final String skyline_db_file;
    public int numSpectra = -1;

    public static HashMap<String, Double> mod2mass = new HashMap<String, Double>() {{
        put("Oxidation@M", 15.994915);
        put("Carbamidomethyl@C", 57.021464);
        put("Phospho@S", 79.966331);
        put("Phospho@T", 79.966331);
        put("Phospho@Y", 79.966331);
    }};

    public static HashMap<String, String> unimod2mod_name = new HashMap<String, String>() {{
        put("C(unimod:4)", "Carbamidomethylation of C");
        put("M(unimod:35)", "Oxidation of M");
        put("S(unimod:21)", "Phosphorylation of S");
        put("T(unimod:21)", "Phosphorylation of T");
        put("Y(unimod:21)", "Phosphorylation of Y");
    }};

    public SkylineIO(String db_file){
        skyline_db_file = db_file;
        File dbFile = new File(db_file);
        if (dbFile.exists()) {
            Cloger.getInstance().logger.warn("The file {} already exists: overwriting it.", db_file);
            if (!dbFile.delete()) {
                Cloger.getInstance().logger.error("Failed to overwrite the file: {}", db_file);
                System.exit(1);
            }
        }
        try {
            SQLiteConfig config = new SQLiteConfig();
            config.setSynchronous(SQLiteConfig.SynchronousMode.OFF);
            connection = DriverManager.getConnection("jdbc:sqlite:" + db_file, config.toProperties());
            statement = connection.createStatement();
        } catch (Exception e) {
            System.out.println(e.getMessage());
            System.exit(1);
        }
    }

    public void add_LibInfo() {
        PreparedStatement preparedStatement = null;
        try {
            statement.executeUpdate("drop table if exists LibInfo;");
            String createTableSQL = "CREATE TABLE LibInfo ("
                    + "libLSID TEXT, "
                    + "createTime TEXT, "
                    + "numSpecs INTEGER, "
                    + "majorVersion INTEGER, "
                    + "minorVersion INTEGER"
                    + ");";
            // Execute the SQL statement
            statement.executeUpdate(createTableSQL);

            // SQL statement to insert a new row
            String insertSQL = "INSERT INTO LibInfo (libLSID, createTime, numSpecs, majorVersion, minorVersion) VALUES (?, ?, ?, ?, ?)";
            preparedStatement = connection.prepareStatement(insertSQL);
            // Set the values
            File F = new File(this.skyline_db_file);
            preparedStatement.setString(1, "urn:lsid:proteome.gs.washington.edu:spectral_library:bibliospec:nr:"+F.getName());
            // get local creation time in ctime() format e.g. Thu Nov 16 17:02:18 2017
            Date now = new Date();
            // Define the ctime() format pattern
            SimpleDateFormat ctimeFormat = new SimpleDateFormat("EEE MMM dd HH:mm:ss yyyy");
            preparedStatement.setString(2, ctimeFormat.format(now));
            preparedStatement.setInt(3, numSpectra);
            preparedStatement.setInt(4, 1);
            preparedStatement.setInt(5, 10);
            // Execute the insert
            preparedStatement.executeUpdate();
        } catch (Exception e) {
            System.out.println(e.getMessage());
        } finally {
            try {
                if (preparedStatement != null) {
                    preparedStatement.close();
                }
            } catch (Exception e) {
                System.out.println(e.getMessage());
                System.exit(1);
            }
        }
    }

    public void add_RefSpectraPeakAnnotations() {
        try {
            statement.executeUpdate("drop table if exists RefSpectraPeakAnnotations;");
            // SQL for creating the RefSpectraPeakAnnotations table
            String createTableSQL = "CREATE TABLE RefSpectraPeakAnnotations ("
                    + "id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, "
                    + "RefSpectraID INTEGER NOT NULL, "
                    + "peakIndex INTEGER NOT NULL, "
                    + "name VARCHAR(256), "
                    + "formula VARCHAR(256), "
                    + "inchiKey VARCHAR(256), "
                    + "otherKeys VARCHAR(256), "
                    + "charge INTEGER, "
                    + "adduct VARCHAR(256), "
                    + "comment VARCHAR(256), "
                    + "mzTheoretical REAL NOT NULL, "
                    + "mzObserved REAL NOT NULL, "
                    + "constraint FK414516A4F1ADF3B4 foreign key (RefSpectraID) references RefSpectra"
                    + ");";

            // Execute the SQL statement
            statement.executeUpdate(createTableSQL);
            // Empty table
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
    }

    public void add_ScoreTypes() {
        try {
            // Drop the table if it exists
            statement.executeUpdate("drop table if exists ScoreTypes;");
            // SQL for creating the ScoreTypes table
            String createTableSQL = "CREATE TABLE ScoreTypes ("
                    + "id INTEGER PRIMARY KEY, "
                    + "scoreType VARCHAR(128), "
                    + "probabilityType VARCHAR(128)"
                    + ");";

            // Execute the SQL statement to create the table
            statement.executeUpdate(createTableSQL);
        } catch (Exception e) {
            System.out.println("Error creating ScoreTypes table: " + e.getMessage());
        }
    }

    public void add_SpectrumSourceFiles() {
        PreparedStatement preparedStatement = null;
        try {
            // Drop the table if it exists
            statement.executeUpdate("drop table if exists SpectrumSourceFiles;");
            // SQL for creating the SpectrumSourceFiles table
            String createTableSQL = "CREATE TABLE SpectrumSourceFiles ("
                    + "id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, "
                    + "fileName VARCHAR(512), "
                    + "idFileName VARCHAR(512), "
                    + "cutoffScore REAL"
                    + ");";

            // Execute the SQL statement to create the table
            statement.executeUpdate(createTableSQL);
            // SQL statement to insert a new row
            String insertSQL = "INSERT INTO SpectrumSourceFiles (fileName) VALUES (?)";
            preparedStatement = connection.prepareStatement(insertSQL);
            // Set the values
            File F = new File(this.skyline_db_file);
            preparedStatement.setString(1, F.getName().replace(".blib", ""));
            // Execute the insert
            preparedStatement.executeUpdate();

        } catch (Exception e) {
            System.out.println("Error creating SpectrumSourceFiles table: " + e.getMessage());
        }
    }

    public void create_Modifications() {
        try {
            // Drop the table if it exists
            statement.executeUpdate("drop table if exists Modifications;");
            // SQL for creating the Modifications table
            String createTableSQL = "CREATE TABLE Modifications ("
                    + "id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, "
                    + "RefSpectraID INTEGER, "
                    + "position INTEGER, "
                    + "mass REAL, "
                    + "constraint FK928405BDF1ADF3B4 foreign key (RefSpectraID) references RefSpectra"
                    + ");";

            // Execute the SQL statement to create the table
            statement.executeUpdate(createTableSQL);
            String insertSQL = "INSERT INTO Modifications (RefSpectraID, position, mass) VALUES (?, ?, ?);";
            pStatementModifications = connection.prepareStatement(insertSQL);
        } catch (Exception e) {
            System.out.println("Error creating Modifications table: " + e.getMessage());
        }
    }

    public void create_RetentionTimes() {
        try {
            // Drop the table if it exists
            statement.executeUpdate("DROP TABLE IF EXISTS RetentionTimes;");

            // SQL for creating the RetentionTimes table
            String createTableSQL = "CREATE TABLE RetentionTimes ("
                    + "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                    + "RefSpectraID BIGINT, "
                    + "RedundantRefSpectraID BIGINT DEFAULT -1, "
                    + "SpectrumSourceId BIGINT DEFAULT 1, "
                    + "ionMobility DOUBLE, "
                    + "collisionalCrossSectionSqA DOUBLE, "
                    + "ionMobilityHighEnergyOffset DOUBLE, "
                    + "ionMobilityType INTEGER DEFAULT 0, "
                    + "retentionTime DOUBLE, "
                    + "bestSpectrum INTEGER DEFAULT 1, "
                    + "CONSTRAINT FKF3D3B64AF1ADF3B4 FOREIGN KEY (RefSpectraID) REFERENCES RefSpectra"
                    + ");";

            // Execute the SQL statement to create the table
            statement.executeUpdate(createTableSQL);
            // SQL for inserting data into RetentionTimes table
            String insertSQL = "INSERT INTO RetentionTimes (RefSpectraID, retentionTime) VALUES (?, ?);";
            pStatementRetentionTimes = connection.prepareStatement(insertSQL);

        } catch (Exception e) {
            System.out.println("Error creating RetentionTimes table: " + e.getMessage());
        }
    }

    public void create_RefSpectraPeaks() {
        try {
            // Drop the table if it exists
            statement.executeUpdate("drop table if exists RefSpectraPeaks;");
            // SQL for creating the RefSpectraPeaks table
            String createTableSQL = "CREATE TABLE RefSpectraPeaks ("
                    + "RefSpectraID INTEGER, "
                    + "peakMZ BLOB, "
                    + "peakIntensity BLOB,"
                    + "constraint FKACE51F3F1ADF3B4 foreign key (RefSpectraID) references RefSpectra"
                    + ");";

            // Execute the SQL statement to create the table
            statement.executeUpdate(createTableSQL);
            // Prepare the SQL insert statement
            String insertSQL = "INSERT INTO RefSpectraPeaks (RefSpectraID, peakMZ, peakIntensity) VALUES (?, ?, ?);";
            pStatementRefSpectraPeaks = connection.prepareStatement(insertSQL);
        } catch (Exception e) {
            System.out.println("Error creating RefSpectraPeaks table: " + e.getMessage());
        }
    }

    public void create_RefSpectra(){
        try {
            statement.executeUpdate("drop table if exists RefSpectra;");
            // SQL for creating the RefSpectra table
            // BiblioSpec SQLite schema: https://raw.githubusercontent.com/ProteoWizard/pwiz/master/pwiz_tools/BiblioSpec/tests/reference/tables.check
            String createTableSQL = "CREATE TABLE RefSpectra ("
                    + "id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, " // lookup key for RefSpectraPeaks
                    + "peptideSeq VARCHAR(150), " // unmodified peptide sequence, can be left blank for small molecule use
                    + "precursorMZ REAL, " // mz of the precursor that produced this spectrum
                    + "precursorCharge INTEGER, " // should agree with adduct if provided
                    + "peptideModSeq VARCHAR(200), " // modified peptide sequence, can be left blank for small molecule use
                    + "prevAA CHAR(1), " // position of peptide in its parent protein (can be left blank)
                    + "nextAA CHAR(1), " // position of peptide in its parent protein (can be left blank)
                    + "copies INTEGER DEFAULT 1, " // number of copies this spectrum was chosen from if it is in a filtered library
                    + "numPeaks INTEGER, " // number of peaks, should agree with corresponding entry in RefSpectraPeaks
                    + "ionMobility REAL, " // ion mobility value, if known (see ionMobilityType for units)
                    + "collisionalCrossSectionSqA REAL, " // precursor CCS in square Angstroms for ion mobility, if known
                    + "ionMobilityHighEnergyOffset REAL, " // ion mobility value increment for fragments (see ionMobilityType for units)
                    + "ionMobilityType TINYINT, " // ion mobility units (required if ionMobility is used, see IonMobilityTypes table for key)
                    + "retentionTime REAL, " // chromatographic retention time in minutes, if known
                    + "fileID INTEGER DEFAULT 1, " // index into SpectrumSourceFiles table for source file information
                    + "SpecIDinFile VARCHAR(256), " // original spectrum label, id, or description in source file
                    + "score REAL DEFAULT 0, " // spectrum score, typically a probability score (see scoreType)
                    + "scoreType TINYINT DEFAULT 0" // spectrum score type, see ScoreTypes table for meaning
                    + ");";
            
            // Execute the SQL statement
            statement.executeUpdate(createTableSQL);
            // SQL statement to insert a new row
            String insertSQL = "INSERT INTO RefSpectra ("
                    + "peptideSeq, " // unmodified peptide sequence, can be left blank for small molecule use
                    + "precursorMZ, " // mz of the precursor that produced this spectrum
                    + "precursorCharge, " // should agree with adduct if provided
                    + "peptideModSeq, " // modified peptide sequence, can be left blank for small molecule use
                    + "prevAA, " // position of peptide in its parent protein (can be left blank)
                    + "nextAA, " // position of peptide in its parent protein (can be left blank)
                    //+ "copies, " // number of copies this spectrum was chosen from if it is in a filtered library
                    + "numPeaks, " // number of peaks, should agree with corresponding entry in RefSpectraPeaks
                    + "ionMobility, " // ion mobility value, if known (see ionMobilityType for units)
                    + "collisionalCrossSectionSqA, " // precursor CCS in square Angstroms for ion mobility, if known
                    + "ionMobilityHighEnergyOffset, " // ion mobility value increment for fragments (see ionMobilityType for units)
                    + "ionMobilityType, " // ion mobility units (required if ionMobility is used, see IonMobilityTypes table for key)
                    + "retentionTime, " // chromatographic retention time in minutes, if known
                    //+ "fileID, " // index into SpectrumSourceFiles table for source file information
                    + "SpecIDinFile" // original spectrum label, id, or description in source file
                    //+ "score, " // spectrum score, typically a probability score (see scoreType)
                    //+ "scoreType" // spectrum score type, see ScoreTypes table for meaning
                    + ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);";


            pStatementRefSpectra = connection.prepareStatement(insertSQL);
        } catch (Exception e) {
            System.out.println(e.getMessage());
            System.exit(1);
        }
    }

    public void close() {
        try {
            if (pStatementRefSpectra != null) {
                pStatementRefSpectra.close();
            }
            if (pStatementModifications != null) {
                pStatementModifications.close();
            }
            if (pStatementRefSpectraPeaks != null) {
                pStatementRefSpectraPeaks.close();
            }
            if (pStatementRetentionTimes != null) {
                pStatementRetentionTimes.close();
            }
            if (statement != null) {
                statement.close();
            }
            if (connection != null) {
                connection.close();
            }
        } catch (Exception e) {
            System.out.println(e.getMessage());
            System.exit(1);
        }
    }

    // Helper method to convert an array of doubles to little-endian byte array
    public static byte[] doublesToLittleEndianBytes(double[] values) {
        ByteBuffer buffer = ByteBuffer.allocate(values.length * Double.BYTES);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        for (double value : values) {
            buffer.putDouble(value);
        }
        return buffer.array();
    }

    // Helper method to convert an array of floats to little-endian byte array
    public static byte[] floatsToLittleEndianBytes(float[] values) {
        ByteBuffer buffer = ByteBuffer.allocate(values.length * Float.BYTES);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        for (float value : values) {
            buffer.putFloat(value);
        }
        return buffer.array();
    }

    // Helper method to compress data using zlib
    public static byte[] compressData(byte[] data) {
        Deflater deflater = new Deflater();
        deflater.setInput(data);
        deflater.finish();
        byte[] compressedData = new byte[1024];
        int compressedDataLength = deflater.deflate(compressedData);
        deflater.end();
        byte[] output = new byte[compressedDataLength];
        System.arraycopy(compressedData, 0, output, 0, compressedDataLength);
        return output;
    }

    public void add_index(){
        try {
            // SQL statement to create an index on peptideSeq
            statement.executeUpdate("CREATE INDEX idxPeptide ON RefSpectra (peptideSeq, precursorCharge)");
            statement.executeUpdate("CREATE INDEX idxPeptideMod ON RefSpectra (peptideModSeq, precursorCharge)");
            statement.executeUpdate("CREATE INDEX idxRefIdPeaks ON RefSpectraPeaks (RefSpectraID)");
            statement.executeUpdate("CREATE INDEX idxRefIdPeakAnnotations ON RefSpectraPeakAnnotations (RefSpectraID)");
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }

    }

    public static void load_skyline_precursor_table(String precursor_table_file, char sep, List<Peptide> all_peptide_forms, List<Integer> precursor_charges){
        CsvReadOptions.Builder builder = CsvReadOptions.builder(precursor_table_file)
                .maxCharsPerColumn(1000000)
                .separator(sep)
                .header(true);
        CsvReadOptions options = builder.build();
        Table precursorTable = Table.read().usingOptions(options);
        int charge = 0;
        String mod_pep = "";
        for(Row row: precursorTable){
            charge = row.getInt("Precursor Charge");
            precursor_charges.add(charge);
            // SSSFSC(unimod:4)PE
            mod_pep = row.getString("Peptide Modified Sequence Unimod Ids");
            if(mod_pep.contains("unimod")){
                // modification
                HashMap<Integer,String> pos2mod = get_unimod_from_peptide(mod_pep);
                Peptide modPeptide = new Peptide(get_peptide(mod_pep));
                for (int pos: pos2mod.keySet()) {
                    modPeptide.addVariableModification(new ModificationMatch(unimod2mod_name.get(pos2mod.get(pos)), pos));
                }
                all_peptide_forms.add(modPeptide);
            }else{
                Peptide modPeptide = new Peptide(mod_pep);
                all_peptide_forms.add(modPeptide);
            }
        }
        all_peptide_forms.forEach(pep -> pep.getMass(PeptideUtils.modificationParameters,PeptideUtils.sequenceProvider,PeptideUtils.sequenceMatchingParameters));
    }

    static Pattern pattern = Pattern.compile("(.?)\\(unimod:(\\d+)\\)");
    private static HashMap<Integer,String> get_unimod_from_peptide(String peptide){
        HashMap<Integer,String> pos2mod = new HashMap<>();
        Matcher matcher = pattern.matcher(peptide);
        int sequencePosition = 0;
        int mod_length = 0;
        String mod_aa;
        while (matcher.find()) {
            String aminoAcid = matcher.group(1);
            String modificationName = "unimod:" + matcher.group(2);
            if(aminoAcid.isEmpty()){
                sequencePosition = 0;
                mod_aa = modificationName;
            }else{
                mod_aa = aminoAcid +"("+ modificationName+")";
                sequencePosition = peptide.indexOf(mod_aa, sequencePosition) + 1;
            }
            pos2mod.put(sequencePosition-mod_length,mod_aa);
            // System.out.println(aminoAcid+"\t"+modificationName +"\t"+ sequencePosition+"\t"+(sequencePosition-mod_length));
            mod_length = mod_length + modificationName.length() + 2;
        }
        return pos2mod;
    }

    private static String get_peptide(String unimod_pep){
        return unimod_pep.replaceAll("\\(unimod:\\d+\\)", "");
    }
}
