package main.java.ai;

import main.java.input.CModification;
import main.java.util.Cloger;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Reads an Osprey output {@code .blib} (BiblioSpecLite SQLite) of detected peptides and
 * writes a DIA-NN-style identification TSV that Carafe's existing DIA-NN report path can consume.
 *
 * <p>Osprey does not generate its own predicted library, so Carafe finetunes its model on
 * Osprey's search results. The blib supplies the peptide identifications (stripped sequence,
 * modifications, precursor charge, apex/boundary RT); Carafe still extracts the measured
 * fragment intensities and performs transition masking from the mzML files exactly as it does
 * for a DIA-NN report (the report never carried measured intensities either).</p>
 *
 * <p>Modified sequences are reconstructed in DIA-NN UniMod notation (e.g.
 * {@code AAAAC(UniMod:4)LDK}) from the structured {@code Modifications} table (RefSpectraID,
 * 1-based position, mass delta) rather than parsing the {@code peptideModSeq} mass-bracket
 * string. Each (residue, mass-delta) is mapped to a {@code X(UniMod:ID)} token using
 * {@link CModification}'s known modifications, with a small fallback table for the common mods.
 * The reconstructed {@code Modified.Sequence} is then read back by
 * {@link AIGear#get_modification_diann(String, String)}.</p>
 *
 * <p><b>Validation note:</b> the blib schema (RefSpectra / SpectrumSourceFiles / Modifications)
 * is read defensively from {@link ResultSetMetaData} so missing optional columns
 * (startTime/endTime/ionMobility) degrade gracefully. Column presence is confirmed against
 * Osprey's {@code BlibWriter}; validate end-to-end against a real Osprey blib.</p>
 */
public class OspreyBlibReader {

    /** Mass tolerance (Da) for matching a blib modification mass delta to a known UniMod. */
    private static final double MOD_MASS_TOL = 0.01;

    /** Fallback (residue, mass-delta) -> UniMod token for the common mods, used when
     *  CModification does not provide a match. */
    private static final Map<String, String> FALLBACK_MOD_TOKENS = new LinkedHashMap<>();

    static {
        // residue + ":" + rounded mass(3dp) -> UniMod token appended after the residue
        FALLBACK_MOD_TOKENS.put("C:57.021", "(UniMod:4)");   // Carbamidomethyl
        FALLBACK_MOD_TOKENS.put("M:15.995", "(UniMod:35)");  // Oxidation
        FALLBACK_MOD_TOKENS.put("S:79.966", "(UniMod:21)");  // Phospho
        FALLBACK_MOD_TOKENS.put("T:79.966", "(UniMod:21)");  // Phospho
        FALLBACK_MOD_TOKENS.put("Y:79.966", "(UniMod:21)");  // Phospho
    }

    /** A single (residue, mass) -> UniMod token entry built from CModification. */
    private static final class ModEntry {
        final char residue;
        final double mass;
        final String token; // e.g. "(UniMod:4)"

        ModEntry(char residue, double mass, String token) {
            this.residue = residue;
            this.mass = mass;
            this.token = token;
        }
    }

    private final String blibPath;
    private final List<ModEntry> modLookup = new ArrayList<>();

    public OspreyBlibReader(String blibPath) {
        this.blibPath = blibPath;
        buildModLookup();
    }

    /**
     * Convenience entry point: read {@code blibPath} and write a DIA-NN-style TSV under
     * {@code outDir}, returning the TSV path.
     *
     * @param blibPath path to the Osprey output .blib
     * @param outDir   Carafe output directory
     * @return path to the written DIA-NN-style TSV
     * @throws IOException on read/write failure
     */
    public static String convertBlibToDiannTsv(String blibPath, String outDir) throws IOException {
        OspreyBlibReader reader = new OspreyBlibReader(blibPath);
        String out = outDir + File.separator + "osprey_report_diann.tsv";
        reader.writeDiannTsv(out);
        return out;
    }

    /** Build the (residue, mass) -> UniMod-token lookup from CModification's known mods. */
    private void buildModLookup() {
        try {
            CModification cmod = CModification.getInstance();
            for (Map.Entry<String, String> e : cmod.unimod2modification_code.entrySet()) {
                // key like "C(UniMod:4)" -> residue 'C', token "(UniMod:4)"
                String key = e.getKey();
                if (key.length() < 2 || key.charAt(1) != '(') {
                    continue;
                }
                char residue = key.charAt(0);
                String token = key.substring(1);
                String code = e.getValue();
                String modName = cmod.modification_code2modification.get(code);
                if (modName == null) {
                    continue;
                }
                double mass;
                try {
                    mass = cmod.getPTMbyName(modName).getMass();
                } catch (Exception ex) {
                    continue;
                }
                modLookup.add(new ModEntry(residue, mass, token));
            }
        } catch (Exception e) {
            Cloger.getInstance().logger.warn(
                    "Could not build modification lookup from CModification; using fallback table only: "
                            + e.getMessage());
        }
    }

    /** Map a (residue, mass-delta) to a DIA-NN UniMod token, or {@code null} if unknown. */
    private String tokenFor(char residue, double massDelta) {
        for (ModEntry m : modLookup) {
            if (m.residue == residue && Math.abs(m.mass - massDelta) <= MOD_MASS_TOL) {
                return m.token;
            }
        }
        String fb = FALLBACK_MOD_TOKENS.get(residue + ":" + String.format("%.3f", massDelta));
        if (fb != null) {
            return fb;
        }
        // N-terminal acetylation (UniMod:1), commonly stored at position 1.
        if (Math.abs(massDelta - 42.010565) <= MOD_MASS_TOL) {
            return "(UniMod:1)";
        }
        return null;
    }

    /**
     * Reconstruct a DIA-NN-format Modified.Sequence from the stripped peptide + its mods.
     * Returns the stripped sequence unchanged when there are no (recognized) modifications.
     */
    private String buildModifiedSequence(String peptide, List<double[]> mods) {
        if (mods == null || mods.isEmpty()) {
            return peptide;
        }
        // position (1-based) -> token to insert after that residue; position 0 -> N-term prefix.
        Map<Integer, String> posToken = new HashMap<>();
        String nTermPrefix = "";
        boolean unrecognized = false;
        for (double[] m : mods) {
            int pos = (int) Math.round(m[0]);
            double massDelta = m[1];
            char residue = (pos >= 1 && pos <= peptide.length()) ? peptide.charAt(pos - 1) : '\0';
            String token = tokenFor(residue, massDelta);
            if (token == null) {
                // Try as an N-terminal modification (acetyl) when at the peptide start.
                if (pos <= 1 && Math.abs(massDelta - 42.010565) <= MOD_MASS_TOL) {
                    nTermPrefix = "(UniMod:1)";
                    continue;
                }
                unrecognized = true;
                continue;
            }
            if (token.equals("(UniMod:1)") && pos <= 1) {
                nTermPrefix = token;
            } else {
                posToken.put(pos, token);
            }
        }
        if (unrecognized) {
            // Emitting a partially de-modified sequence would give this precursor the wrong mass
            // (a dropped mod), so skip the peptide entirely rather than write a corrupt row.
            Cloger.getInstance().logger.warn(
                    "OspreyBlibReader: unrecognized modification on peptide " + peptide
                            + "; skipping this identification.");
            return null;
        }
        StringBuilder sb = new StringBuilder();
        sb.append(nTermPrefix);
        for (int i = 1; i <= peptide.length(); i++) {
            sb.append(peptide.charAt(i - 1));
            String token = posToken.get(i);
            if (token != null) {
                sb.append(token);
            }
        }
        return sb.toString();
    }

    /**
     * Read the blib and write a DIA-NN-style identification TSV.
     *
     * @param outTsv output TSV path
     * @throws IOException on read/write failure
     */
    public void writeDiannTsv(String outTsv) throws IOException {
        File blib = new File(blibPath);
        if (!blib.isFile()) {
            throw new IOException("Osprey blib not found: " + blibPath);
        }

        String url = "jdbc:sqlite:" + blibPath;
        int nRows = 0;
        int nSkipped = 0;
        try (Connection conn = DriverManager.getConnection(url)) {
            // Source-file id -> fileName.
            Map<Long, String> fileNames = readSourceFiles(conn);
            // RefSpectraID -> list of [position, massDelta].
            Map<Long, List<double[]>> modsById = readModifications(conn);

            Set<String> refCols = tableColumns(conn, "RefSpectra");
            boolean hasStart = refCols.contains("startTime");
            boolean hasEnd = refCols.contains("endTime");
            boolean hasIm = refCols.contains("ionMobility");
            boolean hasFileId = refCols.contains("fileID");

            File out = new File(outTsv);
            if (out.getParentFile() != null) {
                out.getParentFile().mkdirs();
            }
            try (BufferedWriter w = Files.newBufferedWriter(out.toPath(), StandardCharsets.UTF_8);
                 Statement st = conn.createStatement();
                 ResultSet rs = st.executeQuery("SELECT * FROM RefSpectra")) {

                // DIA-NN-style header (see PSMConfig.use_osprey_blib_column_names()).
                w.write("File.Name\tStripped.Sequence\tModified.Sequence\tPrecursor.Charge\t"
                        + "Precursor.MZ\tRT\tRT.Start\tRT.Stop\tIM\tMS2.Scan\tQ.Value\tPEP\n");

                while (rs.next()) {
                    long id = rs.getLong("id");
                    String peptide = rs.getString("peptideSeq");
                    if (peptide == null || peptide.isEmpty()) {
                        continue;
                    }
                    int charge = rs.getInt("precursorCharge");
                    double precursorMz = rs.getDouble("precursorMZ");
                    double rt = rs.getDouble("retentionTime");
                    double rtStart = hasStart ? rs.getDouble("startTime") : rt;
                    double rtStop = hasEnd ? rs.getDouble("endTime") : rt;
                    double im = hasIm ? rs.getDouble("ionMobility") : 0.0;
                    String fileName = "osprey";
                    if (hasFileId) {
                        long fileId = rs.getLong("fileID");
                        fileName = fileNames.getOrDefault(fileId, fileName);
                    } else if (fileNames.size() == 1) {
                        fileName = fileNames.values().iterator().next();
                    }

                    String modSeq = buildModifiedSequence(peptide, modsById.get(id));
                    if (modSeq == null) {
                        // Unrecognized modification: skip rather than emit a wrong-mass precursor.
                        nSkipped++;
                        continue;
                    }

                    // Q.Value/PEP are 0: Osprey already FDR-filtered its output, so every row is
                    // a confident identification for finetuning.
                    w.write(fileName + "\t" + peptide + "\t" + modSeq + "\t" + charge + "\t"
                            + precursorMz + "\t" + rt + "\t" + rtStart + "\t" + rtStop + "\t"
                            + im + "\t0\t0\t0\n");
                    nRows++;
                }
            }
        } catch (SQLException e) {
            throw new IOException("Failed to read Osprey blib: " + blibPath, e);
        }
        Cloger.getInstance().logger.info(
                "OspreyBlibReader: wrote " + nRows + " identifications to " + outTsv
                        + (nSkipped > 0
                            ? " (" + nSkipped + " skipped: unrecognized modifications)" : ""));
    }

    /** Read SpectrumSourceFiles into id -> fileName. */
    private Map<Long, String> readSourceFiles(Connection conn) {
        Map<Long, String> map = new HashMap<>();
        try (Statement st = conn.createStatement();
             ResultSet rs = st.executeQuery("SELECT id, fileName FROM SpectrumSourceFiles")) {
            while (rs.next()) {
                map.put(rs.getLong("id"), rs.getString("fileName"));
            }
        } catch (SQLException e) {
            Cloger.getInstance().logger.warn(
                    "OspreyBlibReader: could not read SpectrumSourceFiles: " + e.getMessage());
        }
        return map;
    }

    /** Read the Modifications table into RefSpectraID -> list of [position, massDelta]. */
    private Map<Long, List<double[]>> readModifications(Connection conn) {
        Map<Long, List<double[]>> map = new HashMap<>();
        if (!tableExists(conn, "Modifications")) {
            return map;
        }
        String sql = "SELECT RefSpectraID, position, mass FROM Modifications";
        try (PreparedStatement ps = conn.prepareStatement(sql);
             ResultSet rs = ps.executeQuery()) {
            while (rs.next()) {
                long refId = rs.getLong("RefSpectraID");
                double position = rs.getDouble("position");
                double mass = rs.getDouble("mass");
                map.computeIfAbsent(refId, k -> new ArrayList<>()).add(new double[]{position, mass});
            }
        } catch (SQLException e) {
            Cloger.getInstance().logger.warn(
                    "OspreyBlibReader: could not read Modifications: " + e.getMessage());
        }
        return map;
    }

    /** Column names of a table (empty set if the table is absent). */
    private Set<String> tableColumns(Connection conn, String table) {
        Set<String> cols = new HashSet<>();
        try (Statement st = conn.createStatement();
             ResultSet rs = st.executeQuery("SELECT * FROM " + table + " LIMIT 0")) {
            ResultSetMetaData md = rs.getMetaData();
            for (int i = 1; i <= md.getColumnCount(); i++) {
                cols.add(md.getColumnName(i));
            }
        } catch (SQLException e) {
            Cloger.getInstance().logger.warn(
                    "OspreyBlibReader: could not read columns of " + table + ": " + e.getMessage());
        }
        return cols;
    }

    private boolean tableExists(Connection conn, String table) {
        String sql = "SELECT name FROM sqlite_master WHERE type='table' AND name=?";
        try (PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setString(1, table);
            try (ResultSet rs = ps.executeQuery()) {
                return rs.next();
            }
        } catch (SQLException e) {
            return false;
        }
    }
}
