package main.java.db;

import main.java.util.Cloger;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * Reconcile an FDRBench pairing manifest against the spectral library that was actually predicted
 * from it, so the manifest describes exactly the peptides in the searched library — a single source
 * of truth for Osprey's target-decoy paired competition and for FDRBench.
 *
 * <p>The manifest is written at entrapment-generation time (by {@link EntrapmentFastaGear}) as clean
 * {@code target / p_target / decoy / p_decoy} quartets. The searched library is a downstream
 * prediction of that FASTA and diverges from the manifest: prediction can drop a sequence
 * (AlphaPepDeep failure). Left unreconciled, an entrapment can end up with no target in the manifest,
 * which crashes FDRBench's paired estimator, and FDRBench silently drops any searched peptide missing
 * from the manifest.</p>
 *
 * <p>Because the peptide FASTA is built with the same digest/modification/m-z rules the library
 * prediction uses, the library never <i>adds</i> peptides the manifest lacks — so reconciliation is a
 * <b>prune</b>: keep only manifest rows whose peptide appears in the library, drop any
 * {@code peptide_pair_index} group whose <i>target</i> did not survive (its entrapment/decoy would be
 * orphans, and a decoy whose target is gone has nothing to compete against), and renumber the kept
 * groups contiguously.</p>
 *
 * <p>Sequences are compared with I&rarr;L normalization (isobaric, treated equivalently downstream)
 * so an {@code -I2L} setting mismatch between the two files cannot drop everything.</p>
 */
public final class PairingManifestReconciler {

    /** Library TSV column holding the (unmodified) peptide sequence. */
    private static final String LIBRARY_SEQUENCE_COLUMN = "StrippedPeptide";
    /** Manifest columns (matches {@link EntrapmentFastaGear} output). */
    private static final String MANIFEST_HEADER =
            "sequence\tdecoy\tproteins\tpeptide_type\tpeptide_pair_index";

    private PairingManifestReconciler() {
    }

    /** Summary counts from a reconciliation, returned for logging/testing. */
    public static final class Result {
        public int libraryPeptides;
        public int groupsIn;
        public int groupsKept;
        public int groupsDropped;
        public int rowsIn;
        public int rowsKept;
        public int rowsDropped;
        /** Library peptides absent from the (input) manifest — expected 0; a nonzero value is a warning. */
        public int libraryPeptidesNotInManifest;
        /** Kept groups whose target survived but whose decoy did not (rare prediction-failure edge). */
        public int keptTargetsWithoutDecoy;
    }

    /** One parsed manifest row, carrying its raw column values plus the parsed fields we key on. */
    private static final class Row {
        final String sequence;
        final String decoy;
        final String proteins;
        final String peptideType;

        Row(String sequence, String decoy, String proteins, String peptideType) {
            this.sequence = sequence;
            this.decoy = decoy;
            this.proteins = proteins;
            this.peptideType = peptideType;
        }
    }

    /** I&rarr;L normalized form used for comparing a manifest sequence against a library sequence. */
    static String normalize(String seq) {
        return seq == null ? "" : seq.trim().toUpperCase().replace('I', 'L');
    }

    /**
     * Reconcile {@code manifestIn} against {@code libraryTsv} and write the pruned manifest to
     * {@code manifestOut}.
     *
     * @param manifestIn  the pre-prediction ("prelim") pairing manifest
     * @param libraryTsv  the predicted DIA-NN-format spectral library TSV ({@code carafe_spectral_library.tsv})
     * @param manifestOut where to write the reconciled manifest (may equal {@code manifestIn} is discouraged)
     * @return summary counts
     * @throws IOException on read/write failure
     */
    public static Result run(String manifestIn, String libraryTsv, String manifestOut) throws IOException {
        Cloger.getInstance().logger.info("Reconciling pairing manifest '" + manifestIn
                + "' against predicted library '" + libraryTsv + "'");
        Result r = new Result();

        Set<String> librarySequences = readLibrarySequences(libraryTsv);
        r.libraryPeptides = librarySequences.size();

        // Parse the manifest into pair_index groups, preserving first-seen order.
        Map<String, List<Row>> groups = new LinkedHashMap<>();
        String[] header;
        try (BufferedReader br = Files.newBufferedReader(new File(manifestIn).toPath(), StandardCharsets.UTF_8)) {
            String headerLine = br.readLine();
            if (headerLine == null) {
                throw new IOException("empty manifest: " + manifestIn);
            }
            header = headerLine.split("\t", -1);
            int iSeq = columnIndex(header, "sequence", manifestIn);
            int iDecoy = columnIndex(header, "decoy", manifestIn);
            int iProteins = columnIndex(header, "proteins", manifestIn);
            int iType = columnIndex(header, "peptide_type", manifestIn);
            int iPair = columnIndex(header, "peptide_pair_index", manifestIn);
            // Guard against the highest column index we actually read, not just peptide_pair_index —
            // columns are resolved by name, so any of them could be the last field in a reordered header.
            int maxIdx = Math.max(Math.max(Math.max(iSeq, iDecoy), Math.max(iProteins, iType)), iPair);
            String line;
            int lineNo = 1; // header consumed above
            while ((line = br.readLine()) != null) {
                lineNo++;
                if (line.isEmpty()) {
                    continue;
                }
                String[] c = line.split("\t", -1);
                if (c.length <= maxIdx) {
                    // Fail loudly rather than silently dropping a peptide/group from a corrupt manifest.
                    throw new IOException("malformed manifest " + manifestIn + " at line " + lineNo
                            + ": expected at least " + (maxIdx + 1) + " tab-separated columns, found "
                            + c.length);
                }
                r.rowsIn++;
                groups.computeIfAbsent(c[iPair], k -> new ArrayList<>())
                        .add(new Row(c[iSeq], c[iDecoy], c[iProteins], c[iType]));
            }
        }
        r.groupsIn = groups.size();

        // Track which manifest sequences exist so we can report library peptides with no manifest row.
        Set<String> manifestSequences = new HashSet<>();
        for (List<Row> group : groups.values()) {
            for (Row row : group) {
                manifestSequences.add(normalize(row.sequence));
            }
        }
        for (String libSeq : librarySequences) {
            if (!manifestSequences.contains(libSeq)) {
                r.libraryPeptidesNotInManifest++;
            }
        }

        // Prune and renumber.
        try (BufferedWriter bw = newWriter(manifestOut)) {
            bw.write(MANIFEST_HEADER + "\n");
            int newPairIndex = 0;
            for (List<Row> group : groups.values()) {
                Row target = null;
                for (Row row : group) {
                    if ("target".equals(row.peptideType)) {
                        target = row;
                        break;
                    }
                }
                boolean targetSurvives = target != null && librarySequences.contains(normalize(target.sequence));
                if (!targetSurvives) {
                    r.groupsDropped++;
                    r.rowsDropped += group.size();
                    continue;
                }
                boolean decoyKept = false;
                List<Row> keptRows = new ArrayList<>(group.size());
                for (Row row : group) {
                    if (librarySequences.contains(normalize(row.sequence))) {
                        keptRows.add(row);
                        if ("decoy".equals(row.peptideType)) {
                            decoyKept = true;
                        }
                    } else {
                        r.rowsDropped++;
                    }
                }
                for (Row row : keptRows) {
                    bw.write(row.sequence + "\t" + row.decoy + "\t" + row.proteins + "\t"
                            + row.peptideType + "\t" + newPairIndex + "\n");
                    r.rowsKept++;
                }
                if (!decoyKept) {
                    r.keptTargetsWithoutDecoy++;
                }
                r.groupsKept++;
                newPairIndex++;
            }
        }

        Cloger.getInstance().logger.info(String.format(
                "Reconciled manifest: %d/%d pair groups kept (%d dropped), %d/%d rows kept (%d dropped). "
                        + "Library peptides: %d; not in manifest: %d; kept targets missing a decoy: %d",
                r.groupsKept, r.groupsIn, r.groupsDropped, r.rowsKept, r.rowsIn, r.rowsDropped,
                r.libraryPeptides, r.libraryPeptidesNotInManifest, r.keptTargetsWithoutDecoy));
        if (r.libraryPeptidesNotInManifest > 0) {
            Cloger.getInstance().logger.warn(r.libraryPeptidesNotInManifest + " library peptide(s) are "
                    + "absent from the pairing manifest; FDRBench will drop them. This is expected to be 0 "
                    + "when the peptide FASTA is built with the same rules as the library prediction.");
        }
        return r;
    }

    /**
     * Read the unique I&rarr;L-normalized peptide set from the predicted library. Supports both the
     * DIA-NN-format library TSV (the {@code StrippedPeptide} column) and a Skyline BiblioSpec
     * {@code .blib} (the {@code RefSpectra.peptideSeq} column), so the manifest can be reconciled
     * whether Workflow 4 emits a TSV or a blib.
     */
    private static Set<String> readLibrarySequences(String library) throws IOException {
        if (library == null) {
            throw new IOException("null predicted-library path");
        }
        if (library.toLowerCase(Locale.ROOT).endsWith(".blib")) {
            return readBlibSequences(library);
        }
        Set<String> sequences = new HashSet<>();
        try (BufferedReader br = Files.newBufferedReader(new File(library).toPath(), StandardCharsets.UTF_8)) {
            String headerLine = br.readLine();
            if (headerLine == null) {
                throw new IOException("empty library: " + library);
            }
            String[] header = headerLine.split("\t", -1);
            int iSeq = columnIndex(header, LIBRARY_SEQUENCE_COLUMN, library);
            String line;
            int lineNo = 1; // header consumed above
            while ((line = br.readLine()) != null) {
                lineNo++;
                if (line.isEmpty()) {
                    continue;
                }
                String[] c = line.split("\t", -1);
                if (c.length <= iSeq) {
                    // A short row would silently undercount the library and over-prune the manifest.
                    throw new IOException("malformed library " + library + " at line " + lineNo
                            + ": missing the " + LIBRARY_SEQUENCE_COLUMN + " column (found " + c.length
                            + " tab-separated columns)");
                }
                sequences.add(normalize(c[iSeq]));
            }
        }
        return sequences;
    }

    /** Read the unique I&rarr;L-normalized {@code RefSpectra.peptideSeq} set from a BiblioSpec blib. */
    private static Set<String> readBlibSequences(String blib) throws IOException {
        Set<String> sequences = new HashSet<>();
        String url = "jdbc:sqlite:" + blib;
        try (Connection c = DriverManager.getConnection(url);
             Statement st = c.createStatement();
             ResultSet rs = st.executeQuery("SELECT peptideSeq FROM RefSpectra")) {
            while (rs.next()) {
                String seq = rs.getString(1);
                if (seq != null && !seq.isEmpty()) {
                    sequences.add(normalize(seq));
                }
            }
        } catch (SQLException e) {
            throw new IOException("failed to read peptides from blib: " + blib, e);
        }
        return sequences;
    }

    private static int columnIndex(String[] header, String name, String file) throws IOException {
        for (int i = 0; i < header.length; i++) {
            if (header[i].trim().equalsIgnoreCase(name)) {
                return i;
            }
        }
        throw new IOException("column '" + name + "' not found in " + file);
    }

    private static BufferedWriter newWriter(String out) throws IOException {
        File f = new File(out);
        if (f.getParentFile() != null) {
            f.getParentFile().mkdirs();
        }
        return Files.newBufferedWriter(f.toPath(), StandardCharsets.UTF_8);
    }
}
