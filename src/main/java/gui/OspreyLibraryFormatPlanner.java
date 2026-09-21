package main.java.gui;

import java.util.Locale;

/**
 * Decides the output format of the finetuned library that the Osprey workflows deliver, honoring
 * the GUI's "Spectral Library Format" selection instead of always forcing DIA-NN TSV.
 *
 * <p>When the user chooses <b>Skyline</b>, the finetuned library is written as a BiblioSpec
 * {@code .blib} — the deliverable to import into Skyline. The DIA-NN {@code .tsv} is only kept when
 * something downstream still needs it:</p>
 * <ul>
 *   <li><b>Workflow 4</b> (finetune + build library, no project search): the blib is the whole
 *       deliverable, so only the blib is written. The pairing-manifest reconciler reads the peptide
 *       set straight from the blib.</li>
 *   <li><b>Workflow 5</b> (end-to-end): the Osprey project search still reads a DIA-NN TSV (until
 *       Osprey reads the blib directly), so both the blib (Skyline deliverable) and the TSV
 *       (project search + reconciliation) are written via the {@code "Skyline,DIA-NN"} combined
 *       format.</li>
 * </ul>
 *
 * <p>The initial target-decoy library (lib1) always stays DIA-NN TSV — it feeds Osprey's training
 * search — so this planner only governs the finetuned library (lib2). Split out of
 * {@code CarafeGUI.runCarafe()} so the decision can be unit-tested without the Swing layer.</p>
 */
public final class OspreyLibraryFormatPlanner {

    /** Carafe's default library file basename before the format-specific suffix is applied. */
    public static final String LIB_TSV = "carafe_spectral_library.tsv";
    public static final String LIB_BLIB = "carafe_spectral_library.blib";

    private OspreyLibraryFormatPlanner() {
    }

    /** Immutable plan for the finetuned-library (lib2) build. */
    public static final class Plan {
        /** {@code -lf_type} value for the finetuned-library Carafe step. */
        public final String lfType;
        /** Whether a Skyline {@code .blib} is produced (the deliverable). */
        public final boolean blib;
        /** Whether a DIA-NN {@code .tsv} is produced. */
        public final boolean tsv;
        /** Basename of the library the pairing-manifest reconciler reads. */
        public final String reconcileFileName;
        /** Basename whose presence marks the finetuned-library step complete (reuse skip check). */
        public final String skipCheckFileName;
        /** Basename of the library the Workflow 5 project search reads (always the TSV). */
        public final String searchFileName;

        Plan(String lfType, boolean blib, boolean tsv, String reconcileFileName,
                String skipCheckFileName, String searchFileName) {
            this.lfType = lfType;
            this.blib = blib;
            this.tsv = tsv;
            this.reconcileFileName = reconcileFileName;
            this.skipCheckFileName = skipCheckFileName;
            this.searchFileName = searchFileName;
        }
    }

    /**
     * Plan the finetuned-library output.
     *
     * @param userFormat the GUI "Spectral Library Format" selection (e.g. "DIA-NN", "Skyline")
     * @param endToEnd   whether this is Workflow 5 (has an Osprey project search that reads the TSV)
     * @return the format plan
     */
    public static Plan plan(String userFormat, boolean endToEnd) {
        boolean skyline = userFormat != null && userFormat.toLowerCase(Locale.ROOT).contains("skyline");
        if (!skyline) {
            // DIA-NN (or any non-Skyline TSV format): unchanged behavior.
            return new Plan("DIA-NN", false, true, LIB_TSV, LIB_TSV, LIB_TSV);
        }
        if (endToEnd) {
            // Workflow 5: blib deliverable + TSV for the project search and reconciliation. The reuse
            // skip check is the TSV — the project search needs it, so a missing TSV must re-run lib2.
            return new Plan("Skyline,DIA-NN", true, true, LIB_TSV, LIB_TSV, LIB_TSV);
        }
        // Workflow 4: blib only; the reconciler reads peptides from the blib.
        return new Plan("Skyline", true, false, LIB_BLIB, LIB_BLIB, LIB_TSV);
    }
}
