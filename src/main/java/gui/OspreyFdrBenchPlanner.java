package main.java.gui;

import java.io.File;

/**
 * Decides whether (and where) Osprey should emit an FDRBench input TSV for an Osprey
 * search, via its {@code --fdrbench} option.
 *
 * <p>The FDRBench input is only meaningful for the <b>project</b> search (workflow 5) and only when
 * <b>entrapment</b> peptides are in the searched library (so FDRBench has entrapment hits to
 * estimate FDP from). It is never emitted for the training search, which drives fine-tuning. When
 * emitted, Osprey writes it under a {@code FDRBench} subfolder of the search output directory;
 * the FDR level is taken from Osprey's {@code --fdr-level} (so it follows the Osprey tab
 * setting). Carafe copies the pairing manifest into the same folder so the FDRBench inputs are
 * bundled together.</p>
 *
 * <p>Split out of {@code CarafeGUI} so the decision can be unit-tested without the Swing layer.</p>
 */
public final class OspreyFdrBenchPlanner {

    /** Name of the per-search subfolder that holds the FDRBench inputs. */
    public static final String FDRBENCH_DIR = "FDRBench";
    /** Name of the FDRBench input TSV Osprey writes. */
    public static final String FDRBENCH_INPUT_TSV = "FDRBench-Input.tsv";

    private OspreyFdrBenchPlanner() {
    }

    /**
     * Path passed to Osprey's {@code --fdrbench}, or {@code null} when no FDRBench input should
     * be written.
     *
     * @param entrapmentEnabled whether the "Include entrapment" option is on
     * @param isProjectSearch   whether this is the project search (workflow 5), not the training search
     * @param searchOutDir      the search's output directory (e.g. {@code .../osprey_project})
     * @return {@code <searchOutDir>/FDRBench/FDRBench-Input.tsv}, or {@code null} when not applicable
     */
    public static String fdrBenchInputPath(boolean entrapmentEnabled, boolean isProjectSearch,
            String searchOutDir) {
        if (!entrapmentEnabled || !isProjectSearch || searchOutDir == null || searchOutDir.trim().isEmpty()) {
            return null;
        }
        return searchOutDir + File.separator + FDRBENCH_DIR + File.separator + FDRBENCH_INPUT_TSV;
    }
}
