package main.java.gui;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

/**
 * Publishes a Carafe library-generation step's output from a LOCAL staging directory to its final
 * (possibly network) directory.
 *
 * <p>The Osprey library steps (the initial NoCut library and the fine-tuned library) run the whole
 * AlphaPepDeep prediction on local disk, because the {@code peptide_forms_*.parquet} prediction
 * chunks are read back by the Python predictor and doing that over an SMB share is slow and has
 * caused native fast-fail crashes; the fine-tuned library is additionally written as a BiblioSpec
 * SQLite {@code .blib}, which cannot be created/locked reliably over SMB either. This publisher moves
 * the finished library plus its report/model artifacts to the share and deliberately leaves the bulky
 * prediction scratch on local disk, so the network never sees the per-chunk churn. It mirrors the
 * local staging already used for the Osprey search {@code .blib}.</p>
 */
public final class CarafeLibraryStaging {

    private CarafeLibraryStaging() {
    }

    /**
     * True when {@code path} is a UNC network share ({@code \\server\...} or {@code //server/...}).
     * The Osprey library steps stage on local disk only for such destinations: a local output
     * directory needs no staging (the user may have deliberately pointed the workflow at a fast local
     * disk), so its output is not forced through a temp copy. Mapped network drives are not
     * auto-detected - use a UNC path (or a local directory) for the reliable behavior.
     */
    public static boolean isNetworkOutputPath(String path) {
        if (path == null) {
            return false;
        }
        String p = path.trim();
        return p.startsWith("\\\\") || p.startsWith("//");
    }

    /**
     * True for the prediction scratch chunks that are consumed in-process and must never be copied to
     * the (possibly network) final directory. Everything else a library step writes -
     * {@code carafe_spectral_library.tsv}/{@code .blib}, the training report, the fine-tuned models,
     * evaluation metrics, etc. - is a deliverable and is published.
     */
    public static boolean isPredictionScratch(String fileName) {
        return fileName.startsWith("peptide_forms_") && fileName.endsWith(".parquet");
    }

    /**
     * Move every file in {@code stageDir} to {@code finalDir} except the prediction scratch, which is
     * left behind on local disk. Cross-volume moves (the usual local-&gt;network case) fall back to
     * copy + delete. Sub-directories are ignored (library steps write a flat directory).
     *
     * @param stageDir local directory the Carafe process wrote to
     * @param finalDir final (possibly network) directory to publish deliverables into
     * @return the number of files published to {@code finalDir}
     * @throws IOException if {@code stageDir} is missing or empty (the step produced no output)
     */
    public static int publish(File stageDir, File finalDir) throws IOException {
        File[] staged = stageDir.listFiles();
        if (staged == null || staged.length == 0) {
            throw new IOException("Carafe produced no output in the staging directory: " + stageDir);
        }
        Files.createDirectories(finalDir.toPath());
        int published = 0;
        for (File f : staged) {
            if (f.isDirectory() || isPredictionScratch(f.getName())) {
                continue;
            }
            Path dest = new File(finalDir, f.getName()).toPath();
            try {
                Files.move(f.toPath(), dest, StandardCopyOption.REPLACE_EXISTING);
            } catch (IOException crossDevice) {
                // Staging dir and destination are on different volumes (the usual case for a network
                // destination): Files.move can't rename across them, so copy then drop the local copy.
                Files.copy(f.toPath(), dest, StandardCopyOption.REPLACE_EXISTING);
                f.delete();
            }
            published++;
        }
        return published;
    }

    /**
     * Local staging directory for a library step, under {@code java.io.tmpdir} and keyed on a hash of
     * its final directory, so the initial and fine-tuned libraries (and the train vs project searches)
     * never collide. Pure path computation - no filesystem side effects.
     */
    public static File stageDirFor(String finalDir) {
        return new File(System.getProperty("java.io.tmpdir"),
                "carafe_lib_stage" + File.separator + Integer.toHexString(finalDir.hashCode()));
    }

    /**
     * {@link #stageDirFor(String)}, created and cleared of any previous run's top-level files, so the
     * post-step publish sees only files produced by THIS run (and a shorter chunk list can't read a
     * previous run's stale {@code peptide_forms_*.parquet}). Mirrors the Osprey blib staging.
     */
    public static File freshStageDir(String finalDir) {
        File d = stageDirFor(finalDir);
        d.mkdirs();
        deleteTopLevel(d);
        return d;
    }

    /**
     * {@link #publish(File, File)}, then delete whatever remains in {@code stageDir} - i.e. the
     * prediction scratch publish() intentionally left behind - so it never touches the network and does
     * not accumulate on local disk between runs.
     *
     * @return the number of deliverables published to {@code finalDir}
     * @throws IOException if {@code stageDir} is missing or empty (the step produced no output)
     */
    public static int publishAndCleanScratch(File stageDir, File finalDir) throws IOException {
        int published = publish(stageDir, finalDir);
        deleteTopLevel(stageDir);
        return published;
    }

    /** Delete the top-level files in {@code dir} (leaves any sub-directories, which steps don't write). */
    private static void deleteTopLevel(File dir) {
        File[] entries = dir.listFiles();
        if (entries != null) {
            for (File f : entries) {
                f.delete();
            }
        }
    }
}
