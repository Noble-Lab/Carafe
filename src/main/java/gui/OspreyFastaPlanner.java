package main.java.gui;

/**
 * Decides how to build the OspreySharp peptide FASTAs for workflows 4 and 5.
 *
 * <p>Entrapment ({@code p_target}/{@code p_decoy}) peptides belong ONLY in the library-DB FASTA —
 * the one that feeds the finetuned library used as the project-search library (workflow 5) or as
 * the deliverable (workflow 4). They must NEVER enter the training-DB FASTA: the training search
 * drives AI fine-tuning, and identifying random entrapment sequences would pollute that training.</p>
 *
 * <p>The library FASTA can reuse the training FASTA (no separate build) only when the two source
 * databases are the same file AND no entrapment is requested; otherwise the two FASTAs differ and
 * must be built separately.</p>
 *
 * <p>This pure decision is split out of {@code CarafeGUI.runCarafe()} so it can be unit-tested
 * without the Swing layer.</p>
 */
public final class OspreyFastaPlanner {

    private OspreyFastaPlanner() {
    }

    /** Immutable plan: whether to share one FASTA, and the entrapment flag for each FASTA build. */
    public static final class Plan {
        /** True when the library reuses the training FASTA instead of getting its own build. */
        public final boolean shareTrainingFasta;
        /** Entrapment flag for the training-DB FASTA build. Always {@code false}. */
        public final boolean trainingEntrapment;
        /** Entrapment flag for the library-DB FASTA build (only built when not shared). */
        public final boolean libraryEntrapment;

        public Plan(boolean shareTrainingFasta, boolean trainingEntrapment, boolean libraryEntrapment) {
            this.shareTrainingFasta = shareTrainingFasta;
            this.trainingEntrapment = trainingEntrapment;
            this.libraryEntrapment = libraryEntrapment;
        }
    }

    /**
     * Plan the OspreySharp peptide-FASTA builds.
     *
     * @param sameDb              whether the training and library databases are the same file
     * @param entrapmentRequested whether the user enabled entrapment (the "Include entrapment" box)
     * @return the build plan; the training FASTA is never given entrapment, the library FASTA gets
     *         it when requested, and the two are shared only when {@code sameDb} and no entrapment
     */
    public static Plan plan(boolean sameDb, boolean entrapmentRequested) {
        boolean share = sameDb && !entrapmentRequested;
        boolean libraryEntrapment = share ? false : entrapmentRequested;
        return new Plan(share, false, libraryEntrapment);
    }
}
