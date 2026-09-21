package main.java.db;

import com.compomics.util.experiment.biology.enzymes.Enzyme;
import main.java.util.Cloger;
import net.sf.jfasta.FASTAElement;
import net.sf.jfasta.FASTAFileReader;
import net.sf.jfasta.impl.FASTAElementIterator;
import net.sf.jfasta.impl.FASTAFileReaderImpl;

import java.io.File;
import java.io.IOException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * A pool of foreign-species peptides used as entrapment sequences instead of shuffling the target.
 *
 * <p><b>Why.</b> A shuffled entrapment is an anagram of its own target: it shares the target's
 * exact residue composition and therefore many of its fragment masses, so it is over-identified
 * and the FDP it measures is too high. A real peptide from a phylogenetically distant species
 * (Arabidopsis is the standard choice) has no such relationship to the target while still being a
 * real, physically plausible peptide that is genuinely absent from a human sample.</p>
 *
 * <p><b>The matched assignment.</b> Each target is paired 1:1 with a UNIQUE foreign peptide of
 * near-identical neutral mass. Mass matching is what makes this a controlled comparison: in DIA a
 * library entry is only scored in the isolation window containing its precursor m/z, so an
 * entrapment set drawn without regard to mass would sample a different difficulty regime than the
 * targets and bias the FDP estimate.</p>
 *
 * <p><b>What the assignment optimizes, and why that choice matters.</b> The objective is to
 * MAXIMIZE THE NUMBER of pairs that co-locate - a threshold - not to minimize total mass
 * displacement. Those are different problems and they want different algorithms, which is easy to
 * get wrong:</p>
 * <ul>
 *   <li>Nearest-available in mass order accumulates a deficit (each target eats supply just above
 *       it) and every later target is dragged further off. Measured 81% co-location.</li>
 *   <li>A quantile map (rank i of n to rank i*m/n of m) is the optimal monotone transport and
 *       gives an excellent worst case, but it spreads its error evenly across every pair, so most
 *       pairs sit a few Da off. Optimal for total displacement, wrong for a threshold. Measured
 *       49%.</li>
 *   <li>Nearest-available in an order uncorrelated with mass gives most pairs a near-exact match
 *       and strands the few it serves last. Measured 95%.</li>
 * </ul>
 * <p>This implementation bins the pool finely and serves each target from the nearest non-empty
 * bin within the co-location window. Inside the window every candidate satisfies the objective
 * equally, so bins are consumed in O(1) with no search, and the preference for nearer bins keeps
 * the mass match tight as a secondary benefit.</p>
 *
 * <p>A target whose window is EMPTY falls through to an unbounded search, so it still gets an
 * entrapment peptide - just not a co-locating one, which is counted and reported rather than
 * hidden. A target whose window held candidates that all FAILED the similarity gate is a different
 * case and is dropped instead: widening the radius gives no reason to expect a better ladder
 * match, only a worse mass match. The two exhaustion modes deliberately do not share an
 * outcome.</p>
 *
 * <p><b>Determinism.</b> Bins are filled in mass-sorted order and consumed in a caller-fixed
 * order, so the same inputs always produce the same pairing.</p>
 */
public final class ForeignEntrapmentSource {

    /**
     * Neutral-mass window, in daltons, inside which an entrapment peptide is considered to
     * co-locate with its target. A 3 m/z DIA isolation window spans 6 Da at charge 2 and 9 Da at
     * charge 3; the tighter is used so a pair counted as co-locating does so at either charge.
     */
    public static final double CO_LOCATION_WINDOW_DA = 6.0;

    /** Bin width (Da). Fine enough that the nearest non-empty bin is a tight mass match. */
    private static final double BIN_WIDTH_DA = 0.25;

    /** Bins either side of the target's own bin that still count as co-locating. */
    private static final int CO_LOCATION_BINS = (int) ((CO_LOCATION_WINDOW_DA / 2.0) / BIN_WIDTH_DA);

    /** Candidates examined per target before giving up, when the gate keeps rejecting. */
    private static final int MAX_GATE_ATTEMPTS = 16;

    private final ArrayDeque<String>[] bins;
    private final double minMass;
    private int available;
    private int total;
    private int ilCollisionsDropped;

    @SuppressWarnings("unchecked")
    private ForeignEntrapmentSource(int nBins, double minMass) {
        this.bins = new ArrayDeque[nBins];
        this.minMass = minMass;
    }

    /** Number of distinct foreign peptides still available. */
    public int available() {
        return available;
    }

    /** Pool size before any assignment. */
    public int size() {
        return total;
    }

    /**
     * Foreign peptides excluded because they are I/L-isobaric to a real target. Reported because
     * an exact-string audit shows none of these, so without the count there is no visible evidence
     * the filter did anything.
     */
    public int ilCollisionsDropped() {
        return ilCollisionsDropped;
    }

    /**
     * Digest a foreign proteome into an assignment pool.
     *
     * @param fastaPath     foreign-species protein FASTA (e.g. Arabidopsis UP000006548)
     * @param enzyme        same enzyme used for the target digest
     * @param dbGear        digester (shares the global CParameter digest options)
     * @param excludedIl    real target sequences under {@link EntrapmentFastaGear#ilNormalize} - a
     *                      foreign peptide matching one of these is not absent from the sample, so
     *                      it cannot serve as entrapment. Normalised rather than exact because I
     *                      and L are isobaric: a foreign peptide differing from a human target only
     *                      by I&lt;-&gt;L is mass-identical to it with an identical fragment ladder,
     *                      and will be detected wherever that target is. This is the filter
     *                      foreign proteomes need most - plant and human share conserved proteins
     *                      whose tryptic peptides differ by exactly such conservative
     *                      substitutions, which an exact-string comparison reports as clean.
     * @param applyMzFilter whether to require the peptide fit the precursor m/z window
     */
    public static ForeignEntrapmentSource build(String fastaPath, Enzyme enzyme, DBGear dbGear,
                                                Set<String> excluded, Set<String> excludedIl,
                                                boolean applyMzFilter,
                                                int[] charges, double minMz, double maxMz)
            throws IOException {
        int proteins = 0;
        int droppedHomologous = 0;
        int droppedIlCollision = 0;
        int droppedUnknownAa = 0;
        int droppedOutOfMz = 0;
        Set<String> seen = new HashSet<>();
        List<String> keptSeqs = new ArrayList<>();
        List<Double> keptMasses = new ArrayList<>();
        double lo = Double.MAX_VALUE;
        double hi = -Double.MAX_VALUE;

        File dbFile = new File(fastaPath);
        FASTAFileReader reader = new FASTAFileReaderImpl(dbFile);
        try {
            FASTAElementIterator it = reader.getIterator();
            while (it.hasNext()) {
                FASTAElement el = it.next();
                el.setLineLength(1);
                String sequence = el.getSequence().replaceAll("\\s", "").toUpperCase();
                if (sequence.isEmpty()) {
                    continue;
                }
                proteins++;
                for (String pep : dbGear.digest_protein(enzyme, sequence)) {
                    if (!seen.add(pep)) {
                        continue;
                    }
                    if (excluded.contains(pep)) {
                        droppedHomologous++;
                        continue;
                    }
                    if (excludedIl.contains(EntrapmentFastaGear.ilNormalize(pep))) {
                        // Counted separately because this is the one an exact-string audit reports
                        // as clean: the peptide is not equal to any target, but is isobaric to one
                        // with the same fragment ladder, so it is detected wherever that target is.
                        droppedIlCollision++;
                        continue;
                    }
                    Double mass = EntrapmentFastaGear.peptideNeutralMass(pep);
                    if (mass == null) {
                        droppedUnknownAa++;
                        continue;
                    }
                    if (applyMzFilter
                            && !EntrapmentFastaGear.fitsMzRange(mass, charges, minMz, maxMz)) {
                        droppedOutOfMz++;
                        continue;
                    }
                    keptSeqs.add(pep);
                    keptMasses.add(mass);
                    lo = Math.min(lo, mass);
                    hi = Math.max(hi, mass);
                }
            }
        } finally {
            reader.close();
        }

        if (keptSeqs.isEmpty()) {
            throw new IOException("Foreign entrapment FASTA yielded no usable peptides: " + fastaPath);
        }
        int nBins = (int) ((hi - lo) / BIN_WIDTH_DA) + 2;
        ForeignEntrapmentSource pool = new ForeignEntrapmentSource(nBins, lo);
        // Fill in mass-sorted order so each bin's contents are deterministic and, within a bin,
        // ordered - which keeps a draw's mass error at the low end of the bin where possible.
        Integer[] order = new Integer[keptSeqs.size()];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        java.util.Arrays.sort(order, (a, b) -> {
            int c = Double.compare(keptMasses.get(a), keptMasses.get(b));
            return c != 0 ? c : keptSeqs.get(a).compareTo(keptSeqs.get(b));
        });
        for (Integer idx : order) {
            pool.add(keptSeqs.get(idx), keptMasses.get(idx));
        }

        pool.ilCollisionsDropped = droppedIlCollision;
        Cloger.getInstance().logger.info(String.format(
                "Foreign entrapment source: %d proteins -> %d unique candidate peptides "
                        + "(%d dropped equal to a real target, %d dropped I/L-isobaric to a real "
                        + "target, %d unknown AA, %d out of m/z range)",
                proteins, pool.total, droppedHomologous, droppedIlCollision, droppedUnknownAa,
                droppedOutOfMz));
        return pool;
    }

    private void add(String pep, double mass) {
        int b = binOf(mass);
        if (bins[b] == null) {
            bins[b] = new ArrayDeque<>();
        }
        bins[b].add(pep);
        available++;
        total++;
    }

    private int binOf(double mass) {
        int b = (int) ((mass - minMass) / BIN_WIDTH_DA);
        if (b < 0) {
            return 0;
        }
        return Math.min(b, bins.length - 1);
    }

    /**
     * Assign a unique foreign peptide to {@code targetSeq}, preferring one that co-locates with it.
     *
     * @return the assigned peptide, or {@code null} if the pool is exhausted
     */
    public String assign(String targetSeq, double targetMass) {
        // Nothing left to serve. Without this the two scans below still walk every bin in the
        // pool, per target, finding them all empty - and an exhausted pool is an ANTICIPATED
        // state (build() warns when the pool is smaller than the selected target set), so on a
        // 1.4M-target build that is minutes of pure no-op scanning.
        if (available == 0) {
            return null;
        }
        // One gate budget shared across both scans, and candidates rejected by the first are held
        // aside rather than returned to the pool between them. Restarting the scan with a fresh
        // budget - as this used to - re-polled the same near neighbours, spent the budget on them
        // again, and returned null before ever widening, so the "unbounded" fallback below could
        // not actually reach past the co-location window.
        int[] budget = { MAX_GATE_ATTEMPTS };
        List<String> putBack = new ArrayList<>();
        try {
            String hit = search(targetSeq, targetMass, 0, CO_LOCATION_BINS, budget, putBack);
            if (hit != null) {
                return hit;
            }
            if (budget[0] <= 0) {
                // The window had candidates; they all failed the gate. Widening cannot help within
                // any sane bound, so drop the target - counted and reported by the caller.
                return null;
            }
            // The window was genuinely EMPTY. Take the nearest available anywhere rather than
            // leaving the target unpaired; the mass match is then poor, which shows up in the
            // reported co-location rate rather than being hidden.
            return search(targetSeq, targetMass, CO_LOCATION_BINS + 1, bins.length, budget, putBack);
        } finally {
            // Rejected candidates go back: a different target with a different ladder can use them.
            for (String s : putBack) {
                Double m = EntrapmentFastaGear.peptideNeutralMass(s);
                if (m != null) {
                    int b = binOf(m);
                    if (bins[b] == null) {
                        bins[b] = new ArrayDeque<>();
                    }
                    bins[b].add(s);
                    available++;
                }
            }
        }
    }

    /**
     * Nearest non-empty bin at a radius in {@code [minRadius, maxRadius]}, serving one peptide that
     * passes the gate.
     *
     * @param budget  single-element box holding the remaining gate-rejection allowance, decremented
     *                in place so a caller can continue one budget across successive radius ranges
     * @param putBack accumulates gate-rejected candidates; the CALLER returns them to the pool, so
     *                they are not re-polled by a later scan in the same assignment
     */
    private String search(String targetSeq, double targetMass, int minRadius, int maxRadius,
                          int[] budget, List<String> putBack) {
        int home = binOf(targetMass);
        for (int radius = minRadius; radius <= maxRadius; radius++) {
            for (int sign = 0; sign < (radius == 0 ? 1 : 2); sign++) {
                int b = home + (sign == 0 ? radius : -radius);
                if (b < 0 || b >= bins.length || bins[b] == null || bins[b].isEmpty()) {
                    continue;
                }
                while (!bins[b].isEmpty() && budget[0] > 0) {
                    String candidate = bins[b].poll();
                    available--;
                    if (DecoySimilarityGate.isCandidateAcceptable(targetSeq, candidate)) {
                        return candidate;
                    }
                    putBack.add(candidate);
                    budget[0]--;
                }
                if (budget[0] <= 0) {
                    return null;
                }
            }
        }
        return null;
    }
}
