package main.java.db;

import java.util.Arrays;
import java.util.Map;

/**
 * Reject a candidate decoy or entrapment sequence whose theoretical b/y ladder is too close to
 * its target's, so the caller supplies another candidate instead.
 *
 * <p>This is a transcription of Osprey's {@code DecoyGenerator.IsCandidateAcceptable}
 * (C#: {@code pwiz_tools/Osprey/Osprey.Scoring/DecoyGenerator.cs}; Rust:
 * {@code crates/osprey-scoring/src/lib.rs}), which landed in both implementations on 2026-07-27
 * (pwiz #4480 / maccoss/osprey #58). Carafe builds the libraries Osprey actually searches, so
 * without this gate here the protection existed only on the path nobody uses.</p>
 *
 * <p><b>Why fragment overlap and not sequence identity.</b> Detection is driven by fragment
 * evidence, not positional string similarity. {@code EIVELEK}/{@code EEVEILK} has identity 0.571
 * but overlap 0.333 - positionally similar, ladders diverge, shadows nothing.
 * {@code LMDLIGDR}/{@code IMDLLGDR} has identity 0.750 but overlap 1.000 (isobaric L/I) - the
 * overlap gate catches it, identity nearly misses it. Measured over the pass-1 accepted entrapment
 * of two datasets, an identity gate added on top of this one caught 9 extra cases, all 9 with a
 * source target that was NOT accepted - zero demonstrated shadowing. Identity both misses real
 * harm and flags harmless cases, so it is deliberately not implemented.</p>
 *
 * <p>Computed from stripped sequences only - no modifications. Modifications shift both ladders
 * alike, so they cannot change whether the two coincide.</p>
 */
public final class DecoySimilarityGate {

    /**
     * Maximum fraction of a candidate's theoretical b/y ions that may fall within
     * {@link #LADDER_MATCH_TOLERANCE} of its target's. EncyclopeDIA's threshold
     * ({@code PeptideUtils.getSmartDecoy} rejects above 0.4 and reshuffles).
     */
    public static final double MAX_FRAGMENT_OVERLAP = 0.4;

    /**
     * Fixed m/z window for counting ladder coincidences, in daltons.
     *
     * <p>Deliberately NOT the run's fragment tolerance: the decoy set must be a pure function of
     * the library, and keying it to the search tolerance would make the same library produce
     * different decoys under {@code unit} vs {@code hram}. A fixed window also lets every
     * implementation apply the identical rule without plumbing search config into decoy
     * generation, which is what makes cross-impl parity possible.</p>
     */
    public static final double LADDER_MATCH_TOLERANCE = 0.02;

    private DecoySimilarityGate() {
    }

    /**
     * True when {@code candidateSeq} is far enough from {@code targetSeq} to be an honest null.
     *
     * <p>Deliberately the FULL ladder, matching EncyclopeDIA's rule exactly: the 0.4 threshold is
     * their published number, and it is only their number if measured over their statistic. Two
     * rungs are invariant under any C-terminus-preserving permutation (y1, and b_{n-1} whose
     * prefix multiset never changes), so they always match and impose a 1/(n-1) floor on the
     * ratio. At Carafe's 7-residue minimum peptide length the worst case is 1/6 = 0.167 against a
     * 0.4 budget. A much shorter peptide would floor near the threshold and see every candidate
     * rejected, silently dropping it - which is why the caller must treat "no acceptable
     * candidate" as a drop it counts and reports, never as a silent skip.</p>
     *
     * @param targetSeq    stripped target sequence
     * @param candidateSeq stripped candidate decoy or entrapment sequence
     * @return true when the candidate may be used
     */
    public static boolean isCandidateAcceptable(String targetSeq, String candidateSeq) {
        double[] targetLadder = theoreticalLadder(targetSeq);
        double[] candidateLadder = theoreticalLadder(candidateSeq);
        if (candidateLadder.length == 0) {
            return true;
        }
        Arrays.sort(targetLadder);
        int matches = 0;
        for (double mz : candidateLadder) {
            if (matchesWithinTolerance(targetLadder, mz)) {
                matches++;
            }
        }
        return (double) matches / candidateLadder.length <= MAX_FRAGMENT_OVERLAP;
    }

    /**
     * Fraction of {@code candidateSeq}'s ladder that coincides with {@code targetSeq}'s, in
     * [0, 1]. Exposed so audit tooling can REPORT the overlap distribution rather than only the
     * pass/fail verdict.
     */
    public static double fragmentOverlap(String targetSeq, String candidateSeq) {
        double[] targetLadder = theoreticalLadder(targetSeq);
        double[] candidateLadder = theoreticalLadder(candidateSeq);
        if (candidateLadder.length == 0) {
            return 0.0;
        }
        Arrays.sort(targetLadder);
        int matches = 0;
        for (double mz : candidateLadder) {
            if (matchesWithinTolerance(targetLadder, mz)) {
                matches++;
            }
        }
        return (double) matches / candidateLadder.length;
    }

    /**
     * Singly-charged b and y ion m/z for every cleavage site of a stripped sequence. Ions spanning
     * an unknown residue are skipped rather than aborting the ladder.
     *
     * <p>Uses prefix sums for b ions and SUFFIX sums for y ions. Deriving y from
     * (total - prefix) instead would poison EVERY y ion the moment any residue is unknown,
     * because total itself is then NaN - and a leading unknown residue would empty the ladder
     * outright, which the caller reads as "accept".</p>
     *
     * <p>Residue masses come from {@link EntrapmentFastaGear}'s table, which agrees with Osprey's
     * to 5 decimal places. That difference is 1e-5 Da against a 0.02 Da matching window - 2000x
     * smaller - so it cannot change any verdict, and sharing one table is worth more here than a
     * duplicated one that could drift.</p>
     */
    public static double[] theoreticalLadder(String sequence) {
        if (sequence == null || sequence.length() < 2) {
            return new double[0];
        }
        Map<Character, Double> masses = EntrapmentFastaGear.residueMasses();
        int len = sequence.length();

        double[] prefix = new double[len + 1];
        for (int i = 0; i < len; i++) {
            Double aa = masses.get(sequence.charAt(i));
            prefix[i + 1] = (aa == null || Double.isNaN(prefix[i])) ? Double.NaN : prefix[i] + aa;
        }
        double[] suffix = new double[len + 1];
        for (int i = len - 1; i >= 0; i--) {
            Double aa = masses.get(sequence.charAt(i));
            suffix[len - i] = (aa == null || Double.isNaN(suffix[len - i - 1]))
                    ? Double.NaN
                    : suffix[len - i - 1] + aa;
        }

        double[] ladder = new double[(len - 1) * 2];
        int n = 0;
        for (int ordinal = 1; ordinal < len; ordinal++) {
            double bMass = prefix[ordinal];
            if (!Double.isNaN(bMass)) {
                ladder[n++] = bMass + EntrapmentFastaGear.protonMass();
            }
            // y{ordinal} spans the last `ordinal` residues.
            double yMass = suffix[ordinal];
            if (!Double.isNaN(yMass)) {
                ladder[n++] = yMass + EntrapmentFastaGear.waterMass() + EntrapmentFastaGear.protonMass();
            }
        }
        return n == ladder.length ? ladder : Arrays.copyOf(ladder, n);
    }

    /**
     * True when {@code mz} is within {@link #LADDER_MATCH_TOLERANCE} of any entry of the sorted
     * {@code sortedLadder}. Binary search keeps the gate O(n log n) over a library rather than
     * O(n^2).
     */
    private static boolean matchesWithinTolerance(double[] sortedLadder, double mz) {
        int idx = Arrays.binarySearch(sortedLadder, mz);
        if (idx >= 0) {
            return true;
        }
        idx = -idx - 1;
        if (idx < sortedLadder.length && sortedLadder[idx] - mz <= LADDER_MATCH_TOLERANCE) {
            return true;
        }
        return idx > 0 && mz - sortedLadder[idx - 1] <= LADDER_MATCH_TOLERANCE;
    }
}
