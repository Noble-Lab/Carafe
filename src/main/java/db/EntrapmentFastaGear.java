package main.java.db;

import com.compomics.util.experiment.biology.enzymes.Enzyme;
import com.compomics.util.experiment.biology.ions.impl.ElementaryIon;
import com.compomics.util.experiment.biology.proteins.Peptide;
import main.java.input.CParameter;
import main.java.input.PeptideUtils;
import main.java.util.Cloger;
import net.sf.jfasta.FASTAElement;
import net.sf.jfasta.FASTAFileReader;
import net.sf.jfasta.impl.FASTAElementIterator;
import net.sf.jfasta.impl.FASTAFileReaderImpl;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/**
 * Build a peptide-level FASTA + FDRBench-style pairing manifest from a protein FASTA.
 *
 * <p>This is a Java port of the lab's {@code build_entrapment_peptide_fasta.py}. For each
 * tryptic peptide it can optionally also emit an entrapment ({@code p_target}) peptide, a
 * {@code decoy}, and a {@code p_decoy}. The entrapment ({@code p_target}) sequence is a
 * deterministic shuffle of the target (C-terminus preserved). The decoys ({@code decoy} /
 * {@code p_decoy}) are generated the same way Osprey's {@code DecoyGenerator} does: reverse the
 * peptide (C-terminus preserved), and cycle the internal residues to avoid colliding with a real
 * target. A quartet is dropped when its entrapment shuffle collides with a real target or when a
 * requested reverse decoy cannot be made unique (FDRBench collision-drop policy).</p>
 *
 * <p>The output FASTA has one entry per peptide whose accession is the source protein accession
 * (optionally with a per-source {@code _pepNNNNN} counter so library predictors that dedupe by
 * accession - e.g. Carafe - accept every entry). Carafe is then run over this FASTA in
 * {@code NoCut} mode so AlphaPepDeep predicts spectra/RT for every target and decoy sequence,
 * giving Osprey a library with both target and decoy entries.</p>
 *
 * <p>Digestion reuses {@link DBGear#digest_protein(Enzyme, String)} with the global
 * {@link CParameter} digest options (enzyme, missed cleavages, length, clip N-term M) so the
 * peptide set matches the GUI's configured digestion. The m/z filter and shuffle constants are
 * ported from the Python script.</p>
 *
 * <p><b>Shuffle note:</b> the per-peptide RNG seed is derived as SHA-1({@code master_seed:seq})
 * exactly as in the Python script, but the shuffle itself uses {@link java.util.Random}'s
 * Fisher-Yates rather than CPython's Mersenne-Twister {@code random.shuffle}. The shuffled
 * sequences therefore differ from the Python tool's, but they are deterministic and
 * self-consistent: the manifest always matches the FASTA, which is all Osprey requires.</p>
 */
public class EntrapmentFastaGear {

    /** Header marker suffix for entrapment proteins (matches FDRBench convention). */
    public static final String DEFAULT_ENTRAPMENT_SUFFIX = "_p_target";
    /** Prefix on the FASTA header that flags decoy entries. */
    public static final String DEFAULT_DECOY_PREFIX = "decoy_";

    /** Monoisotopic residue masses (Da). B/Z/J/X/U/O are ambiguity codes - peptides containing
     *  them are skipped silently. */
    private static final Map<Character, Double> AA_MONO_MASS = new LinkedHashMap<>();
    private static final double H2O_MONO = 18.01056;
    private static final double PROTON_MONO = 1.007276;

    static {
        AA_MONO_MASS.put('G', 57.02146);
        AA_MONO_MASS.put('A', 71.03711);
        AA_MONO_MASS.put('S', 87.03203);
        AA_MONO_MASS.put('P', 97.05276);
        AA_MONO_MASS.put('V', 99.06841);
        AA_MONO_MASS.put('T', 101.04768);
        AA_MONO_MASS.put('C', 103.00919);
        AA_MONO_MASS.put('L', 113.08406);
        AA_MONO_MASS.put('I', 113.08406);
        AA_MONO_MASS.put('N', 114.04293);
        AA_MONO_MASS.put('D', 115.02694);
        AA_MONO_MASS.put('Q', 128.05858);
        AA_MONO_MASS.put('K', 128.09496);
        AA_MONO_MASS.put('E', 129.04259);
        AA_MONO_MASS.put('M', 131.04049);
        AA_MONO_MASS.put('H', 137.05891);
        AA_MONO_MASS.put('F', 147.06841);
        AA_MONO_MASS.put('R', 156.10111);
        AA_MONO_MASS.put('Y', 163.06333);
        AA_MONO_MASS.put('W', 186.07931);
    }

    /**
     * Smallest entrapment ratio that has actually been characterised.
     *
     * <p>The ratio sweep this bound comes from ran r = 1, 0.5, 0.25 and 0.1. The combined
     * estimator held steady at ~1.1% FDP across that whole range, which is the result that makes
     * a thin overlay usable - but power falls with the pool, and at r = 0.1 only 32 entrapment
     * peptides were accepted at 1% q. Below that the estimate is dominated by counting noise, and
     * nobody has measured where it stops being meaningful.</p>
     *
     * <p>So this is a deliberate floor on an UNTESTED region, not a statement that lower values
     * cannot work. If a real case needs one, widen it on purpose and record what was measured.</p>
     */
    public static final double MIN_ENTRAPMENT_RATIO = 0.1;

    /** Configuration for a peptide-FASTA build. Defaults match the Python script. */
    public static class Config {
        public String inputFasta;
        public String outputFasta;
        public String manifest = null; // optional
        public boolean addEntrapment = false;
        public boolean addDecoys = true;
        /** Apply the precursor m/z window filter at the configured charges. */
        public boolean applyMzFilter = false;
        public double minMz = 400.0;
        public double maxMz = 900.0;
        public int[] charges = {2, 3};
        public long entrapmentSeed = 42;
        public long decoySeed = 24;
        /**
         * Foreign-species protein FASTA to draw entrapment peptides from. When null (the default)
         * entrapment is a deterministic shuffle of each target, which is what Carafe has always
         * done. When set, entrapment peptides are real foreign peptides mass-matched 1:1 to the
         * targets - see {@link ForeignEntrapmentSource} for why that measures FDP more honestly.
         */
        public String entrapmentSourceFasta = null;
        /**
         * Fraction of targets that carry an entrapment peptide, in
         * [{@value #MIN_ENTRAPMENT_RATIO}, 1.0]. 1.0 pairs every target. A smaller value makes
         * entrapment a thin overlay that barely perturbs the target search, which measures FDP
         * without distorting it - the estimator is ratio-aware, so the reported FDP stays
         * comparable.
         */
        public double entrapmentRatio = 1.0;
        /**
         * Apply {@link DecoySimilarityGate} to generated entrapment and decoy sequences.
         *
         * <p>An AUDIT SWITCH, not a tuning knob. Turning it off reproduces the pre-gate behaviour
         * (one shuffle, exact-collision rejection only), which is how a rebuilt library is proved
         * to differ from an older one ONLY by the gate rather than by some unnoticed change in
         * digest parameters. A library built with this off has the near-copy contamination the
         * gate exists to remove and should not be searched.</p>
         */
        public boolean similarityGate = true;
        /**
         * Refuse to write a manifest that fails {@link EntrapmentPairingValidator#validateQuartets}.
         *
         * <p>Default true: a broken pairing is cheaper to fix here than downstream, where it either
         * crashes a paired FDP estimator or silently mis-scales one. Set false (CLI:
         * {@code -ignore_pairing_errors}) to downgrade the refusal to a logged warning and write the
         * manifest anyway.</p>
         *
         * <p>The escape hatch exists because these invariants can only really be confirmed at a
         * scale the unit tests cannot reach - the checks are asserted against small fixtures, but a
         * real build is 1.4M quartets. Without it, a check that turns out to be wrong about what
         * the generator guarantees leaves no way to produce a manifest at all until the code is
         * patched. Deliberately NOT folded into {@code -no_similarity_gate}: that flag changes which
         * sequences are GENERATED and yields a library that should not be searched, so it is not a
         * neutral override for a pairing check.</p>
         */
        public boolean failOnPairingViolation = true;
        public String entrapmentSuffix = DEFAULT_ENTRAPMENT_SUFFIX;
        public String decoyPrefix = DEFAULT_DECOY_PREFIX;
        public boolean uniqueAccessions = true;
        public String pepSuffixFormat = "_pep%05d";
    }

    /** A single source protein parsed from the input FASTA. */
    static final class ProteinRecord {
        final String accession;
        final String entryName;
        final String db;
        final String sequence;

        ProteinRecord(String accession, String entryName, String db, String sequence) {
            this.accession = accession;
            this.entryName = entryName;
            this.db = db;
            this.sequence = sequence;
        }
    }

    /** The four sequences derived from a single target peptide plus its source proteins. */
    static final class Quartet {
        final String target;
        String pTarget = null;
        String decoy = null;
        String pDecoy = null;
        /** True when this target was chosen to carry entrapment (all of them unless ratio < 1).
         *  Distinguishes "deliberately has none" from "none could be generated", which must be
         *  dropped rather than silently emitted as a bare target+decoy pair. */
        boolean entrapmentSelected = false;
        final List<ProteinRecord> sources = new ArrayList<>();

        Quartet(String target) {
            this.target = target;
        }
    }

    /** Summary counts from a build, returned for logging/testing. */
    public static final class Result {
        public int proteins;
        public int uniqueTargets;
        public int droppedUnknownAa;
        public int droppedOutOfMz;
        public int quartetsBuilt;
        public int quartetsDropped;
        public int keptQuartets;
        public int targetEntries;
        public int pTargetEntries;
        public int decoyEntries;
        public int pDecoyEntries;
        public int sharedEntries;
        /** Targets with no entrapment sequence that passes the similarity gate (quartet dropped). */
        public int droppedNoEntrapment;
        /** Targets with no decoy that passes the similarity gate (quartet dropped). */
        public int droppedNoDecoy;
        /** Entrapment peptides drawn from a foreign proteome rather than shuffled. */
        public int entrapmentFromForeign;
        /** Targets deliberately left without entrapment because {@code entrapmentRatio} < 1. */
        public int entrapmentNotSelected;
        /** Median |target - entrapment| neutral mass (Da) for a foreign-source assignment. */
        public double entrapmentMedianAbsMassDelta;
        /** Fraction of foreign entrapment peptides co-locating with their target's window. */
        public double entrapmentInIsolationWindowFraction;
        /** Foreign candidates excluded for being I/L-isobaric to a real target. */
        public int entrapmentIlCollisionsDropped;
    }

    /**
     * Isoleucine-to-leucine normalised form of a sequence.
     *
     * <p>I and L have identical residue masses (113.08406). Two sequences differing only by
     * I&lt;-&gt;L substitutions are therefore mass-identical AND produce identical b/y ladders -
     * mass spectrometry cannot tell them apart at all. A generated entrapment or decoy that is
     * I/L-identical to a real target is not a model of a false discovery: it will be detected
     * wherever its twin is present, and every such detection is counted as false when it is not.
     *
     * <p>Comparing normalised forms subsumes exact string comparison, since a sequence normalises
     * to itself when it contains no isoleucine.</p>
     */
    public static String ilNormalize(String seq) {
        return seq.indexOf('I') < 0 ? seq : seq.replace('I', 'L');
    }

    /** {@link #ilNormalize} applied across a set, for the one-off build of a collision index. */
    public static Set<String> ilNormalizedSet(Set<String> sequences) {
        Set<String> out = new HashSet<>(Math.max(16, (int) (sequences.size() / 0.7f) + 1));
        for (String s : sequences) {
            out.add(ilNormalize(s));
        }
        return out;
    }

    /** Monoisotopic residue masses, for {@link DecoySimilarityGate}'s theoretical ladder. Shared
     *  rather than duplicated so the two cannot drift apart. */
    static Map<Character, Double> residueMasses() {
        return AA_MONO_MASS;
    }

    /** Proton mass (Da), for {@link DecoySimilarityGate}. */
    static double protonMass() {
        return PROTON_MONO;
    }

    /** Water mass (Da), for {@link DecoySimilarityGate}. */
    static double waterMass() {
        return H2O_MONO;
    }

    /** Monoisotopic neutral mass of a peptide, or {@code null} if any residue is not in
     *  {@link #AA_MONO_MASS} (e.g. contains B/Z/J/X/U/O). */
    public static Double peptideNeutralMass(String seq) {
        double total = H2O_MONO;
        for (int i = 0; i < seq.length(); i++) {
            Double m = AA_MONO_MASS.get(seq.charAt(i));
            if (m == null) {
                return null;
            }
            total += m;
        }
        return total;
    }

    /** True iff at least one allowed charge state produces an m/z inside [minMz, maxMz]. */
    public static boolean fitsMzRange(double neutralMass, int[] charges, double minMz, double maxMz) {
        for (int z : charges) {
            double mz = (neutralMass + z * PROTON_MONO) / z;
            if (minMz <= mz && mz <= maxMz) {
                return true;
            }
        }
        return false;
    }

    /**
     * Modification-aware m/z window test, matching the library-prediction step's per-precursor
     * filter exactly. Enumerates the same peptidoforms the library does via
     * {@link PeptideUtils#calcPeptideIsoforms(String)} (fixed modifications always applied, variable
     * modifications up to {@code maxVarMods}/{@code maxModsPerAA}) and returns true iff at least one
     * {@code (peptidoform, charge)} precursor m/z lands inside {@code [minMz, maxMz]} — the same
     * "does this peptide contribute any precursor to the library" test. Uses the identical proton
     * mass ({@link ElementaryIon#proton}) and compomics-computed modified masses as
     * {@code AIGear.get_mz}, so a Carbamidomethyl-C peptide is filtered on its +57.02 mass and a
     * peptide that only qualifies through an Oxidation-M form is retained. The caller must first drop
     * unknown-residue peptides (via {@link #peptideNeutralMass}) so compomics only sees standard
     * residues here.
     */
    public static boolean fitsMzRangeWithMods(String seq, int[] charges, double minMz, double maxMz) {
        List<Peptide> peptidoforms = PeptideUtils.calcPeptideIsoforms(seq);
        double proton = ElementaryIon.proton.getTheoreticMass();
        for (Peptide form : peptidoforms) {
            double mass = form.getMass();
            for (int z : charges) {
                double mz = (mass + z * proton) / z;
                if (minMz <= mz && mz <= maxMz) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * Deterministically shuffle all but the last residue. The per-peptide RNG seed is derived
     * as SHA-1({@code masterSeed:seq}) so the same (sequence, masterSeed) pair always returns the
     * same shuffle while different sequences get independent shuffles. Length-1 and length-2
     * inputs are returned unchanged.
     */
    public static String shufflePreservingCterm(String seq, long masterSeed) {
        return shufflePreservingCterm(seq, masterSeed, 0);
    }

    /**
     * Attempt-indexed variant used by {@link #generateShuffledEntrapment}. Attempt 0 reproduces
     * the single-argument form exactly, so only peptides the gate actually rejects get a different
     * entrapment; see {@link #derivePepSeed} for why that is deliberate and when it could be
     * dropped.
     */
    public static String shufflePreservingCterm(String seq, long masterSeed, int attempt) {
        if (seq.length() <= 2) {
            return seq;
        }
        char last = seq.charAt(seq.length() - 1);
        List<Character> body = new ArrayList<>(seq.length() - 1);
        for (int i = 0; i < seq.length() - 1; i++) {
            body.add(seq.charAt(i));
        }
        long pepSeed = derivePepSeed(masterSeed, seq, attempt);
        Random rng = new Random(pepSeed);
        // Fisher-Yates so the shuffle is fully determined by the seed.
        for (int i = body.size() - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            Character tmp = body.get(i);
            body.set(i, body.get(j));
            body.set(j, tmp);
        }
        StringBuilder sb = new StringBuilder(seq.length());
        for (char c : body) {
            sb.append(c);
        }
        sb.append(last);
        return sb.toString();
    }

    /**
     * First 8 bytes of SHA-1("{@code masterSeed:seq}") - or "{@code masterSeed:seq:attempt}" for
     * retries - as an unsigned-ish long seed.
     *
     * <p><b>Attempt 0 deliberately keeps the original two-part form</b> rather than uniformly
     * appending "{@code :0}", so a peptide whose first shuffle already passes the gate keeps the
     * exact entrapment sequence it had before the gate existed. Two things depend on that:</p>
     * <ul>
     *   <li>{@code -no_similarity_gate} reproduces a pre-gate library <b>byte for byte</b>. With
     *       the gate off the loop runs exactly one attempt, so attempt 0's derivation IS the
     *       reproduction. That is what proves a rebuild differs from a delivered library only by
     *       the gate, rather than by some unnoticed change in digest parameters - verified on
     *       Astral as an identical SHA-256 over the 349 MB peptide FASTA and all 1,390,979
     *       manifest quartets.</li>
     *   <li>Comparing measured FDP before and after the gate is then almost a PAIRED comparison:
     *       only the ~4% of entrapment the gate actually rejects changes, so the difference is
     *       the gate's effect and not the sampling noise of a fresh 1.39M-peptide draw.</li>
     * </ul>
     *
     * <p><b>This is not merely legacy compatibility.</b> The first property is a permanent
     * regression oracle: "does this generator still reproduce the delivered library exactly?"
     * catches any future drift in digestion or generation, and it keeps working long after the
     * migration that motivated it. Collapsing to a uniform "{@code :attempt}" would spend that
     * oracle to remove a special case, and would also change every entrapment sequence in one
     * step. Both derivations are equally deterministic - neither reads a clock, and the same
     * inputs give the same output on every run and every machine - so uniformity would buy
     * tidiness, not reproducibility. A deliberately different draw is already available by
     * changing {@code -entrapment_seed}, which is the only thing that should ever move it.</p>
     *
     * <p><b>Reviewer note.</b> The one real cost is that a reimplementation must not miss the
     * special case: doing so silently produces a different library rather than an error. If that
     * ever bites, prefer making the legacy derivation explicit (a named constant, or a documented
     * "attempt 0 is the compatibility seed" branch) over removing it.</p>
     */
    private static long derivePepSeed(long masterSeed, String seq, int attempt) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-1");
            String material = attempt == 0
                    ? masterSeed + ":" + seq
                    : masterSeed + ":" + seq + ":" + attempt;
            byte[] digest = md.digest(material.getBytes(StandardCharsets.UTF_8));
            long s = 0L;
            for (int i = 0; i < 8; i++) {
                s = (s << 8) | (digest[i] & 0xffL);
            }
            return s;
        } catch (NoSuchAlgorithmException e) {
            // SHA-1 is guaranteed present on every JVM; fall back to the string hash if not.
            // Mixing in the attempt keeps retries distinct on that path too.
            return ((long) seq.hashCode() << 16) ^ masterSeed ^ ((long) attempt * 0x9E3779B97F4A7C15L);
        }
    }

    /**
     * Maximum shuffles tried before a peptide is declared to have no acceptable entrapment.
     *
     * <p>Chosen so that a peptide with genuine permutational freedom effectively never exhausts
     * it, while a peptide with none - {@code AAAAAAAAAAAAAAAAGATCLER}, 17 alanines, where every
     * permutation is the same sequence - fails fast. Dropping that peptide is the correct answer:
     * no permutation of 17 alanines is a valid entrapment.</p>
     */
    public static final int MAX_ENTRAPMENT_SHUFFLE_ATTEMPTS = 20;

    /**
     * Generate an entrapment peptide by shuffling the target, retrying until the result is
     * distinct, collides with no real target, and passes {@link DecoySimilarityGate}. Returns
     * {@code null} when no acceptable shuffle exists, so the caller drops the quartet.
     *
     * <p>Before this existed the shuffle was called ONCE with no retry, and the only rejection was
     * an exact string collision with a real target. That let an entrapment land close enough to
     * its own target to be detected wherever the target is - measured at ~27% of the ACCEPTED
     * entrapment set on two datasets, of which roughly half shadowed a target that was itself
     * accepted, co-eluting within 0.05 min. Those are not false peptides at all, but the FDP
     * estimator counts them as such, inflating measured FDP by ~25%.</p>
     */
    public static String generateShuffledEntrapment(String seq, long masterSeed, Set<String> targetSet) {
        return generateShuffledEntrapment(seq, masterSeed, targetSet, ilNormalizedSet(targetSet), true);
    }

    /**
     * As {@link #generateShuffledEntrapment(String, long, Set)}, with {@code gate} false
     * reproducing the pre-fix behaviour: a single shuffle rejected only on an EXACT collision, with
     * neither the similarity gate nor I/L-normalised collision rejection. For audit builds only -
     * see {@link Config#similarityGate}.
     *
     * @param targetSet   real target sequences, for the legacy exact-collision check
     * @param targetSetIl the same set under {@link #ilNormalize}, built once by the caller
     */
    public static String generateShuffledEntrapment(String seq, long masterSeed,
                                                    Set<String> targetSet, Set<String> targetSetIl,
                                                    boolean gate) {
        int attempts = gate ? MAX_ENTRAPMENT_SHUFFLE_ATTEMPTS : 1;
        for (int attempt = 0; attempt < attempts; attempt++) {
            String candidate = shufflePreservingCterm(seq, masterSeed, attempt);
            if (candidate.equals(seq)) {
                continue;
            }
            if (!collides(candidate, targetSet, targetSetIl, gate)
                    && (!gate || DecoySimilarityGate.isCandidateAcceptable(seq, candidate))) {
                return candidate;
            }
        }
        return null;
    }

    /**
     * Reverse all but the last residue (C-terminus preserved), matching Osprey's
     * {@code DecoyGenerator.ReverseSequence} for tryptic peptides. Length-1 and length-2 inputs
     * are returned unchanged.
     */
    public static String reversePreservingCterm(String seq) {
        int len = seq.length();
        if (len <= 2) {
            return seq;
        }
        StringBuilder sb = new StringBuilder(len);
        for (int i = len - 2; i >= 0; i--) {
            sb.append(seq.charAt(i));
        }
        sb.append(seq.charAt(len - 1));
        return sb.toString();
    }

    /**
     * Cyclically rotate the internal residues (C-terminus preserved) by {@code cycleLength},
     * matching Osprey's {@code DecoyGenerator.CycleSequence} (e.g. {@code ABCDEK -> BCDEAK} for
     * cycleLength 1). Length-1 and length-2 inputs are returned unchanged.
     */
    public static String cyclePreservingCterm(String seq, int cycleLength) {
        int len = seq.length();
        if (len <= 2 || cycleLength == 0) {
            return seq;
        }
        int middleLen = len - 1;
        int effectiveCycle = cycleLength % middleLen;
        StringBuilder sb = new StringBuilder(len);
        for (int i = 0; i < middleLen; i++) {
            sb.append(seq.charAt((i + effectiveCycle) % middleLen));
        }
        sb.append(seq.charAt(len - 1));
        return sb.toString();
    }

    /**
     * Generate a decoy the way Osprey does: reverse the peptide (C-terminus preserved); if the
     * reversal equals the original or collides with a real target, cycle the internal residues by
     * 1..min(len, 10) until the result is unique. Returns {@code null} when no unique decoy can be
     * produced (the caller drops that target-decoy pair, as Osprey does).
     */
    public static String generateReverseDecoy(String seq, Set<String> targetSet) {
        return generateReverseDecoy(seq, targetSet, ilNormalizedSet(targetSet), true);
    }

    /**
     * As {@link #generateReverseDecoy(String, Set)}, with {@code gate} false reproducing the
     * pre-fix behaviour (exact-uniqueness rejection only). For audit builds only - see
     * {@link Config#similarityGate}.
     *
     * @param targetSet   sequences the decoy must not collide with, for the legacy exact check
     * @param targetSetIl the same set under {@link #ilNormalize}, built once by the caller
     */
    public static String generateReverseDecoy(String seq, Set<String> targetSet,
                                              Set<String> targetSetIl, boolean gate) {
        String rev = reversePreservingCterm(seq);
        if (!rev.equals(seq) && !collides(rev, targetSet, targetSetIl, gate)
                && (!gate || DecoySimilarityGate.isCandidateAcceptable(seq, rev))) {
            return rev;
        }
        int maxRetries = Math.min(seq.length(), 10);
        for (int c = 1; c <= maxRetries; c++) {
            String cyc = cyclePreservingCterm(seq, c);
            if (!cyc.equals(seq) && !collides(cyc, targetSet, targetSetIl, gate)
                    && (!gate || DecoySimilarityGate.isCandidateAcceptable(seq, cyc))) {
                return cyc;
            }
        }
        return null;
    }

    /**
     * True when a generated sequence is indistinguishable from something on the target side.
     * Gated builds compare I/L-normalised forms, which subsumes exact equality; legacy audit
     * builds keep the exact comparison so they still reproduce pre-fix output.
     */
    private static boolean collides(String candidate, Set<String> targetSet,
                                    Set<String> targetSetIl, boolean gate) {
        return gate ? targetSetIl.contains(ilNormalize(candidate)) : targetSet.contains(candidate);
    }

    /** Build the {@code db|accession|entry} label for a peptide kind. */
    static String buildProteinLabel(ProteinRecord rec, boolean pTarget, boolean decoy,
                                    String entrapSuffix, String decoyPrefix,
                                    Integer pepCounter, String pepSuffixFormat) {
        String acc = pTarget ? rec.accession + entrapSuffix : rec.accession;
        if (pepCounter != null) {
            acc = acc + String.format(pepSuffixFormat, pepCounter);
        }
        String entry = pTarget ? rec.entryName + entrapSuffix : rec.entryName;
        String label = rec.db + "|" + acc + "|" + entry;
        if (decoy) {
            label = decoyPrefix + label;
        }
        return label;
    }

    /**
     * Run a full peptide-FASTA + manifest build.
     *
     * @param cfg build configuration
     * @return summary counts
     * @throws IOException on read/write failure
     */
    public static Result run(Config cfg) throws IOException {
        if (cfg.minMz >= cfg.maxMz) {
            throw new IllegalArgumentException("invalid m/z range: " + cfg.minMz + "-" + cfg.maxMz);
        }
        for (int z : cfg.charges) {
            if (z < 1 || z > 10) {
                throw new IllegalArgumentException("invalid charge state (allowed 1..10): " + z);
            }
        }

        Cloger.getInstance().logger.info("Reading FASTA: " + cfg.inputFasta);
        DBGear.init_enzymes();
        Enzyme enzyme = DBGear.getEnzymeByIndex(CParameter.enzyme);
        DBGear dbGear = new DBGear();

        // Step 1: parse + digest every protein, collecting unique target peptides with their
        // source proteins. Apply residue (unknown AA) and optional m/z filters at digest time.
        Map<String, List<ProteinRecord>> targetToSources = new LinkedHashMap<>();
        Result r = new Result();

        File dbFile = new File(cfg.inputFasta);
        FASTAFileReader reader = new FASTAFileReaderImpl(dbFile);
        try {
            FASTAElementIterator it = reader.getIterator();
            while (it.hasNext()) {
                FASTAElement el = it.next();
                el.setLineLength(1);
                ProteinRecord rec = parseHeader(el.getHeader(), cleanSeq(el.getSequence()));
                if (rec == null) {
                    continue;
                }
                r.proteins++;
                Set<String> peptides = dbGear.digest_protein(enzyme, rec.sequence);
                for (String pep : peptides) {
                    List<ProteinRecord> existing = targetToSources.get(pep);
                    if (existing != null) {
                        existing.add(rec);
                        continue;
                    }
                    // Drop unknown-residue peptides first (B/Z/J/X/U/O) so the modification-aware m/z
                    // filter below only hands standard residues to compomics.
                    Double neutralMass = peptideNeutralMass(pep);
                    if (neutralMass == null) {
                        r.droppedUnknownAa++;
                        continue;
                    }
                    // Select peptides by the same modification-aware precursor m/z window the library
                    // prediction uses, so the FASTA's target set matches what the library will contain.
                    if (cfg.applyMzFilter
                            && !fitsMzRangeWithMods(pep, cfg.charges, cfg.minMz, cfg.maxMz)) {
                        r.droppedOutOfMz++;
                        continue;
                    }
                    List<ProteinRecord> sources = new ArrayList<>();
                    sources.add(rec);
                    targetToSources.put(pep, sources);
                }
            }
        } finally {
            reader.close();
        }
        r.uniqueTargets = targetToSources.size();
        Cloger.getInstance().logger.info(String.format(
                "Digested %d proteins: %d unique target peptides retained, %d dropped (unknown AA), "
                        + "%d dropped (out of m/z range)",
                r.proteins, r.uniqueTargets, r.droppedUnknownAa, r.droppedOutOfMz));

        // Step 2: build quartets in two passes, matching how Osprey would build decoys over a
        // library that contains both targets and entrapments. Pass 1 adds the deterministic-shuffle
        // entrapments (p_target). Pass 2 adds the decoys (decoy / p_decoy) the Osprey way: reverse
        // with the C-terminus preserved, cycling the internal residues by 1..min(len, 10) until the
        // result is unique against the combined target+entrapment set (entrapments are targets in
        // Osprey's world, so decoys must dodge them too). Like Osprey, decoys are NOT checked against
        // other decoys. A target whose decoy cannot be made unique yields a null decoy, dropped below.
        Set<String> targetSet = targetToSources.keySet();
        // Built once and reused for every candidate. I and L are isobaric, so a generated sequence
        // whose I->L normalised form matches a real target is mass-identical to it with an
        // identical fragment ladder - it will be detected wherever that target is, which makes it
        // a genuinely present peptide rather than a model of a false one.
        Set<String> targetSetIl = ilNormalizedSet(targetSet);
        List<Quartet> quartets = new ArrayList<>(targetToSources.size());
        for (Map.Entry<String, List<ProteinRecord>> e : targetToSources.entrySet()) {
            Quartet q = new Quartet(e.getKey());
            q.sources.addAll(e.getValue());
            quartets.add(q);
        }
        if (cfg.addEntrapment) {
            assignEntrapment(cfg, quartets, targetSet, targetSetIl, enzyme, dbGear, r);
        }
        // Decoys stay unique against every real target PLUS every entrapment.
        Set<String> targetSideSet = new HashSet<>(targetSet);
        if (cfg.addEntrapment) {
            for (Quartet q : quartets) {
                if (q.pTarget != null) {
                    targetSideSet.add(q.pTarget);
                }
            }
        }
        Set<String> targetSideSetIl = cfg.addEntrapment
                ? ilNormalizedSet(targetSideSet)
                : targetSetIl;
        if (cfg.addDecoys) {
            for (Quartet q : quartets) {
                q.decoy = generateReverseDecoy(
                        q.target, targetSideSet, targetSideSetIl, cfg.similarityGate);
                if (cfg.addEntrapment && q.pTarget != null) {
                    q.pDecoy = generateReverseDecoy(
                            q.pTarget, targetSideSet, targetSideSetIl, cfg.similarityGate);
                }
            }
        }
        r.quartetsBuilt = quartets.size();

        // Step 3: drop a quartet when no acceptable entrapment could be generated for a target that
        // was selected to carry one, when its entrapment collides with a real target (belt and
        // braces - both generators already reject that), or when a requested decoy could not be
        // produced. A target that was deliberately left without entrapment (ratio < 1) is kept.
        List<Quartet> kept = new ArrayList<>(quartets.size());
        for (Quartet q : quartets) {
            boolean drop = (q.entrapmentSelected && q.pTarget == null)
                    || (q.pTarget != null && targetSet.contains(q.pTarget))
                    || (cfg.addDecoys && q.decoy == null)
                    || (cfg.addDecoys && cfg.addEntrapment && q.pTarget != null && q.pDecoy == null);
            if (drop && cfg.addDecoys && q.decoy == null) {
                r.droppedNoDecoy++;
            }
            if (!drop) {
                kept.add(q);
            }
        }
        r.quartetsDropped = quartets.size() - kept.size();
        r.keptQuartets = kept.size();
        String composition = cfg.addEntrapment
                ? "target+p_target+decoy+p_decoy"
                : (cfg.addDecoys ? "target+decoy" : "target only");
        Cloger.getInstance().logger.info(String.format(
                "Collision-drop pass: %d peptide groups dropped, %d retained (each group = %s)",
                r.quartetsDropped, r.keptQuartets, composition));
        if (cfg.addEntrapment || cfg.addDecoys) {
            Cloger.getInstance().logger.info(String.format(
                    "Similarity gate (max fragment overlap %.2f): %d dropped with no acceptable "
                            + "entrapment, %d with no acceptable decoy",
                    DecoySimilarityGate.MAX_FRAGMENT_OVERLAP, r.droppedNoEntrapment, r.droppedNoDecoy));
        }

        // Stable ordering: sort by target sequence so the manifest pair_index is reproducible,
        // and sort each quartet's sources so the primary (header) accession is deterministic.
        kept.sort((a, b) -> a.target.compareTo(b.target));
        for (Quartet q : kept) {
            q.sources.sort((x, y) -> {
                int c = x.accession.compareTo(y.accession);
                return c != 0 ? c : x.entryName.compareTo(y.entryName);
            });
        }

        // Step 4: write FASTA (one entry per peptide).
        writeFasta(cfg, kept, r);

        // Step 5: optional manifest.
        if (cfg.manifest != null) {
            writeManifest(cfg, kept);
        }
        return r;
    }

    /**
     * Populate {@code pTarget} on every quartet selected to carry entrapment, either by shuffling
     * the target (default) or by drawing a mass-matched foreign peptide when
     * {@link Config#entrapmentSourceFasta} is set.
     *
     * <p>Quartets are processed in sequence-sorted order so the assignment does not depend on the
     * order proteins happened to appear in the input FASTA. That matters for the foreign source,
     * where each draw consumes a peptide and therefore changes what later targets can get.</p>
     */
    private static void assignEntrapment(Config cfg, List<Quartet> quartets, Set<String> targetSet,
                                         Set<String> targetSetIl, Enzyme enzyme, DBGear dbGear,
                                         Result r) throws IOException {
        if (cfg.entrapmentRatio > 1.0) {
            throw new IllegalArgumentException(
                    "entrapment ratio cannot exceed 1.0 (one entrapment peptide per target): "
                            + cfg.entrapmentRatio);
        }
        if (cfg.entrapmentRatio < MIN_ENTRAPMENT_RATIO) {
            throw new IllegalArgumentException(String.format(
                    "entrapment ratio %.4f is below the tested minimum of %.2f. Ratios down to "
                            + "%.2f are characterised (the combined FDP estimate is stable across "
                            + "r = 1, 0.5, 0.25 and 0.1), but at %.2f only ~32 entrapment peptides "
                            + "were accepted at 1%% q, so below it the estimate is dominated by "
                            + "counting noise and has not been validated. If you have a case that "
                            + "needs a lower ratio, raise it so the bound can be widened "
                            + "deliberately.",
                    cfg.entrapmentRatio, MIN_ENTRAPMENT_RATIO, MIN_ENTRAPMENT_RATIO,
                    MIN_ENTRAPMENT_RATIO));
        }
        List<Quartet> ordered = new ArrayList<>(quartets);
        ordered.sort((a, b) -> a.target.compareTo(b.target));

        List<Quartet> selected;
        int nSelected = (int) Math.round(cfg.entrapmentRatio * ordered.size());
        if (nSelected >= ordered.size()) {
            selected = ordered;
        } else {
            // Seeded selection so two machines building at the same ratio pick the same subset.
            List<Quartet> shuffled = new ArrayList<>(ordered);
            Random rng = new Random(cfg.entrapmentSeed);
            for (int i = shuffled.size() - 1; i > 0; i--) {
                int j = rng.nextInt(i + 1);
                Quartet tmp = shuffled.get(i);
                shuffled.set(i, shuffled.get(j));
                shuffled.set(j, tmp);
            }
            selected = new ArrayList<>(shuffled.subList(0, nSelected));
            selected.sort((a, b) -> a.target.compareTo(b.target));
        }
        r.entrapmentNotSelected = ordered.size() - selected.size();
        for (Quartet q : selected) {
            q.entrapmentSelected = true;
        }

        if (cfg.entrapmentSourceFasta == null) {
            for (Quartet q : selected) {
                q.pTarget = generateShuffledEntrapment(
                        q.target, cfg.entrapmentSeed, targetSet, targetSetIl, cfg.similarityGate);
                if (q.pTarget == null) {
                    r.droppedNoEntrapment++;
                }
            }
            return;
        }

        Cloger.getInstance().logger.info(
                "Drawing entrapment from foreign proteome: " + cfg.entrapmentSourceFasta);
        ForeignEntrapmentSource pool = ForeignEntrapmentSource.build(
                cfg.entrapmentSourceFasta, enzyme, dbGear, targetSet, targetSetIl,
                cfg.applyMzFilter, cfg.charges, cfg.minMz, cfg.maxMz);
        r.entrapmentIlCollisionsDropped = pool.ilCollisionsDropped();
        if (pool.available() < selected.size()) {
            // Not fatal - the shortfall shows up as dropped quartets and a lower achieved ratio -
            // but it silently changes what was asked for, so say so before it happens.
            Cloger.getInstance().logger.warn(String.format(
                    "Foreign entrapment pool has %d peptides for %d selected targets; the achieved "
                            + "ratio will be below the requested %.3f",
                    pool.available(), selected.size(), cfg.entrapmentRatio));
        }
        // Assignment order is deliberately UNCORRELATED with mass (sequence order). Sweeping in
        // mass order lets each target consume supply just above it, so a deficit accumulates and
        // later targets are dragged progressively further off their own mass.
        List<Quartet> drawOrder = new ArrayList<>(selected);
        drawOrder.removeIf(q -> peptideNeutralMass(q.target) == null);
        r.droppedNoEntrapment += selected.size() - drawOrder.size();

        double[] massDeltas = new double[drawOrder.size()];
        int nDeltas = 0;
        for (Quartet q : drawOrder) {
            double mass = peptideNeutralMass(q.target);
            String foreign = pool.assign(q.target, mass);
            if (foreign == null) {
                r.droppedNoEntrapment++;
                continue;
            }
            q.pTarget = foreign;
            r.entrapmentFromForeign++;
            Double foreignMass = peptideNeutralMass(foreign);
            if (foreignMass != null) {
                massDeltas[nDeltas++] = Math.abs(mass - foreignMass);
            }
        }
        Cloger.getInstance().logger.info(String.format(
                "Foreign entrapment assigned to %d of %d selected targets (%d unmatched); "
                        + "%d peptides left unused in the pool",
                r.entrapmentFromForeign, selected.size(), r.droppedNoEntrapment, pool.available()));

        // Match quality is what makes this a CONTROLLED comparison rather than just a swap: an
        // entrapment peptide only samples the same difficulty as its target if it co-locates in
        // the same isolation window. Report it rather than leaving it to be discovered later.
        if (nDeltas > 0) {
            double[] sorted = java.util.Arrays.copyOf(massDeltas, nDeltas);
            java.util.Arrays.sort(sorted);
            int inWindow = 0;
            for (int i = 0; i < nDeltas; i++) {
                // CO_LOCATION_WINDOW_DA (6 Da neutral) expressed as m/z at charge 2, i.e. the
                // pair sits within +/-3 m/z - a 6 m/z span, not a 3 m/z one.
                if (massDeltas[i] / 2.0 < ForeignEntrapmentSource.CO_LOCATION_WINDOW_DA / 2.0) {
                    inWindow++;
                }
            }
            r.entrapmentMedianAbsMassDelta = sorted[nDeltas / 2];
            r.entrapmentInIsolationWindowFraction = (double) inWindow / nDeltas;
            Cloger.getInstance().logger.info(String.format(
                    "Foreign entrapment mass match: median |dm| %.4f Da, 90th %.4f, 99th %.4f, "
                            + "max %.2f; %.2f%% within +/-3 m/z of their target at charge 2",
                    sorted[nDeltas / 2], sorted[(int) (0.90 * (nDeltas - 1))],
                    sorted[(int) (0.99 * (nDeltas - 1))], sorted[nDeltas - 1],
                    100.0 * r.entrapmentInIsolationWindowFraction));
        }
    }

    private static void writeFasta(Config cfg, List<Quartet> kept, Result r) throws IOException {
        Cloger.getInstance().logger.info("Writing FASTA: " + cfg.outputFasta);
        File out = new File(cfg.outputFasta);
        if (out.getParentFile() != null) {
            out.getParentFile().mkdirs();
        }
        // Per-source-protein peptide counter, incremented in lockstep across all sources of a
        // shared peptide so a joined header carries coherent per-source suffixes.
        Map<String, Integer> proteinPepCounter = new LinkedHashMap<>();
        try (BufferedWriter fo = Files.newBufferedWriter(out.toPath(), StandardCharsets.UTF_8)) {
            for (Quartet q : kept) {
                List<Object[]> sourcesWithCounters = new ArrayList<>(q.sources.size());
                if (cfg.uniqueAccessions) {
                    for (ProteinRecord src : q.sources) {
                        int cnt = proteinPepCounter.getOrDefault(src.accession, 0) + 1;
                        proteinPepCounter.put(src.accession, cnt);
                        sourcesWithCounters.add(new Object[]{src, cnt});
                    }
                } else {
                    for (ProteinRecord src : q.sources) {
                        sourcesWithCounters.add(new Object[]{src, null});
                    }
                }
                if (q.sources.size() > 1) {
                    r.sharedEntries++;
                }

                fo.write(">" + joinedLabel(cfg, sourcesWithCounters, false, false) + "\n");
                fo.write(q.target + "\n");
                r.targetEntries++;
                if (q.pTarget != null) {
                    fo.write(">" + joinedLabel(cfg, sourcesWithCounters, true, false) + "\n");
                    fo.write(q.pTarget + "\n");
                    r.pTargetEntries++;
                }
                if (q.decoy != null) {
                    fo.write(">" + joinedLabel(cfg, sourcesWithCounters, false, true) + "\n");
                    fo.write(q.decoy + "\n");
                    r.decoyEntries++;
                }
                if (q.pDecoy != null) {
                    fo.write(">" + joinedLabel(cfg, sourcesWithCounters, true, true) + "\n");
                    fo.write(q.pDecoy + "\n");
                    r.pDecoyEntries++;
                }
            }
        }
        Cloger.getInstance().logger.info(String.format(
                "Wrote FASTA: %d target, %d p_target, %d decoy, %d p_decoy entries "
                        + "(%d shared across multiple source proteins)",
                r.targetEntries, r.pTargetEntries, r.decoyEntries, r.pDecoyEntries, r.sharedEntries));
    }

    /** Semicolon-joined label for all sources of a peptide (each decorated + counter-suffixed). */
    private static String joinedLabel(Config cfg, List<Object[]> sourcesWithCounters,
                                      boolean pTarget, boolean decoy) {
        StringBuilder sb = new StringBuilder();
        for (Object[] sc : sourcesWithCounters) {
            if (sb.length() > 0) {
                sb.append(";");
            }
            ProteinRecord src = (ProteinRecord) sc[0];
            Integer cnt = (Integer) sc[1];
            sb.append(buildProteinLabel(src, pTarget, decoy, cfg.entrapmentSuffix,
                    cfg.decoyPrefix, cnt, cfg.pepSuffixFormat));
        }
        return sb.toString();
    }

    private static void writeManifest(Config cfg, List<Quartet> kept) throws IOException {
        // Refuse to emit a pairing a paired FDP estimator cannot consume. Downstream this surfaces
        // as a crash or a quietly mis-scaled FDP hours into a search; here it costs one pass over
        // quartets already in memory.
        EntrapmentPairingValidator.enforce(
                EntrapmentPairingValidator.validateQuartets(kept), cfg.failOnPairingViolation);
        Cloger.getInstance().logger.info("Writing manifest: " + cfg.manifest);
        File out = new File(cfg.manifest);
        if (out.getParentFile() != null) {
            out.getParentFile().mkdirs();
        }
        try (BufferedWriter fm = Files.newBufferedWriter(out.toPath(), StandardCharsets.UTF_8)) {
            fm.write("sequence\tdecoy\tproteins\tpeptide_type\tpeptide_pair_index\n");
            int pairIdx = 0;
            for (Quartet q : kept) {
                fm.write(q.target + "\tNo\t" + proteinsFor(cfg, q, false, false)
                        + "\ttarget\t" + pairIdx + "\n");
                if (q.pTarget != null) {
                    fm.write(q.pTarget + "\tNo\t" + proteinsFor(cfg, q, true, false)
                            + "\tp_target\t" + pairIdx + "\n");
                }
                if (q.decoy != null) {
                    fm.write(q.decoy + "\tYes\t" + proteinsFor(cfg, q, false, true)
                            + "\tdecoy\t" + pairIdx + "\n");
                }
                if (q.pDecoy != null) {
                    fm.write(q.pDecoy + "\tYes\t" + proteinsFor(cfg, q, true, true)
                            + "\tp_decoy\t" + pairIdx + "\n");
                }
                pairIdx++;
            }
        }
        Cloger.getInstance().logger.info("Wrote manifest with " + kept.size() + " pair_index groups");
    }

    /** Manifest {@code proteins} column: clean (un-suffixed) accessions joined by ';'. */
    private static String proteinsFor(Config cfg, Quartet q, boolean pTarget, boolean decoy) {
        StringBuilder sb = new StringBuilder();
        for (ProteinRecord src : q.sources) {
            if (sb.length() > 0) {
                sb.append(";");
            }
            sb.append(buildProteinLabel(src, pTarget, decoy, cfg.entrapmentSuffix,
                    cfg.decoyPrefix, null, cfg.pepSuffixFormat));
        }
        return sb.toString();
    }

    private static String cleanSeq(String seq) {
        if (seq == null) {
            return "";
        }
        seq = seq.replaceAll("^\\*", "").replaceAll("\\*$", "").toUpperCase();
        return seq;
    }

    /** Parse a uniprot {@code db|accession|entry description} header into a {@link ProteinRecord}. */
    static ProteinRecord parseHeader(String header, String sequence) {
        if (sequence == null || sequence.isEmpty()) {
            return null;
        }
        if (header.startsWith(">")) {
            header = header.substring(1);
        }
        String[] parts = header.trim().split("\\s+", 2);
        String ident = parts[0];
        String[] idp = ident.split("\\|");
        String db;
        String acc;
        String entry;
        if (idp.length >= 3) {
            db = idp[0];
            acc = idp[1];
            entry = idp[2];
        } else if (idp.length == 2) {
            db = idp[0];
            acc = idp[1];
            entry = idp[1];
        } else {
            db = "sp";
            acc = ident;
            entry = ident;
        }
        return new ProteinRecord(acc, entry, db, sequence);
    }
}
