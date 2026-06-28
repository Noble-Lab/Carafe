package main.java.db;

import com.compomics.util.experiment.biology.enzymes.Enzyme;
import main.java.input.CParameter;
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
     * Deterministically shuffle all but the last residue. The per-peptide RNG seed is derived
     * as SHA-1({@code masterSeed:seq}) so the same (sequence, masterSeed) pair always returns the
     * same shuffle while different sequences get independent shuffles. Length-1 and length-2
     * inputs are returned unchanged.
     */
    public static String shufflePreservingCterm(String seq, long masterSeed) {
        if (seq.length() <= 2) {
            return seq;
        }
        char last = seq.charAt(seq.length() - 1);
        List<Character> body = new ArrayList<>(seq.length() - 1);
        for (int i = 0; i < seq.length() - 1; i++) {
            body.add(seq.charAt(i));
        }
        long pepSeed = derivePepSeed(masterSeed, seq);
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

    /** First 8 bytes of SHA-1("{@code masterSeed:seq}") as an unsigned-ish long seed. */
    private static long derivePepSeed(long masterSeed, String seq) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-1");
            byte[] digest = md.digest((masterSeed + ":" + seq).getBytes(StandardCharsets.UTF_8));
            long s = 0L;
            for (int i = 0; i < 8; i++) {
                s = (s << 8) | (digest[i] & 0xffL);
            }
            return s;
        } catch (NoSuchAlgorithmException e) {
            // SHA-1 is guaranteed present on every JVM; fall back to the string hash if not.
            return ((long) seq.hashCode() << 16) ^ masterSeed;
        }
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
        String rev = reversePreservingCterm(seq);
        if (!rev.equals(seq) && !targetSet.contains(rev)) {
            return rev;
        }
        int maxRetries = Math.min(seq.length(), 10);
        for (int c = 1; c <= maxRetries; c++) {
            String cyc = cyclePreservingCterm(seq, c);
            if (!cyc.equals(seq) && !targetSet.contains(cyc)) {
                return cyc;
            }
        }
        return null;
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
                    Double neutralMass = peptideNeutralMass(pep);
                    if (neutralMass == null) {
                        r.droppedUnknownAa++;
                        continue;
                    }
                    if (cfg.applyMzFilter
                            && !fitsMzRange(neutralMass, cfg.charges, cfg.minMz, cfg.maxMz)) {
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

        // Step 2: build quartets. Entrapment (p_target) sequences are deterministic shuffles, while
        // decoys (decoy / p_decoy) are reverse-with-C-term-preserved, cycling internal residues to
        // avoid collisions with real targets, matching Osprey's DecoyGenerator. A target whose decoy
        // cannot be made unique yields a null decoy and is dropped below.
        Set<String> targetSet = targetToSources.keySet();
        List<Quartet> quartets = new ArrayList<>(targetToSources.size());
        for (Map.Entry<String, List<ProteinRecord>> e : targetToSources.entrySet()) {
            Quartet q = new Quartet(e.getKey());
            q.sources.addAll(e.getValue());
            if (cfg.addEntrapment) {
                q.pTarget = shufflePreservingCterm(q.target, cfg.entrapmentSeed);
            }
            if (cfg.addDecoys) {
                q.decoy = generateReverseDecoy(q.target, targetSet);
                if (cfg.addEntrapment && q.pTarget != null) {
                    q.pDecoy = generateReverseDecoy(q.pTarget, targetSet);
                }
            }
            quartets.add(q);
        }
        r.quartetsBuilt = quartets.size();

        // Step 3: drop a quartet when its entrapment shuffle collides with a real target, or when a
        // requested reverse decoy could not be made unique (generateReverseDecoy returned null).
        List<Quartet> kept = new ArrayList<>(quartets.size());
        for (Quartet q : quartets) {
            boolean drop = (q.pTarget != null && targetSet.contains(q.pTarget))
                    || (cfg.addDecoys && q.decoy == null)
                    || (cfg.addDecoys && cfg.addEntrapment && q.pTarget != null && q.pDecoy == null);
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
