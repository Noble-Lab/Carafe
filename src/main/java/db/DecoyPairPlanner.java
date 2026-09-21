package main.java.db;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * Plans the {@code DecoyPairs} table Carafe writes into a Skyline BiblioSpec {@code .blib}: the
 * additive, Skyline-neutral target/decoy pairing described in {@code docs/blib-decoy-pairing-spec.md}
 * (a copy lives in the {@code skyline-osprey-tool} repo).
 *
 * <p>The blib stores decoys as ordinary {@code RefSpectra} (reversed sequences); nothing in the
 * standard schema marks which entries are decoys or links a target to its paired decoy - that lives
 * only in the side-car {@code osprey_library_db_pairing} manifest. This planner joins that manifest to
 * the {@code RefSpectra} rows actually written (by normalized stripped sequence) and emits, per
 * target/decoy precursor pair, two {@code DecoyPairs} rows sharing a {@code PairID}.</p>
 *
 * <p>Semantics (per the spec):</p>
 * <ul>
 *   <li>One row per participating precursor ({@code peptideModSeq} + charge). {@code IsDecoy} 0/1 comes
 *       straight from the manifest's {@code decoy} column (target/p_target = 0; decoy/p_decoy = 1).</li>
 *   <li>{@code IsEntrapment} 0/1 is orthogonal to {@code IsDecoy} and marks FDRBench entrapment
 *       peptides ({@code p_target}/{@code p_decoy} = 1; real {@code target}/{@code decoy} = 0). Osprey
 *       scores an entrapment target as a target and its entrapment decoy as a decoy; this flag carries
 *       that annotation forward so follow-on tools can measure performance against the entrapment set.</li>
 *   <li>A target precursor is paired with its decoy precursor <b>at the same charge and with the same
 *       modification set</b>, so an Oxidation-M isoform pairs with the decoy's matching isoform. When a
 *       charge/mod bucket has several equivalent precursors on each side (e.g. two oxidised Met sites)
 *       they are zipped in id order - each pair is still mass- and charge-matched. Only buckets present
 *       on <b>both</b> sides are emitted (a charge on only one side is skipped).</li>
 *   <li>{@code Method} labels how the decoy was built - {@code reverse} (C-terminus preserved) or
 *       {@code cycle} (the collision fallback), inferred by comparing the decoy sequence to
 *       {@link EntrapmentFastaGear#reversePreservingCterm}. Set on decoy rows; {@code null} on targets.</li>
 *   <li>Entrapment quartets pair both {@code target}&harr;{@code decoy} and
 *       {@code p_target}&harr;{@code p_decoy}; each side pair gets its own {@code PairID}s.</li>
 * </ul>
 */
public final class DecoyPairPlanner {

    private DecoyPairPlanner() {
    }

    /** A written {@code RefSpectra} row that may participate in a pair. */
    public record Precursor(int refSpectraId, String strippedSeq, int charge, String modKey) {
    }

    /** One row of the pairing manifest. */
    public record ManifestEntry(String sequence, boolean isDecoy, String peptideType, int pairIndex) {
    }

    /** A planned {@code DecoyPairs} row: {@code method} is null on target rows. */
    public record DecoyPair(int refSpectraId, int isDecoy, int isEntrapment, int pairId, String method) {
    }

    /**
     * Stable key for a precursor's modification set: sorted, semicolon-joined modification names
     * (positions dropped). A target isoform and its decoy carry the same residue composition, hence the
     * same modification set, so this key matches corresponding isoforms across the pair.
     */
    public static String modKey(String modsSemicolon) {
        if (modsSemicolon == null || modsSemicolon.isBlank()) {
            return "";
        }
        List<String> names = new ArrayList<>();
        for (String m : modsSemicolon.split(";")) {
            String t = m.trim();
            if (!t.isEmpty()) {
                names.add(t);
            }
        }
        names.sort(Comparator.naturalOrder());
        return String.join(";", names);
    }

    /**
     * Plan the {@code DecoyPairs} rows for a library.
     *
     * @param precursors every written {@code RefSpectra} row (id, stripped sequence, charge, mod key)
     * @param manifest   the pairing manifest rows (sequence, isDecoy, peptide_type, pair_index)
     * @return the pairing rows to insert; empty if nothing pairs
     */
    public static List<DecoyPair> plan(List<Precursor> precursors, List<ManifestEntry> manifest) {
        // Index precursors by I<->L-normalized stripped sequence.
        Map<String, List<Precursor>> bySeq = new LinkedHashMap<>();
        for (Precursor p : precursors) {
            bySeq.computeIfAbsent(norm(p.strippedSeq()), k -> new ArrayList<>()).add(p);
        }

        // Group manifest sequences by pair index, keyed by peptide_type (target/p_target/decoy/p_decoy).
        Map<Integer, Map<String, String>> groups = new TreeMap<>();
        for (ManifestEntry e : manifest) {
            groups.computeIfAbsent(e.pairIndex(), k -> new LinkedHashMap<>())
                    .put(e.peptideType() == null ? "" : e.peptideType().trim().toLowerCase(), e.sequence());
        }

        List<DecoyPair> out = new ArrayList<>();
        int[] nextPairId = {1};
        for (Map<String, String> g : groups.values()) {
            // The primary target/decoy pair and, for entrapment quartets, the p_target/p_decoy pair.
            emitSidePair(g.get("target"), g.get("decoy"), false, bySeq, out, nextPairId);
            emitSidePair(g.get("p_target"), g.get("p_decoy"), true, bySeq, out, nextPairId);
        }
        return out;
    }

    /** Emit pairs for one target-side / decoy-side sequence pairing within a manifest group. */
    private static void emitSidePair(String targetSeq, String decoySeq, boolean entrapment,
                                     Map<String, List<Precursor>> bySeq,
                                     List<DecoyPair> out, int[] nextPairId) {
        if (targetSeq == null || decoySeq == null) {
            return; // need both sides present in the manifest
        }
        List<Precursor> targets = bySeq.get(norm(targetSeq));
        List<Precursor> decoys = bySeq.get(norm(decoySeq));
        if (targets == null || decoys == null) {
            return; // one side didn't make it into the predicted library
        }
        String method = EntrapmentFastaGear.reversePreservingCterm(targetSeq).equals(decoySeq)
                ? "reverse" : "cycle";

        Map<String, List<Precursor>> tByBucket = bucketByChargeAndMods(targets);
        Map<String, List<Precursor>> dByBucket = bucketByChargeAndMods(decoys);
        for (Map.Entry<String, List<Precursor>> te : tByBucket.entrySet()) {
            List<Precursor> ds = dByBucket.get(te.getKey());
            if (ds == null) {
                continue; // this (charge, mod set) exists only on the target side
            }
            List<Precursor> ts = te.getValue();
            ts.sort(Comparator.comparingInt(Precursor::refSpectraId));
            ds.sort(Comparator.comparingInt(Precursor::refSpectraId));
            int ent = entrapment ? 1 : 0;
            int n = Math.min(ts.size(), ds.size());
            for (int i = 0; i < n; i++) {
                int pairId = nextPairId[0]++;
                out.add(new DecoyPair(ts.get(i).refSpectraId(), 0, ent, pairId, null));
                out.add(new DecoyPair(ds.get(i).refSpectraId(), 1, ent, pairId, method));
            }
        }
    }

    private static Map<String, List<Precursor>> bucketByChargeAndMods(List<Precursor> ps) {
        Map<String, List<Precursor>> m = new LinkedHashMap<>();
        for (Precursor p : ps) {
            m.computeIfAbsent(p.charge() + "|" + p.modKey(), k -> new ArrayList<>()).add(p);
        }
        return m;
    }

    /** I&rarr;L normalized comparison key (matches {@link PairingManifestReconciler#normalize}). */
    private static String norm(String seq) {
        return PairingManifestReconciler.normalize(seq);
    }

    /**
     * Parse a pairing manifest ({@code sequence, decoy, proteins, peptide_type, peptide_pair_index})
     * into {@link ManifestEntry} rows, tolerating column reordering (matched by header name).
     */
    public static List<ManifestEntry> parseManifest(File manifestFile) throws IOException {
        List<ManifestEntry> rows = new ArrayList<>();
        try (BufferedReader br = Files.newBufferedReader(manifestFile.toPath(), StandardCharsets.UTF_8)) {
            String headerLine = br.readLine();
            if (headerLine == null) {
                return rows;
            }
            String[] header = headerLine.split("\t", -1);
            int iSeq = colIndex(header, "sequence");
            int iDecoy = colIndex(header, "decoy");
            int iType = colIndex(header, "peptide_type");
            int iPair = colIndex(header, "peptide_pair_index");
            int maxIdx = Math.max(Math.max(iSeq, iDecoy), Math.max(iType, iPair));
            String line;
            int lineNo = 1;
            while ((line = br.readLine()) != null) {
                lineNo++;
                if (line.isEmpty()) {
                    continue;
                }
                String[] c = line.split("\t", -1);
                if (c.length <= maxIdx) {
                    throw new IOException("Malformed pairing manifest at line " + lineNo + ": expected at least "
                            + (maxIdx + 1) + " columns, found " + c.length);
                }
                boolean isDecoy = "yes".equalsIgnoreCase(c[iDecoy].trim());
                int pairIndex;
                try {
                    pairIndex = Integer.parseInt(c[iPair].trim());
                } catch (NumberFormatException nfe) {
                    throw new IOException("Malformed pairing manifest at line " + lineNo
                            + ": non-integer peptide_pair_index '" + c[iPair] + "'");
                }
                rows.add(new ManifestEntry(c[iSeq].trim(), isDecoy, c[iType].trim(), pairIndex));
            }
        }
        return rows;
    }

    private static int colIndex(String[] header, String name) throws IOException {
        for (int i = 0; i < header.length; i++) {
            if (header[i].trim().equalsIgnoreCase(name)) {
                return i;
            }
        }
        throw new IOException("Pairing manifest is missing the '" + name + "' column");
    }
}
