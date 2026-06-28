package main.java.dia;

import java.util.Map;
import java.util.SortedMap;

/**
 * Matches a precursor to the MS2 scan that should carry its fragments: the scan whose isolation
 * window contains the precursor m/z and whose retention time is closest to the precursor's apex RT.
 *
 * <p>Extracted from {@code AIGear.add_ms2spectrum_index} so the matching logic -- which decides the
 * per-precursor MS2 scan ordinal written into the training input, and whose mis-resolution surfaces
 * downstream as "Spectrum not found" -- can be unit-tested in isolation.
 */
public final class ApexMatcher {

    private ApexMatcher() {
    }

    /**
     * Pick the MS2 scan whose isolation window {@code [isolation_mz_start, isolation_mz_end]}
     * contains {@code precursorMz} and whose {@code rt} is closest to {@code rt}. Each value map
     * must carry the keys {@code rt}, {@code isolation_mz_start} and {@code isolation_mz_end}.
     * Returns the matching ordinal (the map key), or {@code -1} when no window contains the
     * precursor.
     *
     * <p>The map must be ordered by MS2 ordinal, which is acquisition order and therefore
     * RT-ascending; that ordering is what makes the "rt is already &gt; 2 min past the target"
     * early-exit sound. Pass a {@link SortedMap} (e.g. {@code TreeMap}).
     */
    public static int matchByIsolationRange(
            SortedMap<Integer, ? extends Map<String, Double>> index, double precursorMz, double rt) {
        double deltaRt = Double.POSITIVE_INFINITY;
        int matched = -1;
        for (Map.Entry<Integer, ? extends Map<String, Double>> e : index.entrySet()) {
            Map<String, Double> w = e.getValue();
            if (w.get("isolation_mz_start") <= precursorMz
                    && precursorMz <= w.get("isolation_mz_end")) {
                double d = Math.abs(w.get("rt") - rt);
                if (d < deltaRt) {
                    deltaRt = d;
                    matched = e.getKey();
                } else if (w.get("rt") > rt + 2) {
                    // Past the apex by > 2 min and no longer improving: later scans are farther.
                    break;
                }
            }
        }
        return matched;
    }
}
