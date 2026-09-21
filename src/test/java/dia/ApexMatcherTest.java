package test.java.dia;

import main.java.dia.ApexMatcher;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

/**
 * Tests for {@link ApexMatcher#matchByIsolationRange}, the precursor -> MS2-scan resolution that
 * feeds the training input's {@code ms2index}. Mis-resolving it is what surfaces downstream as
 * "Spectrum not found" / zero valid PSMs, so the in-window, closest-RT behavior is pinned here on a
 * small synthetic interleaved DIA acquisition (two 4 m/z windows cycling over three RT points).
 */
public class ApexMatcherTest {

    private static Map<String, Double> win(double rt, double start, double end) {
        Map<String, Double> m = new HashMap<>();
        m.put("rt", rt);
        m.put("isolation_mz_start", start);
        m.put("isolation_mz_end", end);
        return m;
    }

    /** Ordinal -> window, in acquisition order: A,B at rt 10; A,B at rt 12; A,B at rt 14. */
    private static TreeMap<Integer, Map<String, Double>> diaIndex() {
        TreeMap<Integer, Map<String, Double>> idx = new TreeMap<>();
        int ord = 0;
        for (double rt : new double[] { 10.0, 12.0, 14.0 }) {
            idx.put(ord++, win(rt, 696.567, 700.567)); // window A (even ordinals)
            idx.put(ord++, win(rt, 700.567, 704.567)); // window B (odd ordinals)
        }
        return idx;
    }

    @Test
    public void picksClosestRtScanInTheContainingWindow() {
        TreeMap<Integer, Map<String, Double>> idx = diaIndex();
        // 698.0 is in window A (ordinals 0,2,4 at rt 10,12,14); rt 12.1 -> ordinal 2.
        Assert.assertEquals(ApexMatcher.matchByIsolationRange(idx, 698.0, 12.1), 2);
        // 702.0 is in window B (ordinals 1,3,5); rt 13.9 -> ordinal 5.
        Assert.assertEquals(ApexMatcher.matchByIsolationRange(idx, 702.0, 13.9), 5);
    }

    @Test
    public void returnsMinusOneWhenNoWindowContainsThePrecursor() {
        Assert.assertEquals(ApexMatcher.matchByIsolationRange(diaIndex(), 710.0, 12.0), -1);
    }

    @Test
    public void windowBoundaryIsInclusive() {
        // 700.567 is the shared A-end / B-start boundary; it must resolve to a window, not -1.
        Assert.assertTrue(ApexMatcher.matchByIsolationRange(diaIndex(), 700.567, 10.0) >= 0);
    }

    @Test
    public void onATieTheEarlierScanWins() {
        // 698.0 in window A; rt 11.0 is equidistant from ordinal 0 (rt 10) and ordinal 2 (rt 12).
        // The comparison is strict (<), so the earlier scan is kept.
        Assert.assertEquals(ApexMatcher.matchByIsolationRange(diaIndex(), 698.0, 11.0), 0);
    }

    @Test
    public void resolvesWithinThePrecursorsOwnWindowNotAcrossWindows() {
        // A window-B precursor whose apex RT (10.0) coincides with a window-A scan must still pick a
        // window-B scan -- wrong-window resolution was the heart of the SEA-AD-class failure.
        int m = ApexMatcher.matchByIsolationRange(diaIndex(), 702.0, 10.0);
        Assert.assertEquals(m % 2, 1, "expected a window-B (odd) ordinal, got " + m);
    }
}
