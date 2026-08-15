package test.java.dia;

import main.java.dia.IsolationWindow;
import main.java.dia.MsRunMeta;
import main.java.dia.MsRunMeta.AcquisitionEvidence;
import main.java.dia.MsRunMeta.AcquisitionType;
import main.java.dia.MsRunMeta.IsolationScheme;
import main.java.dia.MsRunMeta.ScanReadMode;
import main.java.dia.MsRunMeta.SchemeMetrics;

import org.testng.SkipException;
import org.testng.annotations.Test;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.stream.Stream;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertNotEquals;
import static org.testng.Assert.assertTrue;

/**
 * Tests for MsRunMeta.
 *
 * <p>The classification tests are pure and need no MS data. The two tests at the end need a real
 * mzML file and skip when none is available; point them at one with -Dcarafe.testdata=&lt;dir&gt;
 * or $CARAFE_TESTDATA.
 */
public class MsRunMetaTest {

    private static final double EPS = 1e-6;

    /**
     * Builds n windows of a fixed width, each stepped by step m/z from the previous one.
     */
    private static List<IsolationWindow> fixed_windows(double first_lower, double width,
                                                       double step, int n) {
        List<IsolationWindow> windows = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            double lo = first_lower + i * step;
            windows.add(new IsolationWindow(lo, lo + width));
        }
        return windows;
    }

    private static List<IsolationWindow> windows_of(double[][] pairs) {
        List<IsolationWindow> windows = new ArrayList<>(pairs.length);
        for (double[] pair : pairs) {
            windows.add(new IsolationWindow(pair[0], pair[1]));
        }
        return windows;
    }

    /**
     * Windows that tile a range with widths growing linearly, as a variable-window SWATH method does.
     */
    private static List<IsolationWindow> variable_tiling(double from, double to, int n) {
        List<IsolationWindow> windows = new ArrayList<>(n);
        // widths grow linearly from w0 to 3*w0 and sum to (to - from)
        double w0 = (to - from) / (2.0 * n);
        double lo = from;
        for (int i = 0; i < n; i++) {
            double width = w0 * (1.0 + 2.0 * i / Math.max(1, n - 1));
            windows.add(new IsolationWindow(lo, lo + width));
            lo += width;
        }
        return windows;
    }

    private static AcquisitionEvidence evidence(int n_ms2, int n_distinct, double periodicity,
                                                double median_width, double coverage,
                                                double charge_frac) {
        double distinct_ratio = n_ms2 == 0 ? 0 : (double) n_distinct / n_ms2;
        double cycles = n_distinct == 0 ? 0 : (double) n_ms2 / n_distinct;
        return new AcquisitionEvidence(n_ms2 / 10, n_ms2, n_distinct, distinct_ratio, n_distinct,
                periodicity, median_width, charge_frac, charge_frac, coverage, 10, cycles, false);
    }

    /**
     * A DIA-style window sequence: the same schedule stepped through over and over.
     */
    private static int[] cyclic_sequence(int n_windows, int n_cycles) {
        int[] seq = new int[n_windows * n_cycles];
        for (int i = 0; i < seq.length; i++) {
            seq[i] = i % n_windows;
        }
        return seq;
    }

    // ------------------------------------------------------------------ isolation scheme

    @Test
    public void testFixedNonOverlapping() {
        SchemeMetrics m = MsRunMeta.classify_isolation_scheme(fixed_windows(400, 8, 8, 75));
        assertEquals(m.scheme(), IsolationScheme.FIXED);
        assertTrue(m.is_fixed_width());
        assertFalse(m.is_overlapped());
        assertEquals(m.median_offset_ratio(), 1.0, 1e-9);
        assertEquals(m.n_width_outliers(), 0);
    }

    /**
     * A 1 m/z guard band on an 8 m/z window gives an offset ratio of 0.875. This must not be read
     * as staggered - it is the misclassification the 0.70 upper bound exists to prevent.
     */
    @Test
    public void testFixedWithGuardBand() {
        SchemeMetrics m = MsRunMeta.classify_isolation_scheme(fixed_windows(400, 8, 7, 75));
        assertEquals(m.scheme(), IsolationScheme.FIXED);
        assertFalse(m.is_overlapped());
        assertEquals(m.median_offset_ratio(), 0.875, 1e-9);
    }

    /**
     * The tightest realistic guard band case: a 1 m/z band on a 4 m/z window sits at 0.75, just
     * outside the overlapped range.
     */
    @Test
    public void testNarrowWindowGuardBand() {
        SchemeMetrics m = MsRunMeta.classify_isolation_scheme(fixed_windows(400, 4, 3, 100));
        assertFalse(m.is_overlapped());
        assertEquals(m.median_offset_ratio(), 0.75, 1e-9);
        assertEquals(m.scheme(), IsolationScheme.FIXED);
    }

    @Test
    public void testStaggered() {
        SchemeMetrics m = MsRunMeta.classify_isolation_scheme(fixed_windows(400, 8, 4, 75));
        assertEquals(m.scheme(), IsolationScheme.OVERLAPPED);
        assertTrue(m.is_overlapped());
        // staggered schemes are still uniform in width, and that fact must survive
        assertTrue(m.is_fixed_width());
        assertEquals(m.median_offset_ratio(), 0.5, 1e-9);
    }

    @Test
    public void testStaggeredNarrow() {
        SchemeMetrics m = MsRunMeta.classify_isolation_scheme(fixed_windows(400, 2, 1, 300));
        assertEquals(m.scheme(), IsolationScheme.OVERLAPPED);
        assertTrue(m.is_fixed_width());
    }

    @Test
    public void testVariableWidth() {
        SchemeMetrics m = MsRunMeta.classify_isolation_scheme(variable_tiling(400, 1000, 40));
        assertEquals(m.scheme(), IsolationScheme.VARIABLE);
        assertFalse(m.is_fixed_width());
        assertFalse(m.is_overlapped());
    }

    /**
     * The common "N equal windows plus one narrow remainder" layout. A plain coefficient of
     * variation rule calls this VARIABLE; the modal width rule gets it right.
     */
    @Test
    public void testFixedPlusRemainderWindow() {
        List<IsolationWindow> windows = fixed_windows(400, 25, 25, 23);
        double lo = 400 + 23 * 25.0;
        windows.add(new IsolationWindow(lo, lo + 13));
        SchemeMetrics m = MsRunMeta.classify_isolation_scheme(windows);
        assertEquals(m.scheme(), IsolationScheme.FIXED);
        assertTrue(m.is_fixed_width());
        assertEquals(m.n_width_outliers(), 1);
        assertTrue(m.width_cv() > 0.05, "the CV alone would have called this VARIABLE: " + m.width_cv());
    }

    @Test
    public void testVariableStaggered() {
        List<IsolationWindow> tiling = variable_tiling(400, 1000, 40);
        List<IsolationWindow> staggered = new ArrayList<>();
        for (IsolationWindow w : tiling) {
            double width = w.mz_upper - w.mz_lower;
            staggered.add(new IsolationWindow(w.mz_lower, w.mz_upper));
            staggered.add(new IsolationWindow(w.mz_lower + width / 2, w.mz_upper + width / 2));
        }
        SchemeMetrics m = MsRunMeta.classify_isolation_scheme(staggered);
        assertEquals(m.scheme(), IsolationScheme.OVERLAPPED);
        assertFalse(m.is_fixed_width());
    }

    @Test
    public void testTooFewWindows() {
        SchemeMetrics m = MsRunMeta.classify_isolation_scheme(fixed_windows(400, 8, 4, 2));
        assertFalse(m.is_overlapped(), "two windows cannot support an overlap call");
    }

    @Test
    public void testEmptyWindowList() {
        SchemeMetrics m = MsRunMeta.classify_isolation_scheme(new ArrayList<>());
        assertEquals(m.scheme(), IsolationScheme.UNKNOWN);
        assertEquals(m.n_windows(), 0);
    }

    @Test
    public void testCoverageFractionTiling() {
        assertEquals(MsRunMeta.coverage_fraction(fixed_windows(400, 8, 8, 75)), 1.0, 1e-9);
    }

    @Test
    public void testCoverageFractionStaggeredIsNotAboveOne() {
        double coverage = MsRunMeta.coverage_fraction(fixed_windows(400, 8, 4, 75));
        assertTrue(coverage <= 1.0 + 1e-9, "union coverage must not exceed 1, got " + coverage);
        assertTrue(coverage > 0.95, "a staggered scheme still tiles its range, got " + coverage);
    }

    @Test
    public void testCoverageFractionSparseTargets() {
        double[][] targets = new double[20][2];
        for (int i = 0; i < 20; i++) {
            targets[i][0] = 400 + i * 30.0;
            targets[i][1] = targets[i][0] + 4;
        }
        assertTrue(MsRunMeta.coverage_fraction(windows_of(targets)) < 0.2);
    }

    // ------------------------------------------------------------------ window merging

    @Test
    public void testMergeWindowsJitter() {
        List<double[]> raw = new ArrayList<>();
        for (int i = 0; i < 500; i++) {
            double jitter = ((i % 3) - 1) * 1e-4;
            raw.add(new double[]{400.04 + jitter, 424.04 + jitter});
        }
        // an edge sitting right on a 0.1 m/z bin boundary must not split the window
        raw.add(new double[]{400.06, 424.06});
        assertEquals(MsRunMeta.merge_windows(raw, 0.05).size(), 1);
    }

    @Test
    public void testMergeWindowsDistinct() {
        List<double[]> raw = new ArrayList<>();
        raw.add(new double[]{400.0, 424.0});
        raw.add(new double[]{400.06, 424.06});
        assertEquals(MsRunMeta.merge_windows(raw, 0.05).size(), 2);
    }

    @Test
    public void testMergeWindowsAveragesEdges() {
        List<double[]> raw = new ArrayList<>();
        raw.add(new double[]{400.00, 424.00});
        raw.add(new double[]{400.02, 424.02});
        List<IsolationWindow> merged = MsRunMeta.merge_windows(raw, 0.05);
        assertEquals(merged.size(), 1);
        assertEquals(merged.get(0).mz_lower, 400.01, 1e-9);
        assertEquals(merged.get(0).mz_upper, 424.01, 1e-9);
    }

    @Test
    public void testMergeWindowsSortsByLower() {
        List<double[]> raw = new ArrayList<>();
        raw.add(new double[]{700.0, 708.0});
        raw.add(new double[]{400.0, 408.0});
        raw.add(new double[]{550.0, 558.0});
        List<IsolationWindow> merged = MsRunMeta.merge_windows(raw, 0.05);
        assertEquals(merged.size(), 3);
        assertEquals(merged.get(0).mz_lower, 400.0, EPS);
        assertEquals(merged.get(1).mz_lower, 550.0, EPS);
        assertEquals(merged.get(2).mz_lower, 700.0, EPS);
    }

    @Test
    public void testMergeWindowsEmpty() {
        assertTrue(MsRunMeta.merge_windows(new ArrayList<>(), 0.05).isEmpty());
    }

    // ------------------------------------------------------------------ acquisition type

    @Test
    public void testAcqDia() {
        assertEquals(MsRunMeta.classify_acquisition(evidence(3000, 60, 1.0, 10, 1.0, 0.0)),
                AcquisitionType.DIA);
    }

    @Test
    public void testAcqDda() {
        assertEquals(MsRunMeta.classify_acquisition(evidence(3000, 2800, 0.002, 1.6, 0.99, 0.95)),
                AcquisitionType.DDA);
    }

    /**
     * Wide-window DDA: a width-first rule would call this DIA.
     */
    @Test
    public void testAcqWideWindowDda() {
        assertEquals(MsRunMeta.classify_acquisition(evidence(3000, 2600, 0.001, 25, 1.0, 0.9)),
                AcquisitionType.DDA);
    }

    /**
     * Narrow-window DIA on an Astral: a width-first rule would call this DDA. Real Astral data has
     * 1-2 m/z windows, narrower than the 2 m/z a Q Exactive uses for DDA.
     */
    @Test
    public void testAcqNarrowWindowDia() {
        assertEquals(MsRunMeta.classify_acquisition(evidence(3000, 300, 1.0, 2, 1.0, 0.0)),
                AcquisitionType.DIA);
    }

    /**
     * Regression guard for the discriminator this class originally used. Over a long gradient a
     * DDA run re-selects the same precursors many times, so its distinct-window ratio falls into
     * the range DIA occupies - these numbers are measured from a real 200 minute Q Exactive run,
     * where a ratio rule called it PRM. Only the lack of periodicity separates it.
     */
    @Test
    public void testAcqLongGradientDdaIsNotMistakenForTargeted() {
        AcquisitionEvidence e = evidence(33042, 4583, 0.0004, 2.0, 0.69, 1.0);
        assertTrue(e.distinct_ratio() < 0.5, "the ratio alone no longer says DDA here");
        assertEquals(MsRunMeta.classify_acquisition(e), AcquisitionType.DDA);
    }

    @Test
    public void testAcqPrm() {
        assertEquals(MsRunMeta.classify_acquisition(evidence(3000, 20, 1.0, 4, 0.15, 0.0)),
                AcquisitionType.PRM);
    }

    @Test
    public void testAcqMs1Only() {
        assertEquals(MsRunMeta.classify_acquisition(evidence(0, 0, 0, 0, 0, 0)),
                AcquisitionType.MS1_ONLY);
    }

    @Test
    public void testAcqSrmScanType() {
        AcquisitionEvidence e = new AcquisitionEvidence(10, 3000, 2000, 0.67, 2000, 0.0, 1.0, 0.0,
                0.0, 0.9, 10, 1.5, true);
        assertEquals(MsRunMeta.classify_acquisition(e), AcquisitionType.SRM);
    }

    @Test
    public void testAcqNoIsolationWindowsIsDda() {
        assertEquals(MsRunMeta.classify_acquisition(evidence(3000, 0, 0, 0, 0, 0.9)),
                AcquisitionType.DDA);
    }

    // ------------------------------------------------------------------ periodicity

    @Test
    public void testPeriodicityCyclic() {
        assertEquals(MsRunMeta.periodicity(cyclic_sequence(60, 50), 60), 1.0, 1e-9);
        assertEquals(MsRunMeta.periodicity(cyclic_sequence(250, 30), 250), 1.0, 1e-9);
    }

    @Test
    public void testPeriodicityScattered() {
        // pseudo-random but deterministic, standing in for DDA precursor selection
        int[] seq = new int[8000];
        long state = 12345;
        for (int i = 0; i < seq.length; i++) {
            state = state * 6364136223846793005L + 1442695040888963407L;
            seq[i] = (int) Math.floorMod(state >> 20, 4000);
        }
        assertTrue(MsRunMeta.periodicity(seq, 4000) < 0.1,
                "scattered precursors must not look periodic");
    }

    @Test
    public void testPeriodicitySequenceShorterThanCycle() {
        assertEquals(MsRunMeta.periodicity(cyclic_sequence(60, 1), 60), 0.0, 1e-9);
        assertEquals(MsRunMeta.periodicity(new int[0], 10), 0.0, 1e-9);
        assertEquals(MsRunMeta.periodicity(cyclic_sequence(10, 5), 0), 0.0, 1e-9);
    }

    @Test
    public void testPeriodicityIgnoresUnassignedScans() {
        int[] seq = cyclic_sequence(20, 20);
        for (int i = 0; i < seq.length; i += 37) {
            seq[i] = -1;
        }
        assertEquals(MsRunMeta.periodicity(seq, 20), 1.0, 1e-9);
    }

    @Test
    public void testPeriodicityToleratesOccasionalGaps() {
        int[] seq = cyclic_sequence(20, 40);
        // a handful of cycles miss a window; the schedule is still clearly periodic
        for (int i = 5; i < seq.length; i += 200) {
            seq[i] = 19;
        }
        assertTrue(MsRunMeta.periodicity(seq, 20) > 0.9);
    }

    // ------------------------------------------------------------------ cycle length

    @Test
    public void testEstimateCycleLengthOnCleanSchedule() {
        assertEquals(MsRunMeta.estimate_cycle_length(cyclic_sequence(60, 40)), 60);
        assertEquals(MsRunMeta.estimate_cycle_length(cyclic_sequence(250, 12)), 250);
    }

    /**
     * The Eclipse plant DIA case: a 101 window schedule plus one stray window seen once. Taking
     * the period from the distinct window count gives 102 and scores a clean DIA run at zero, so
     * the cycle length has to be measured from the sequence instead.
     */
    @Test
    public void testStrayWindowDoesNotChangeTheCycleLength() {
        int cycle = 101;
        int[] seq = new int[cycle * 70 + 1];
        seq[0] = cycle; // the stray window, never seen again
        for (int i = 1; i < seq.length; i++) {
            seq[i] = (i - 1) % cycle;
        }
        int n_distinct = cycle + 1;
        assertEquals(MsRunMeta.estimate_cycle_length(seq), cycle);
        // the stray scan itself is the one position that cannot match a cycle later
        double measured = MsRunMeta.periodicity(seq, MsRunMeta.estimate_cycle_length(seq));
        assertTrue(measured > 0.99, "expected a clearly periodic sequence, got " + measured);
        assertTrue(measured < 1.0, "the stray scan should still cost exactly one comparison");
        assertEquals(MsRunMeta.periodicity(seq, n_distinct), 0.0, 1e-9,
                "using the window count as the period is what broke on the real file");
    }

    /**
     * The Exploris yeast DIA case: each isolation window is acquired twice in a row, so a distance
     * of 1 is the most frequently observed repeat distance while the cycle is 300 scans long.
     * Taking the most common distance gives a period of 1 and scores the run at 0.4975.
     */
    @Test
    public void testWindowsAcquiredTwiceInARow() {
        int windows = 150;
        int[] seq = new int[windows * 2 * 25];
        for (int i = 0; i < seq.length; i++) {
            seq[i] = (i % (windows * 2)) / 2;
        }
        assertEquals(MsRunMeta.estimate_cycle_length(seq), windows * 2);
        assertEquals(MsRunMeta.periodicity(seq, MsRunMeta.estimate_cycle_length(seq)), 1.0, 1e-9);
        assertEquals(MsRunMeta.periodicity(seq, 1), 0.5, 0.01,
                "the most common repeat distance here is 1, which is what broke on the real file");
    }

    @Test
    public void testEstimateCycleLengthPrefersTheShortestTruePeriod() {
        // a period of 60 also repeats at 120; the shorter one is the real cycle
        int[] seq = cyclic_sequence(60, 40);
        assertEquals(MsRunMeta.estimate_cycle_length(seq), 60);
    }

    @Test
    public void testEstimateCycleLengthToleratesMissingScans() {
        int[] seq = cyclic_sequence(20, 40);
        for (int i = 7; i < seq.length; i += 111) {
            seq[i] = -1;
        }
        assertEquals(MsRunMeta.estimate_cycle_length(seq), 20);
    }

    @Test
    public void testEstimateCycleLengthWithNoRepeats() {
        int[] seq = new int[50];
        for (int i = 0; i < seq.length; i++) {
            seq[i] = i;
        }
        assertEquals(MsRunMeta.estimate_cycle_length(seq), 0);
        assertEquals(MsRunMeta.periodicity(seq, 0), 0.0, 1e-9);
    }

    // ------------------------------------------------------------------ window assignment

    @Test
    public void testAssignWindowsMapsBackToMerged() {
        List<double[]> raw = new ArrayList<>();
        for (int cycle = 0; cycle < 10; cycle++) {
            for (int w = 0; w < 5; w++) {
                double lo = 400.0 + w * 8.0;
                raw.add(new double[]{lo, lo + 8.0});
            }
        }
        List<IsolationWindow> merged = MsRunMeta.merge_windows(raw, 0.05);
        assertEquals(merged.size(), 5);
        int[] assigned = MsRunMeta.assign_windows(raw, merged, 0.05);
        assertEquals(assigned.length, raw.size());
        for (int i = 0; i < assigned.length; i++) {
            assertEquals(assigned[i], i % 5, "position " + i);
        }
        assertEquals(MsRunMeta.periodicity(assigned, 5), 1.0, 1e-9);
    }

    @Test
    public void testAssignWindowsMarksUnassignable() {
        List<IsolationWindow> merged = windows_of(new double[][]{{400.0, 408.0}});
        int[] assigned = MsRunMeta.assign_windows(
                List.of(new double[]{400.0, 408.0}, new double[]{600.0, 608.0}), merged, 0.05);
        assertEquals(assigned[0], 0);
        assertEquals(assigned[1], -1);
    }

    /**
     * A scan that declares no isolation window must occupy a slot rather than be dropped.
     * Dropping it shifts every later scan by one and destroys periodicity for the rest of the run.
     */
    @Test
    public void testAssignWindowsKeepsSlotForNullWindow() {
        List<IsolationWindow> merged = windows_of(new double[][]{{400.0, 408.0}, {408.0, 416.0}});
        List<double[]> raw = new ArrayList<>();
        raw.add(new double[]{400.0, 408.0});
        raw.add(null);
        raw.add(new double[]{408.0, 416.0});
        int[] assigned = MsRunMeta.assign_windows(raw, merged, 0.05);
        assertEquals(assigned.length, 3);
        assertEquals(assigned[0], 0);
        assertEquals(assigned[1], -1);
        assertEquals(assigned[2], 1);
    }

    /**
     * The end-to-end version of the gap bug: a DIA run where a small fraction of MS2 scans carry
     * no isolation window must still measure as periodic.
     */
    @Test
    public void testPeriodicitySurvivesScansWithoutWindows() {
        int n_windows = 60;
        List<double[]> raw = new ArrayList<>();
        for (int cycle = 0; cycle < 200; cycle++) {
            for (int w = 0; w < n_windows; w++) {
                // every 199th scan reports no window at all
                if ((cycle * n_windows + w) % 199 == 0) {
                    raw.add(null);
                } else {
                    double lo = 400.0 + w * 10.0;
                    raw.add(new double[]{lo, lo + 10.0});
                }
            }
        }
        List<double[]> present = new ArrayList<>();
        for (double[] w : raw) {
            if (w != null) {
                present.add(w);
            }
        }
        List<IsolationWindow> merged = MsRunMeta.merge_windows(present, 0.05);
        assertEquals(merged.size(), n_windows);
        int[] assigned = MsRunMeta.assign_windows(raw, merged, 0.05);
        assertEquals(MsRunMeta.periodicity(assigned, n_windows), 1.0, 1e-9);
    }

    /**
     * Two windows 0.08 m/z apart merge to distinct entries but collide on IsolationWindow's
     * 0.1 m/z id, so per-window counts must not be keyed on that id.
     */
    @Test
    public void testDistinctWindowsCanShareAnId() {
        List<double[]> raw = new ArrayList<>();
        raw.add(new double[]{399.96, 423.96});
        raw.add(new double[]{400.04, 424.04});
        List<IsolationWindow> merged = MsRunMeta.merge_windows(raw, 0.05);
        assertEquals(merged.size(), 2, "the two windows are further apart than the tolerance");
        assertEquals(merged.get(0).id, merged.get(1).id,
                "and they collide on the 0.1 m/z id, which is why it cannot be the count key");
    }

    /**
     * A single repeated window is trivially periodic and trivially covers its own range, so it
     * cannot be called DIA - it is equally consistent with SIM or a single-target PRM method.
     */
    @Test
    public void testAcqSingleWindowIsUnknown() {
        assertEquals(MsRunMeta.classify_acquisition(evidence(3000, 1, 1.0, 4.0, 1.0, 0.0)),
                AcquisitionType.UNKNOWN);
    }

    @Test
    public void testCoverageFractionSingleWindowIsDegenerate() {
        // documents why n_distinct == 1 is special-cased: coverage cannot fall below 1 for n=1
        assertEquals(MsRunMeta.coverage_fraction(windows_of(new double[][]{{500.0, 504.0}})),
                1.0, 1e-9);
    }

    @Test
    public void testWindowSetsDifferAcrossRun() {
        // same schedule throughout
        int[] steady = cyclic_sequence(20, 30);
        assertFalse(MsRunMeta.window_sets_differ(steady, 20));
        // first half uses windows 0-9, second half uses 10-19
        int[] segmented = new int[600];
        for (int i = 0; i < segmented.length; i++) {
            segmented[i] = (i < 300 ? 0 : 10) + (i % 10);
        }
        assertTrue(MsRunMeta.window_sets_differ(segmented, 20));
    }

    // ------------------------------------------------------------------ instrument mapping

    @Test
    public void testInstrumentFromModel() {
        assertEquals(MsRunMeta.instrument_from_model("Orbitrap Astral"), "Astral");
        assertEquals(MsRunMeta.instrument_from_model("Orbitrap Eclipse"), "Eclipse");
        assertEquals(MsRunMeta.instrument_from_model("Orbitrap Exploris 480"), "Exploris");
        assertEquals(MsRunMeta.instrument_from_model("Q Exactive HF-X Orbitrap"), "QEHFX");
        assertEquals(MsRunMeta.instrument_from_model("Q Exactive HF"), "QEHF");
        assertEquals(MsRunMeta.instrument_from_model("Q Exactive Plus"), "QE+");
        assertEquals(MsRunMeta.instrument_from_model("Q Exactive"), "QE");
        assertEquals(MsRunMeta.instrument_from_model("TripleTOF 6600"), "SciexTOF");
        assertEquals(MsRunMeta.instrument_from_model("timsTOF Pro 2"), "timsTOF");
        assertEquals(MsRunMeta.instrument_from_model("Some Future Instrument"), "");
        assertEquals(MsRunMeta.instrument_from_model(""), "");
        assertEquals(MsRunMeta.instrument_from_model(null), "");
    }

    /**
     * "Orbitrap Fusion Lumos" also matches "Orbitrap Fusion", so pattern order matters.
     */
    @Test
    public void testInstrumentFromModelPrefersMostSpecific() {
        assertEquals(MsRunMeta.instrument_from_model("Orbitrap Fusion Lumos"), "Lumos");
        assertEquals(MsRunMeta.instrument_from_model("Orbitrap Fusion"), "Fusion");
    }

    /**
     * Pins the accession mappings this class ships with.
     *
     * <p>This does NOT verify agreement with DIAMeta.get_ms_instrument. That map is a local
     * variable inside the method, so it cannot be read from here, and the two are already known to
     * differ: MS:1000932 and MS:1002417 exist only in MsRunMeta. Keeping them in step is manual.
     */
    @Test
    public void testAccessionMapEntries() {
        assertEquals(MsRunMeta.CV_ACCESSION_2_INSTRUMENT.get("MS:1003029"), "Eclipse");
        assertEquals(MsRunMeta.CV_ACCESSION_2_INSTRUMENT.get("MS:1003028"), "Exploris");
        assertEquals(MsRunMeta.CV_ACCESSION_2_INSTRUMENT.get("MS:1003378"), "Astral");
        assertEquals(MsRunMeta.CV_ACCESSION_2_INSTRUMENT.get("MS:1002416"), "Fusion");
        assertEquals(MsRunMeta.CV_ACCESSION_2_INSTRUMENT.get("MS:1002732"), "Lumos");
        assertEquals(MsRunMeta.CV_ACCESSION_2_INSTRUMENT.get("MS:1001911"), "QE");
        assertEquals(MsRunMeta.CV_ACCESSION_2_INSTRUMENT.get("MS:1002523"), "QEHF");
        assertEquals(MsRunMeta.CV_ACCESSION_2_INSTRUMENT.get("MS:1002526"), "QE+");
        assertEquals(MsRunMeta.CV_ACCESSION_2_INSTRUMENT.get("MS:1002634"), "QE+");
        assertEquals(MsRunMeta.CV_ACCESSION_2_INSTRUMENT.get("MS:1002877"), "QEHFX");
        assertEquals(MsRunMeta.CV_ACCESSION_2_INSTRUMENT.get("MS:1002533"), "SciexTOF");
    }

    @Test
    public void testIsMzml() {
        assertTrue(MsRunMeta.is_mzml("a.mzML"));
        assertTrue(MsRunMeta.is_mzml("a.mzml"));
        assertTrue(MsRunMeta.is_mzml("/path/to/A_RUN.MZML"));
        assertFalse(MsRunMeta.is_mzml("a.mgf"));
        assertFalse(MsRunMeta.is_mzml("a.mzML.gz"));
        assertFalse(MsRunMeta.is_mzml(null));
    }

    // ------------------------------------------------------------------ report shape

    @Test
    public void testAsRowHasNoNullsAndNoTabs() {
        MsRunMeta meta = new MsRunMeta();
        meta.ms_file = "/tmp/example.mzML";
        meta.notes.add("a note");
        meta.as_row().forEach((k, v) -> {
            assertNotEquals(v, null, "column " + k + " is null");
            assertFalse(String.valueOf(v).contains("\t"), "column " + k + " contains a tab");
        });
    }

    @Test
    public void testAsRowColumnsAreStableAcrossRuns() {
        MsRunMeta profiled = new MsRunMeta();
        profiled.ms_file = "/tmp/dia.mzML";
        profiled.scheme_metrics = MsRunMeta.classify_isolation_scheme(fixed_windows(400, 8, 8, 75));
        MsRunMeta bare = new MsRunMeta();
        bare.ms_file = "/tmp/failed.mzML";
        // the batch TSV writes one header from the first row, so every row needs the same columns
        assertEquals(new ArrayList<>(profiled.as_row().keySet()),
                new ArrayList<>(bare.as_row().keySet()));
    }

    /**
     * A file that could not be read is a failure; a file that was read but whose acquisition type
     * could not be decided is an honest verdict. The exit code keys on the first, not the second.
     */
    @Test
    public void testFailedToReadDistinguishesUnreadableFromUndecided() {
        MsRunMeta unread = new MsRunMeta();
        assertTrue(unread.failed_to_read());

        MsRunMeta wrong_format = new MsRunMeta().profile("not_an_ms_file.txt");
        assertTrue(wrong_format.failed_to_read());

        MsRunMeta undecided = new MsRunMeta();
        undecided.scan_read_mode = ScanReadMode.FULL;
        undecided.acquisition_type = AcquisitionType.UNKNOWN;
        assertFalse(undecided.failed_to_read(),
                "an undecided verdict on a file that was read is not a read failure");
    }

    @Test
    public void testHumanSize() {
        assertEquals(MsRunMeta.human_size(0), "0B");
        assertEquals(MsRunMeta.human_size(512), "512B");
        assertEquals(MsRunMeta.human_size(1024), "1.0K");
        assertEquals(MsRunMeta.human_size(4L * 1024), "4.0K");
        assertEquals(MsRunMeta.human_size(790L * 1024 * 1024), "790M");
        assertEquals(MsRunMeta.human_size(1932L * 1024 * 1024), "1.9G");
        assertEquals(MsRunMeta.human_size(3L * 1024 * 1024 * 1024 * 1024), "3.0T");
    }

    @Test
    public void testSummaryStatesFileSize() {
        MsRunMeta meta = new MsRunMeta();
        meta.ms_file = "/tmp/example.mzML";
        meta.file_size_bytes = 1932L * 1024 * 1024;
        assertTrue(meta.summary().contains("(1.9G)"), meta.summary());
        // the row keeps raw bytes so the column sorts correctly
        assertEquals(meta.as_row().get("file_size_bytes"), 1932L * 1024 * 1024);
    }

    @Test
    public void testProfileRecordsFileSize() throws Exception {
        Path root = Files.createTempDirectory("mzml_size");
        try {
            Path file = root.resolve("sized.mzML");
            write_minimal_mzml(file);
            MsRunMeta meta = new MsRunMeta().profile(file.toString());
            assertEquals(meta.file_size_bytes, Files.size(file));
            assertTrue(meta.file_size_bytes > 0);
        } finally {
            delete_tree(root);
        }
    }

    @Test
    public void testSummaryStatesScanReadMode() {
        MsRunMeta unread = new MsRunMeta();
        unread.ms_file = "/tmp/example.mzML";
        assertTrue(unread.summary().contains("Scan headers: NOT_READ"), unread.summary());

        MsRunMeta sampled = new MsRunMeta();
        sampled.ms_file = "/tmp/example.mzML";
        sampled.scan_read_mode = ScanReadMode.SAMPLED;
        String summary = sampled.summary();
        assertTrue(summary.contains("Scan headers: SAMPLED"), summary);
        assertTrue(summary.contains(ScanReadMode.SAMPLED.description), summary);
        // it qualifies the evidence counts, so it has to be stated before them
        assertTrue(summary.indexOf("Scan headers:") < summary.indexOf("Run:"), summary);
    }

    @Test
    public void testAsRowUsesEffectiveScanReadMode() {
        MsRunMeta meta = new MsRunMeta();
        meta.full_scan = false;
        meta.scan_read_mode = ScanReadMode.FULL; // an automatic fallback, not an explicit -full
        assertEquals(meta.as_row().get("scan_read_mode"), "FULL");

        MsRunMeta unread = new MsRunMeta();
        assertEquals(unread.as_row().get("scan_read_mode"), "NOT_READ");
    }

    @Test
    public void testProfileClearsPreviousResultButPreservesConfiguration() {
        MsRunMeta meta = new MsRunMeta();
        meta.ms_instrument = "Astral";
        meta.acquisition_type = AcquisitionType.DIA;
        meta.isolation_scheme = IsolationScheme.FIXED;
        meta.is_fixed_width = true;
        meta.isolation_windows.add(new IsolationWindow(400, 408));
        meta.window_scan_counts = new int[]{10};
        meta.evidence = evidence(100, 10, 1.0, 8, 1.0, 0.0);
        meta.notes.add("note from the previous run");
        meta.n_ms1 = 10;
        meta.n_ms2 = 100;
        meta.full_scan = true;
        meta.probe_scans_per_slice = 123;
        meta.window_merge_tol = 0.01;
        meta.num_threads = 2;

        meta.profile("second.raw");

        assertEquals(meta.ms_file, "second.raw");
        assertEquals(meta.ms_instrument, "");
        assertEquals(meta.acquisition_type, AcquisitionType.UNKNOWN);
        assertEquals(meta.isolation_scheme, IsolationScheme.UNKNOWN);
        assertFalse(meta.is_fixed_width);
        assertTrue(meta.isolation_windows.isEmpty());
        assertEquals(meta.window_scan_counts.length, 0);
        assertEquals(meta.evidence, null);
        assertEquals(meta.n_ms1, 0);
        assertEquals(meta.n_ms2, 0);
        assertEquals(meta.notes.size(), 1);
        assertTrue(meta.notes.get(0).startsWith("Not an mzML file"));
        assertEquals(meta.scan_read_mode, ScanReadMode.NOT_READ);
        assertTrue(meta.full_scan);
        assertEquals(meta.probe_scans_per_slice, 123);
        assertEquals(meta.window_merge_tol, 0.01, 1e-12);
        assertEquals(meta.num_threads, 2);
    }

    @Test
    public void testBatchTsvRoundTrip() throws IOException {
        MsRunMeta meta = new MsRunMeta();
        meta.ms_file = "/tmp/example.mzML";
        File out = File.createTempFile("mzml_acquisition_summary", ".tsv");
        out.deleteOnExit();
        MsRunMeta.write_batch_tsv(List.of(meta), out);
        List<String> lines = java.nio.file.Files.readAllLines(out.toPath());
        assertEquals(lines.size(), 2);
        assertEquals(lines.get(0).split("\t", -1).length, lines.get(1).split("\t", -1).length);
    }

    /**
     * Parser exception messages routinely embed newlines. Left unsanitized they split one run
     * across several lines and corrupt every row after it.
     */
    @Test
    public void testBatchTsvSurvivesMultilineNotes() throws IOException {
        MsRunMeta bad = new MsRunMeta();
        bad.ms_file = "/tmp/broken.mzML";
        bad.notes.add("Failed to read the file: ParseError at [row,col]:[12,34]\r\nMessage: boom\ttab");
        MsRunMeta good = new MsRunMeta();
        good.ms_file = "/tmp/fine.mzML";
        File out = File.createTempFile("mzml_acquisition_summary_multiline", ".tsv");
        out.deleteOnExit();
        MsRunMeta.write_batch_tsv(List.of(bad, good), out);
        List<String> lines = java.nio.file.Files.readAllLines(out.toPath());
        assertEquals(lines.size(), 3, "one header plus exactly one row per run");
        int columns = lines.get(0).split("\t", -1).length;
        for (String line : lines) {
            assertEquals(line.split("\t", -1).length, columns, "ragged row: " + line);
        }
    }

    /**
     * The smallest mzML msftbx will parse without complaint. Real content rather than an empty
     * file, so profiling succeeds and the test does not log parse errors that read as failures.
     */
    private static final String MINIMAL_MZML = """
            <?xml version="1.0" encoding="utf-8"?>
            <mzML xmlns="http://psi.hupo.org/ms/mzml" version="1.1.0">
              <cvList count="1">
                <cv id="MS" fullName="PSI-MS" URI="https://psidev.info/psi-ms.obo"/>
              </cvList>
              <fileDescription><fileContent/></fileDescription>
              <instrumentConfigurationList count="1">
                <instrumentConfiguration id="IC1">
                  <cvParam cvRef="MS" accession="MS:1001911" name="Q Exactive" value=""/>
                </instrumentConfiguration>
              </instrumentConfigurationList>
              <dataProcessingList count="1">
                <dataProcessing id="dp1">
                  <processingMethod order="0" softwareRef="sw1">
                    <cvParam cvRef="MS" accession="MS:1000544" name="Conversion to mzML" value=""/>
                  </processingMethod>
                </dataProcessing>
              </dataProcessingList>
              <run id="r" defaultInstrumentConfigurationRef="IC1">
                <spectrumList count="1" defaultDataProcessingRef="dp1">
                  <spectrum index="0" id="scan=1" defaultArrayLength="0">
                    <cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="1"/>
                    <binaryDataArrayList count="0"/>
                  </spectrum>
                </spectrumList>
              </run>
            </mzML>
            """;

    private static void write_minimal_mzml(Path path) throws IOException {
        Files.createDirectories(path.getParent());
        Files.writeString(path, MINIMAL_MZML);
    }

    private static void delete_tree(Path root) throws IOException {
        try (Stream<Path> paths = Files.walk(root)) {
            for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) {
                Files.deleteIfExists(path);
            }
        }
    }

    private static List<String> detail_json_names(Path output) throws IOException {
        try (Stream<Path> paths = Files.list(output)) {
            return paths.map(path -> path.getFileName().toString())
                    .filter(name -> name.endsWith("_acquisition.json"))
                    .sorted()
                    .toList();
        }
    }

    /**
     * Re-profiling into an output directory that already holds a window table from a DIA run must
     * not leave that table behind when the new run is not DIA. A stale table beside a DDA summary
     * reads as belonging to it.
     */
    @Test
    public void testNonDiaRunRemovesStaleIsolationWindowTable() throws Exception {
        Path root = Files.createTempDirectory("mzml_stale_windows");
        try {
            Path output = Files.createDirectories(root.resolve("output"));
            Path stale = output.resolve("sample_isolation_windows.tsv");
            Files.writeString(stale, "window_index\tmz_lower\n0\t400.0\n");

            MsRunMeta dda = new MsRunMeta();
            dda.ms_file = root.resolve("sample.mzML").toString();
            dda.acquisition_type = AcquisitionType.DDA;
            dda.write_detail_outputs(output.toString());

            assertTrue(Files.exists(output.resolve("sample_acquisition.json")));
            assertFalse(Files.exists(stale), "a window table from an earlier DIA run was left behind");
        } finally {
            delete_tree(root);
        }
    }

    @Test
    public void testDiaRunWritesIsolationWindowTable() throws Exception {
        Path root = Files.createTempDirectory("mzml_dia_windows");
        try {
            Path output = Files.createDirectories(root.resolve("output"));
            MsRunMeta dia = new MsRunMeta();
            dia.ms_file = root.resolve("sample.mzML").toString();
            dia.acquisition_type = AcquisitionType.DIA;
            dia.isolation_windows = fixed_windows(400, 8, 8, 4);
            dia.window_scan_counts = new int[]{3, 3, 3, 3};
            dia.write_detail_outputs(output.toString());

            Path table = output.resolve("sample_isolation_windows.tsv");
            assertTrue(Files.exists(table));
            assertEquals(Files.readAllLines(table).size(), 5, "header plus one row per window");
        } finally {
            delete_tree(root);
        }
    }

    @Test
    public void testFolderDetailNamesUseNormalizedRelativePaths() throws Exception {
        Path root = Files.createTempDirectory("mzml_detail_names");
        try {
            Path input = Files.createDirectories(root.resolve("input"));
            Path output = Files.createDirectories(root.resolve("output"));
            write_minimal_mzml(input.resolve("a").resolve("Run.mzML"));
            write_minimal_mzml(input.resolve("b").resolve("run.mzML"));
            write_minimal_mzml(input.resolve("deep").resolve("x").resolve("y").resolve("run.mzML"));

            MsRunMeta.main(new String[]{"-ms", input.toString(), "-o", output.toString(), "-detail"});

            // the source folder survives in the name, so runs stay distinguishable once flattened
            assertEquals(detail_json_names(output), List.of("a__Run_acquisition.json",
                    "b__run_acquisition.json", "deep__x__y__run_acquisition.json"));
        } finally {
            delete_tree(root);
        }
    }

    /**
     * Two different folder layouts can flatten to the same name, because the separator used when
     * flattening is itself a legal folder name. Both runs must still get their own files.
     */
    @Test
    public void testFolderDetailNamesDisambiguateFlattenedCollision() throws Exception {
        Path root = Files.createTempDirectory("mzml_flatten_collision");
        try {
            Path input = Files.createDirectories(root.resolve("input"));
            Path output = Files.createDirectories(root.resolve("output"));
            // "x/y/run.mzML" and "x__y/run.mzML" both flatten to "x__y__run"
            write_minimal_mzml(input.resolve("x").resolve("y").resolve("run.mzML"));
            write_minimal_mzml(input.resolve("x__y").resolve("run.mzML"));

            MsRunMeta.main(new String[]{"-ms", input.toString(), "-o", output.toString(), "-detail"});

            List<String> names = detail_json_names(output);
            assertEquals(names.size(), 2, "each run needs its own file: " + names);
            assertEquals(names, List.of("x__y__run_2_acquisition.json", "x__y__run_acquisition.json"));
        } finally {
            delete_tree(root);
        }
    }

    // ------------------------------------------------------------------ real files

    private static File resolve_testdata() {
        String p = System.getProperty("carafe.testdata");
        if (p == null || p.isEmpty()) {
            p = System.getenv("CARAFE_TESTDATA");
        }
        if (p == null || p.isEmpty()) {
            p = new File("testdata").getAbsolutePath();
        }
        return new File(p);
    }

    private static List<File> testdata_mzml_files() throws IOException {
        File dir = resolve_testdata();
        if (!dir.isDirectory()) {
            throw new SkipException("no test data directory at " + dir
                    + "; set -Dcarafe.testdata=<dir> or $CARAFE_TESTDATA");
        }
        List<File> files = MsRunMeta.find_mzml_files(dir);
        if (files.isEmpty()) {
            throw new SkipException("no .mzML file under " + dir
                    + "; set -Dcarafe.testdata=<dir> or $CARAFE_TESTDATA");
        }
        return files;
    }

    @Test
    public void testProfileRealMzml() throws IOException {
        for (File f : testdata_mzml_files()) {
            MsRunMeta meta = new MsRunMeta().profile(f.getAbsolutePath());
            System.out.println(meta.summary());
            assertNotEquals(meta.acquisition_type, AcquisitionType.UNKNOWN,
                    "could not classify " + f.getName());
            if (meta.acquisition_type == AcquisitionType.DIA) {
                assertFalse(meta.isolation_windows.isEmpty());
                assertNotEquals(meta.isolation_scheme, IsolationScheme.UNKNOWN);
                assertNotEquals(meta.isolation_scheme, IsolationScheme.NOT_APPLICABLE);
            }
        }
    }

    /**
     * The test that actually validates the sampling strategy: reading three slices must reach the
     * same verdict as reading every scan header.
     */
    @Test
    public void testSampledMatchesFull() throws IOException {
        for (File f : testdata_mzml_files()) {
            MsRunMeta sampled = new MsRunMeta();
            sampled.profile(f.getAbsolutePath());
            MsRunMeta full = new MsRunMeta();
            full.full_scan = true;
            full.profile(f.getAbsolutePath());
            System.out.printf(Locale.ROOT, "%s: sampled=%s/%s full=%s/%s%n", f.getName(),
                    sampled.acquisition_type, sampled.isolation_scheme,
                    full.acquisition_type, full.isolation_scheme);
            assertEquals(sampled.acquisition_type, full.acquisition_type,
                    "acquisition type differs for " + f.getName());
            assertEquals(sampled.isolation_scheme, full.isolation_scheme,
                    "isolation scheme differs for " + f.getName());
        }
    }
}
