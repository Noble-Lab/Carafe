package test.java.dia;

import main.java.dia.DIAMap;
import main.java.dia.IsolationWindow;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.util.List;

/**
 * Tests for {@link DIAMap#get_isolation_window} / {@link DIAMap#get_isolation_windows}: which DIA
 * isolation window(s) a precursor m/z falls into. This bucketing is what assigns each precursor to
 * the per-window index during finetune and decoy pairing -- the exact membership logic the SEA-AD
 * pairing diagnosis turned on -- so it is exercised here directly from a hand-built
 * {@code isolationWindowMap}, without needing to parse an mzML.
 */
public class DIAMapTest {

    /** A DIAMap whose isolation windows are the given {@code [lower, upper]} pairs. */
    private static DIAMap mapWith(double[]... windows) {
        DIAMap m = new DIAMap();
        for (double[] w : windows) {
            String id = IsolationWindow.generate_id(w[0], w[1]);
            m.meta.isolationWindowMap.put(id, new IsolationWindow(w[0], w[1]));
            m.target_isolation_wins.add(id);
        }
        return m;
    }

    @Test
    public void precursorInOneWindowResolvesToThatWindow() {
        DIAMap m = mapWith(new double[] { 400, 404 }, new double[] { 404, 408 }, new double[] { 408, 412 });
        Assert.assertEquals(m.get_isolation_window(402.0), IsolationWindow.generate_id(400, 404));
        Assert.assertEquals(m.get_isolation_window(406.0), IsolationWindow.generate_id(404, 408));
        Assert.assertEquals(m.get_isolation_windows(410.0),
                List.of(IsolationWindow.generate_id(408, 412)));
    }

    @Test
    public void precursorOutsideAllWindowsResolvesToEmpty() {
        DIAMap m = mapWith(new double[] { 400, 404 }, new double[] { 404, 408 });
        Assert.assertEquals(m.get_isolation_window(500.0), "");
        Assert.assertTrue(m.get_isolation_windows(500.0).isEmpty());
    }

    @Test
    public void boundaryMzIsInclusiveInBothAdjacentWindows() {
        DIAMap m = mapWith(new double[] { 400, 404 }, new double[] { 404, 408 });
        // 404 is the shared A-end / B-start boundary -> belongs to both (inclusive comparison).
        Assert.assertEquals(m.get_isolation_windows(404.0).size(), 2);
        Assert.assertFalse(m.get_isolation_window(404.0).isEmpty());
    }

    @Test
    public void overlappingStaggeredWindowsAreAllReported() {
        // Staggered/overlapping acquisition: 403 falls in both A[400,404] and B[402,406].
        DIAMap m = mapWith(new double[] { 400, 404 }, new double[] { 402, 406 });
        List<String> wins = m.get_isolation_windows(403.0);
        Assert.assertEquals(wins.size(), 2, "a precursor in the overlap must report both windows");
        Assert.assertTrue(wins.contains(IsolationWindow.generate_id(400, 404)));
        Assert.assertTrue(wins.contains(IsolationWindow.generate_id(402, 406)));
    }
}
