package test.java.dia;

import main.java.dia.IsolationWindow;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * Tests for {@link IsolationWindow#generate_id}, the isolation-window bucket key. The library
 * decoy-pairing path depends on this id being identical whether it is computed from a scan's
 * precursor range (when indexing) or from a window definition (when bucketing) -- a mismatch is
 * what made the SEA-AD pairing diagnosis subtle -- so its rounding behavior is pinned here.
 */
public class IsolationWindowTest {

    @Test
    public void generateIdIsDeterministic() {
        Assert.assertEquals(IsolationWindow.generate_id(696.567, 700.567),
                IsolationWindow.generate_id(696.567, 700.567));
    }

    @Test
    public void generateIdRoundsToTenthMz() {
        // Each bound is rounded to 0.1 m/z (x10, round): 696.567 -> 6966, 700.567 -> 7006.
        Assert.assertEquals(IsolationWindow.generate_id(696.567, 700.567), "6966_7006");
    }

    @Test
    public void floatJitterWithinRoundingDoesNotChangeTheId() {
        // Two reads of the "same" window that differ only by sub-0.05 m/z float jitter must map to
        // the same id -- this is why the per-scan index and the bucketing map agree.
        Assert.assertEquals(IsolationWindow.generate_id(696.5670001, 700.5669998),
                IsolationWindow.generate_id(696.567, 700.567));
    }

    @Test
    public void differentWindowsGetDifferentIds() {
        Assert.assertNotEquals(IsolationWindow.generate_id(700.567, 704.567),
                IsolationWindow.generate_id(696.567, 700.567));
        // A shift past the 0.1 rounding bucket changes the id (696.567 -> 6966, 696.70 -> 6967).
        Assert.assertNotEquals(IsolationWindow.generate_id(696.70, 700.567),
                IsolationWindow.generate_id(696.567, 700.567));
    }

    @Test
    public void instanceIdMatchesStaticGenerator() {
        IsolationWindow w = new IsolationWindow(500.25, 504.25);
        Assert.assertEquals(w.id, IsolationWindow.generate_id(500.25, 504.25));
    }
}
