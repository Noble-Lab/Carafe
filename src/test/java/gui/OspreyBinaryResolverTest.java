package test.java.gui;

import main.java.gui.OspreyBinaryResolver;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * Precedence tests for {@link OspreyBinaryResolver#choose}. The key invariant is that an Osprey
 * bundled with the installer wins over a saved/explicit path, so an MSI install never silently runs
 * a stale dev build a user once pointed at.
 *
 * <p>TestNG argument order {@code assertEquals(actual, expected, message)} matches the project's
 * other tests so these run under {@code mvn test}.
 */
public class OspreyBinaryResolverTest {

    private static final String BUNDLED = "C:\\install\\osprey\\win-x64\\Osprey.exe";
    private static final String SAVED = "D:\\dev\\pwiz\\Osprey\\bin\\Osprey.exe";
    private static final String HOME = "C:\\Users\\me\\.carafe\\osprey\\win-x64\\Osprey.exe";
    private static final String ON_PATH = "C:\\tools\\Osprey.exe";

    @Test
    public void bundledWinsOverEverythingElse() {
        // The MSI-install case: bundled must beat a stale saved path (the bug this guards against).
        Assert.assertEquals(OspreyBinaryResolver.choose(BUNDLED, SAVED, HOME, ON_PATH), BUNDLED);
        Assert.assertEquals(OspreyBinaryResolver.choose(BUNDLED, SAVED, null, null), BUNDLED);
    }

    @Test
    public void savedUsedOnlyWhenNoBundledBuild() {
        // Source / command-line run: no Osprey next to the jar, so the explicit path is honored.
        Assert.assertEquals(OspreyBinaryResolver.choose(null, SAVED, HOME, ON_PATH), SAVED);
    }

    @Test
    public void homeUsedWhenNoBundledOrSaved() {
        Assert.assertEquals(OspreyBinaryResolver.choose(null, null, HOME, ON_PATH), HOME);
    }

    @Test
    public void pathIsLastResort() {
        Assert.assertEquals(OspreyBinaryResolver.choose(null, null, null, ON_PATH), ON_PATH);
    }

    @Test
    public void emptyWhenNothingResolves() {
        Assert.assertEquals(OspreyBinaryResolver.choose(null, null, null, null), "");
        // Blank (not just null) candidates are skipped too.
        Assert.assertEquals(OspreyBinaryResolver.choose("", "   ", "", null), "");
    }
}
