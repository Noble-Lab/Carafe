package test.java.gui;

import main.java.gui.OspreyMsInputPlanner;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.File;
import java.util.List;

/**
 * Coverage for {@link OspreyMsInputPlanner#allInSameDirectory}: the guard that stops the Osprey
 * training path from silently dropping files selected from different folders.
 */
public class OspreyMsInputPlannerTest {

    private static String p(String... parts) {
        return String.join(File.separator, parts);
    }

    @Test
    public void singleFileOrEmptyIsTriviallySameDirectory() {
        Assert.assertTrue(OspreyMsInputPlanner.allInSameDirectory(List.of()));
        Assert.assertTrue(OspreyMsInputPlanner.allInSameDirectory(List.of(p("data", "a.mzML"))));
    }

    @Test
    public void allInSameFolderIsAccepted() {
        Assert.assertTrue(OspreyMsInputPlanner.allInSameDirectory(List.of(
                p("data", "run", "a.mzML"),
                p("data", "run", "b.mzML"),
                p("data", "run", "c.mzML"))));
    }

    @Test
    public void filesFromDifferentFoldersAreRejected() {
        Assert.assertFalse(OspreyMsInputPlanner.allInSameDirectory(List.of(
                p("data", "run1", "a.mzML"),
                p("data", "run2", "b.mzML"))));
    }

    @Test
    public void bareFileNameWithNoParentIsRejectedForMultiple() {
        // A path with no parent directory can't be collapsed to a folder alongside another file.
        Assert.assertFalse(OspreyMsInputPlanner.allInSameDirectory(List.of("a.mzML", p("data", "b.mzML"))));
    }
}
