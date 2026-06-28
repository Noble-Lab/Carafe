package test.java.gui;

import main.java.gui.OspreyFdrBenchPlanner;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.File;

/**
 * Coverage for {@link OspreyFdrBenchPlanner}: the rule that an OspreySharp FDRBench input TSV is
 * emitted only for the project search (workflow 5) when entrapment is enabled, under a {@code
 * FDRBench} subfolder of the search output directory, and never for the training search.
 *
 * <p>TestNG style (argument order {@code assertEquals(actual, expected, message)}) to match the
 * project's other tests so these run under {@code mvn test}.</p>
 */
public class OspreyFdrBenchPlannerTest {

    @Test
    public void projectSearchWithEntrapment_emitsPathUnderFdrBenchFolder() {
        String dir = "C:" + File.separator + "out" + File.separator + "osprey_project";
        String path = OspreyFdrBenchPlanner.fdrBenchInputPath(true, true, dir);
        String expected = dir + File.separator + OspreyFdrBenchPlanner.FDRBENCH_DIR
                + File.separator + OspreyFdrBenchPlanner.FDRBENCH_INPUT_TSV;
        Assert.assertEquals(path, expected, "project + entrapment writes under the FDRBench folder");
    }

    @Test
    public void trainingSearch_neverEmits() {
        // Even with entrapment on, the training search (drives fine-tuning) must not get FDRBench input.
        Assert.assertNull(OspreyFdrBenchPlanner.fdrBenchInputPath(true, false, "/out/osprey_train"),
                "training search must never emit an FDRBench input");
    }

    @Test
    public void noEntrapment_emitsNothing() {
        Assert.assertNull(OspreyFdrBenchPlanner.fdrBenchInputPath(false, true, "/out/osprey_project"),
                "no entrapment means no FDRBench input (no entrapment hits to score)");
    }

    @Test
    public void missingOutDir_emitsNothing() {
        Assert.assertNull(OspreyFdrBenchPlanner.fdrBenchInputPath(true, true, null),
                "null output dir yields no path");
        Assert.assertNull(OspreyFdrBenchPlanner.fdrBenchInputPath(true, true, "   "),
                "blank output dir yields no path");
    }
}
