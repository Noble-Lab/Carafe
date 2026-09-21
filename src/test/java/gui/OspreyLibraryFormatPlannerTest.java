package test.java.gui;

import main.java.gui.OspreyLibraryFormatPlanner;
import main.java.gui.OspreyLibraryFormatPlanner.Plan;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * Coverage for {@link OspreyLibraryFormatPlanner}: the finetuned-library output-format decision for
 * the Osprey workflows. DIA-NN stays a TSV; Skyline yields a .blib deliverable, with the .tsv kept
 * only for Workflow 5's project search.
 */
public class OspreyLibraryFormatPlannerTest {

    @Test
    public void diannIsUnchangedTsvForBothWorkflows() {
        for (boolean endToEnd : new boolean[] { false, true }) {
            Plan p = OspreyLibraryFormatPlanner.plan("DIA-NN", endToEnd);
            Assert.assertEquals(p.lfType, "DIA-NN");
            Assert.assertFalse(p.blib);
            Assert.assertTrue(p.tsv);
            Assert.assertEquals(p.reconcileFileName, OspreyLibraryFormatPlanner.LIB_TSV);
            Assert.assertEquals(p.skipCheckFileName, OspreyLibraryFormatPlanner.LIB_TSV);
            Assert.assertEquals(p.searchFileName, OspreyLibraryFormatPlanner.LIB_TSV);
        }
    }

    @Test
    public void workflow4SkylineEmitsBlibOnly() {
        // Workflow 4 (no project search): blib is the whole deliverable, reconciler reads the blib.
        Plan p = OspreyLibraryFormatPlanner.plan("Skyline", false);
        Assert.assertEquals(p.lfType, "Skyline");
        Assert.assertTrue(p.blib);
        Assert.assertFalse(p.tsv, "no TSV needed when there is no project search");
        Assert.assertEquals(p.reconcileFileName, OspreyLibraryFormatPlanner.LIB_BLIB);
        Assert.assertEquals(p.skipCheckFileName, OspreyLibraryFormatPlanner.LIB_BLIB);
    }

    @Test
    public void workflow5SkylineEmitsBlibAndTsv() {
        // Workflow 5 (end-to-end): blib deliverable + TSV for the Osprey project search + reconcile.
        Plan p = OspreyLibraryFormatPlanner.plan("Skyline", true);
        Assert.assertEquals(p.lfType, "Skyline,DIA-NN");
        Assert.assertTrue(p.blib);
        Assert.assertTrue(p.tsv);
        Assert.assertEquals(p.reconcileFileName, OspreyLibraryFormatPlanner.LIB_TSV,
                "reconcile against the TSV when both are produced");
        Assert.assertEquals(p.skipCheckFileName, OspreyLibraryFormatPlanner.LIB_TSV,
                "skip check on the TSV the project search needs, so a missing TSV re-runs lib2");
        Assert.assertEquals(p.searchFileName, OspreyLibraryFormatPlanner.LIB_TSV);
    }

    @Test
    public void nonSkylineFormatsAndNullAreTsv() {
        Assert.assertFalse(OspreyLibraryFormatPlanner.plan("EncyclopeDIA", false).blib);
        Assert.assertFalse(OspreyLibraryFormatPlanner.plan("mzSpecLib", true).blib);
        Assert.assertFalse(OspreyLibraryFormatPlanner.plan(null, false).blib);
        Assert.assertEquals(OspreyLibraryFormatPlanner.plan(null, false).lfType, "DIA-NN");
    }

    @Test
    public void skylineDetectionIsCaseInsensitive() {
        Assert.assertTrue(OspreyLibraryFormatPlanner.plan("skyline", false).blib);
        Assert.assertTrue(OspreyLibraryFormatPlanner.plan("SKYLINE", true).blib);
    }
}
