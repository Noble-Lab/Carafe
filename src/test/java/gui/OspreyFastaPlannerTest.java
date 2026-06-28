package test.java.gui;

import main.java.gui.OspreyFastaPlanner;
import main.java.gui.OspreyFastaPlanner.Plan;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * Coverage for {@link OspreyFastaPlanner}: the rule that Osprey entrapment peptides go only
 * into the library-DB FASTA (which feeds the finetuned library / project search), never into the
 * training-DB FASTA (which drives AI fine-tuning), and that the two FASTAs are shared only when the
 * databases are identical AND no entrapment is requested.
 *
 * <p>TestNG style (argument order {@code assertEquals(actual, expected, message)}) to match the
 * project's other tests so these run under {@code mvn test}.</p>
 */
public class OspreyFastaPlannerTest {

    /** The core invariant: the training FASTA is NEVER built with entrapment, for any inputs. */
    @Test
    public void trainingFastaNeverHasEntrapment() {
        for (boolean sameDb : new boolean[]{true, false}) {
            for (boolean entrap : new boolean[]{true, false}) {
                Plan p = OspreyFastaPlanner.plan(sameDb, entrap);
                Assert.assertFalse(p.trainingEntrapment,
                        "training FASTA must never carry entrapment (sameDb=" + sameDb
                                + ", entrapmentRequested=" + entrap + ")");
            }
        }
    }

    /** Identical DBs, no entrapment: one shared target+decoy FASTA, no entrapment anywhere. */
    @Test
    public void sameDbNoEntrapment_sharesOneFasta() {
        Plan p = OspreyFastaPlanner.plan(true, false);
        Assert.assertTrue(p.shareTrainingFasta, "identical DBs with no entrapment should share one FASTA");
        Assert.assertFalse(p.trainingEntrapment, "no entrapment in training FASTA");
        Assert.assertFalse(p.libraryEntrapment, "no entrapment in library FASTA");
    }

    /** Identical DBs WITH entrapment: must split so the library FASTA can carry entrapment. */
    @Test
    public void sameDbWithEntrapment_buildsSeparateLibraryFastaWithEntrapment() {
        Plan p = OspreyFastaPlanner.plan(true, true);
        Assert.assertFalse(p.shareTrainingFasta,
                "entrapment forces a separate library FASTA even when the databases are identical");
        Assert.assertTrue(p.libraryEntrapment, "library FASTA must carry entrapment when requested");
        Assert.assertFalse(p.trainingEntrapment, "training FASTA must stay entrapment-free");
    }

    /** Different DBs: never shared (the library FASTA is always built separately). */
    @Test
    public void differentDb_neverShares() {
        Assert.assertFalse(OspreyFastaPlanner.plan(false, false).shareTrainingFasta,
                "different databases cannot share a FASTA");
        Assert.assertFalse(OspreyFastaPlanner.plan(false, true).shareTrainingFasta,
                "different databases cannot share a FASTA");
    }

    /** Different DBs with entrapment: library FASTA carries entrapment, training does not. */
    @Test
    public void differentDbWithEntrapment_onlyLibraryHasEntrapment() {
        Plan p = OspreyFastaPlanner.plan(false, true);
        Assert.assertTrue(p.libraryEntrapment, "library FASTA carries entrapment when requested");
        Assert.assertFalse(p.trainingEntrapment, "training FASTA stays entrapment-free");
    }

    /** Different DBs, no entrapment: separate builds, neither carries entrapment. */
    @Test
    public void differentDbNoEntrapment_neitherHasEntrapment() {
        Plan p = OspreyFastaPlanner.plan(false, false);
        Assert.assertFalse(p.shareTrainingFasta, "different databases are built separately");
        Assert.assertFalse(p.libraryEntrapment, "no entrapment requested");
        Assert.assertFalse(p.trainingEntrapment, "no entrapment in training FASTA");
    }
}
