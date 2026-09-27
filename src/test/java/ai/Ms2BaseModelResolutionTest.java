package test.java.ai;

import main.java.ai.AIGear;
import main.java.input.CParameter;
import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.io.File;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Resolution of the MS2 base model used for prediction and as the fine-tuning starting point.
 * <p>
 * Precedence is: {@code -ms2_model none} forces the installed generic model, {@code -ms2_model tmt}
 * names the bundled model in the "models" folder next to the jar, and any other value is a path.
 * An empty result means "leave the command alone and use the installed generic model", so existing
 * runs that do not ask for a base model are unaffected.
 */
public class Ms2BaseModelResolutionTest {

    @BeforeClass
    public void setUp() {
        CParameter.init();
    }

    private String resolve(AIGear aiGear) throws Exception {
        Method method = AIGear.class.getDeclaredMethod("resolve_ms2_model_path");
        method.setAccessible(true);
        return (String) method.invoke(aiGear);
    }

    @Test
    public void testDefaultIsUnchanged() throws Exception {
        // No labelling and no explicit model: the command must not gain an --ms2_model flag,
        // so every existing non-TMT workflow behaves exactly as before.
        Assert.assertEquals(resolve(new AIGear()), "");
    }

    @Test
    public void testExplicitModelWins() throws Exception {
        AIGear aiGear = new AIGear();
        aiGear.ms2_model = "C:/somewhere/my_model.pt";
        Assert.assertEquals(resolve(aiGear), "C:/somewhere/my_model.pt",
                "a path must be used as given");
    }

    @Test
    public void testNoneForcesTheInstalledGenericModel() throws Exception {
        AIGear aiGear = new AIGear();
        aiGear.ms2_model = "none";
        Assert.assertEquals(resolve(aiGear), "",
                "-ms2_model none must opt out of the bundled model");
    }

    @Test
    public void testTmtKeywordSelectsTheBundledModel() throws Exception {
        // -ms2_model tmt names the bundled model rather than a path, so the GUI (and a user)
        // can ask for it without knowing where it was installed.
        AIGear aiGear = new AIGear();
        aiGear.ms2_model = "tmt";
        Path bundled = Paths.get(AIGear.get_jar_path(), "models", AIGear.TMT_MS2_MODEL_FILE);
        String resolved = resolve(aiGear);
        if (Files.isRegularFile(bundled)) {
            Assert.assertEquals(resolved, bundled.toAbsolutePath().toString());
        } else {
            Assert.assertEquals(resolved, "", "must fall back to the generic model when absent");
        }
    }

    @Test
    public void testBundledModelNameRecognition() throws Exception {
        Method method = AIGear.class.getDeclaredMethod("is_bundled_model_name", String.class);
        method.setAccessible(true);
        for (String name : new String[]{"tmt", "TMT", "TMTpro", "tmt6plex"}) {
            Assert.assertTrue((boolean) method.invoke(null, name), name + " should name the bundled model");
        }
        for (String name : new String[]{null, "", "none", "C:/models/my.pt", "generic"}) {
            Assert.assertFalse((boolean) method.invoke(null, name),
                    String.valueOf(name) + " must be treated as a path or opt-out, not a bundled name");
        }
    }

    /**
     * The NCE range the bundled model was trained on is used to warn about extrapolation, so it
     * must describe the data the model actually saw: the NIST CPTAC3 TMT library covers NCE
     * 30, 31, 32, 37 and 38 only.
     */
    @Test
    public void testTrainedNceRangeCoversTheTrainingData() {
        for (double nce : new double[]{30, 31, 32, 37, 38}) {
            Assert.assertTrue(nce >= AIGear.TMT_MS2_MODEL_MIN_NCE && nce <= AIGear.TMT_MS2_MODEL_MAX_NCE,
                    "NCE " + nce + " is present in the training data but outside the declared range");
        }
        // Carafe's own default NCE is outside it, which is what the warning exists to catch.
        Assert.assertTrue(27.0 < AIGear.TMT_MS2_MODEL_MIN_NCE,
                "the default NCE of 27 should fall outside the trained range");
    }

    @Test
    public void testBundledModelPathIsEmptyForAMissingFile() throws Exception {
        Method method = AIGear.class.getDeclaredMethod("resolve_bundled_model_path", String.class);
        method.setAccessible(true);
        Assert.assertEquals(method.invoke(null, "definitely_not_a_model_" + System.nanoTime() + ".pt"), "");
    }

    @Test
    public void testBundledModelIsLookedUpNextToTheJar() throws Exception {
        // The release layout is <jar dir>/models/<file>.
        Method method = AIGear.class.getDeclaredMethod("resolve_bundled_model_path", String.class);
        method.setAccessible(true);
        File models = new File(AIGear.get_jar_path(), "models");
        File probe = new File(models, "carafe_test_probe.pt");
        boolean created = false;
        try {
            if (!models.isDirectory() && !models.mkdirs()) {
                throw new org.testng.SkipException("cannot create " + models);
            }
            Files.write(probe.toPath(), new byte[]{0});
            created = true;
            Assert.assertEquals(method.invoke(null, probe.getName()),
                    probe.toPath().toAbsolutePath().toString());
        } finally {
            if (created) {
                Files.deleteIfExists(probe.toPath());
            }
        }
    }

    /**
     * The NCE range warning for the bundled TMT model.
     * <p>
     * It used to be gated on {@code -nce} having been passed, which meant a predict-only run that
     * never sets it extrapolated silently at the default 27 -- the case the GUI blocks and the
     * command line did not. The check now runs at prediction time against the effective NCE.
     */
    @Test
    public void testTmtNceRangeWarning() throws Exception {
        java.lang.reflect.Method m = AIGear.class.getDeclaredMethod(
                "nce_needs_tmt_range_warning", String.class, String.class, double.class);
        m.setAccessible(true);

        // The default NCE is outside the trained range, and nothing has to be passed to hit it.
        Assert.assertTrue((Boolean) m.invoke(null, "tmt", null, 27.0),
                "the default NCE 27 extrapolates and must warn even though -nce was never given");
        Assert.assertTrue((Boolean) m.invoke(null, "tmt", null, 39.0));

        // Inside the trained range, and on its boundaries, nothing is said.
        for (double nce : new double[]{30.0, 32.0, 38.0}) {
            Assert.assertFalse((Boolean) m.invoke(null, "tmt", null, nce), "NCE " + nce + " is in range");
        }

        // The generic model has no such range; this must not start warning on existing runs.
        Assert.assertFalse((Boolean) m.invoke(null, "", null, 27.0));
        Assert.assertFalse((Boolean) m.invoke(null, "/some/custom/model.pt", null, 27.0));
    }

    @Test
    public void testTmtCidKeywordSelectsTheBundledCidModel() throws Exception {
        AIGear aiGear = new AIGear();
        aiGear.ms2_model = "tmtcid";
        Path bundled = Paths.get(AIGear.get_jar_path(), "models", AIGear.TMT_CID_MS2_MODEL_FILE);
        String resolved = resolve(aiGear);
        if (Files.isRegularFile(bundled)) {
            Assert.assertEquals(resolved, bundled.toAbsolutePath().toString());
        } else {
            Assert.assertEquals(resolved, "", "must fall back to the generic model when absent");
        }
    }

    @Test
    public void testBundledCidModelNameRecognition() throws Exception {
        Method cid = AIGear.class.getDeclaredMethod("is_bundled_cid_model_name", String.class);
        cid.setAccessible(true);
        Method hcd = AIGear.class.getDeclaredMethod("is_bundled_model_name", String.class);
        hcd.setAccessible(true);
        for (String name : new String[]{"tmtcid", "TMTCID", "tmt_cid", "TMT-CID"}) {
            Assert.assertTrue((boolean) cid.invoke(null, name), name + " should name the bundled CID model");
            Assert.assertFalse((boolean) hcd.invoke(null, name), name + " must not name the HCD model");
        }
        for (String name : new String[]{null, "", "tmt", "tmtpro", "none", "cid", "tmtcid.pt", "C:/models/tmtcid.pt"}) {
            Assert.assertFalse((boolean) cid.invoke(null, name),
                    String.valueOf(name) + " must not name the bundled CID model");
        }
    }

    /** The bundled TMT CID model saw collision energy 35 only, so every other energy warns. */
    @Test
    public void testTmtCidNceWarning() throws Exception {
        java.lang.reflect.Method m = AIGear.class.getDeclaredMethod(
                "nce_needs_tmt_range_warning", String.class, String.class, double.class);
        m.setAccessible(true);
        Assert.assertEquals(AIGear.TMT_CID_MS2_MODEL_NCE, 35.0);
        Assert.assertFalse((Boolean) m.invoke(null, "tmtcid", null, 35.0), "35 is the trained energy");
        for (double nce : new double[]{27.0, 30.0, 34.0, 36.0}) {
            Assert.assertTrue((Boolean) m.invoke(null, "tmtcid", null, nce), "NCE " + nce + " must warn");
        }

        java.io.File dir = java.nio.file.Files.createTempDirectory("carafe_model").toFile();
        dir.deleteOnExit();
        java.io.File finetuned = new java.io.File(dir, "ms2_model.pt");
        java.nio.file.Files.write(finetuned.toPath(), new byte[]{1});
        finetuned.deleteOnExit();
        Assert.assertFalse((Boolean) m.invoke(null, "tmtcid", dir.getAbsolutePath(), 27.0),
                "a fine-tuned model takes precedence in ai_pred.py and learned this NCE");
    }

    /** A fine-tuned model has adapted to the user's own collision energy, so it is not warned about. */
    @Test
    public void testFineTunedModelSuppressesTheWarning() throws Exception {
        java.lang.reflect.Method m = AIGear.class.getDeclaredMethod(
                "nce_needs_tmt_range_warning", String.class, String.class, double.class);
        m.setAccessible(true);

        java.io.File dir = java.nio.file.Files.createTempDirectory("carafe_model").toFile();
        dir.deleteOnExit();
        Assert.assertTrue((Boolean) m.invoke(null, "tmt", dir.getAbsolutePath(), 27.0),
                "no fine-tuned model present, so the base model predicts and the range applies");

        java.io.File finetuned = new java.io.File(dir, "ms2_model.pt");
        java.nio.file.Files.write(finetuned.toPath(), new byte[]{1});
        finetuned.deleteOnExit();
        Assert.assertFalse((Boolean) m.invoke(null, "tmt", dir.getAbsolutePath(), 27.0),
                "a fine-tuned model takes precedence in ai_pred.py and learned this NCE");
    }
}
