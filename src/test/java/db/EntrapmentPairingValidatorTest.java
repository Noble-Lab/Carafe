package test.java.db;

import main.java.db.EntrapmentPairingValidator;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * The library-against-manifest half of the pairing checks, driven by the real defect it exists to
 * catch: a Met-clip form that lands on only one side of a target/entrapment pair.
 *
 * <p>The quartet-integrity half is exercised through the generator's own write path, which now
 * refuses to emit a manifest that fails it.</p>
 */
public class EntrapmentPairingValidatorTest {

    /** target / p_target for pair 0 and pair 1, as the generator writes them. */
    private static Map<String, Integer> pairs() {
        Map<String, Integer> m = new HashMap<>();
        m.put("TGTAAAK", 0);
        m.put("MENTAAAK", 0);   // entrapment starts with M; its target does not
        m.put("MTGTBBBK", 1);   // both start with M
        m.put("MENTBBBK", 1);
        return m;
    }

    private static Map<String, String> types() {
        Map<String, String> m = new HashMap<>();
        m.put("TGTAAAK", "target");
        m.put("MENTAAAK", "p_target");
        m.put("MTGTBBBK", "target");
        m.put("MENTBBBK", "p_target");
        return m;
    }

    @Test
    public void testClippedEntrapmentWithNoTargetTwinIsReported() {
        // Exactly the shipped defect: the predictor clipped the M off both entrapments, but pair 0's
        // target never had an M to clip, so "ENTAAAK" has no target twin and a paired estimator
        // cannot consume it. Pair 1's clip is fine - both sides clipped.
        List<String> libEntrapment = Arrays.asList("MENTAAAK", "ENTAAAK", "MENTBBBK", "ENTBBBK");
        List<String> libTargets = Arrays.asList("TGTAAAK", "MTGTBBBK", "TGTBBBK");

        List<EntrapmentPairingValidator.Violation> v =
                EntrapmentPairingValidator.validateLibraryAgainstManifest(
                        libEntrapment, libTargets, pairs(), types());

        String msg = EntrapmentPairingValidator.describe(v);
        Assert.assertTrue(msg.contains("no target twin"),
                "the orphaned clipped entrapment must be reported, got: " + msg);
        Assert.assertTrue(msg.contains("ENTAAAK"),
                "the offending sequence should be named, got: " + msg);
    }

    @Test
    public void testWellFormedLibraryReportsNothing() {
        // Without this the test would pass equally well against a validator that flagged
        // everything, which would block every library build.
        List<String> libEntrapment = Arrays.asList("MENTAAAK", "MENTBBBK");
        List<String> libTargets = Arrays.asList("TGTAAAK", "MTGTBBBK");

        List<EntrapmentPairingValidator.Violation> v =
                EntrapmentPairingValidator.validateLibraryAgainstManifest(
                        libEntrapment, libTargets, pairs(), types());

        Assert.assertTrue(v.isEmpty(),
                "a manifest-consistent library must report no violations, got: "
                        + EntrapmentPairingValidator.describe(v));
    }

    /**
     * At an entrapment ratio below 1.0 most targets deliberately carry no entrapment, so the
     * manifest simply omits their {@code p_target} row. Those must not be reported: one violation
     * per unentrapped target would bury the real signal, and at r=0.1 that is 90% of the library.
     */
    @Test
    public void testRatioBelowOneDoesNotFlagDeliberatelyUnentrappedTargets() {
        // Pair 0 carries entrapment; pair 1 was not selected (no p_target row at all).
        Map<String, Integer> pairs = new HashMap<>();
        pairs.put("TGTAAAK", 0);
        pairs.put("MENTAAAK", 0);
        pairs.put("TGTBBBK", 1);
        Map<String, String> types = new HashMap<>();
        types.put("TGTAAAK", "target");
        types.put("MENTAAAK", "p_target");
        types.put("TGTBBBK", "target");

        List<EntrapmentPairingValidator.Violation> v =
                EntrapmentPairingValidator.validateLibraryAgainstManifest(
                        Arrays.asList("MENTAAAK"), Arrays.asList("TGTAAAK", "TGTBBBK"), pairs, types);

        Assert.assertTrue(v.isEmpty(),
                "a target the manifest never selected for entrapment is by design, not a defect; got: "
                        + EntrapmentPairingValidator.describe(v));
    }

    /**
     * The refusal is the default, and its message has to route the user to the RIGHT flag.
     * {@code -no_similarity_gate} cannot serve here: the collision check is exact string equality,
     * which that flag leaves on, so it would not clear the violation - and it yields a library
     * carrying the near-copy contamination the gate exists to remove.
     */
    @Test
    public void testViolationsThrowByDefaultAndNameTheCorrectFlag() {
        List<EntrapmentPairingValidator.Violation> v =
                EntrapmentPairingValidator.validateLibraryAgainstManifest(
                        Arrays.asList("MENTAAAK", "ENTAAAK"),
                        Arrays.asList("TGTAAAK"), pairs(), types());
        Assert.assertFalse(v.isEmpty(), "fixture must actually produce a violation to enforce on");

        try {
            EntrapmentPairingValidator.enforce(v, true);
            Assert.fail("a violation must refuse the manifest by default");
        } catch (IllegalStateException e) {
            Assert.assertTrue(e.getMessage().contains("-ignore_pairing_errors"),
                    "the error must name the flag that overrides it, got: " + e.getMessage());
            Assert.assertTrue(e.getMessage().contains("Do NOT use -no_similarity_gate"),
                    "the error must steer users away from the destructive flag, got: "
                            + e.getMessage());
        }
    }

    @Test
    public void testEscapeHatchDowngradesTheRefusalToAWarning() {
        List<EntrapmentPairingValidator.Violation> v =
                EntrapmentPairingValidator.validateLibraryAgainstManifest(
                        Arrays.asList("MENTAAAK", "ENTAAAK"),
                        Arrays.asList("TGTAAAK"), pairs(), types());
        Assert.assertFalse(v.isEmpty(), "fixture must actually produce a violation to enforce on");

        // Must not throw. Without an escape hatch, a check that is wrong about what the generator
        // guarantees leaves no way to build a library at all until the code is patched.
        EntrapmentPairingValidator.enforce(v, false);
    }

    @Test
    public void testCleanResultIsNeverRefusedUnderEitherSetting() {
        // Without this the throwing test would pass equally well against an enforce() that threw
        // unconditionally, which would block every build.
        List<EntrapmentPairingValidator.Violation> none =
                EntrapmentPairingValidator.validateLibraryAgainstManifest(
                        Arrays.asList("MENTAAAK", "MENTBBBK"),
                        Arrays.asList("TGTAAAK", "MTGTBBBK"), pairs(), types());
        Assert.assertTrue(none.isEmpty(), "control fixture must be clean");
        EntrapmentPairingValidator.enforce(none, true);
        EntrapmentPairingValidator.enforce(none, false);
    }

    @Test
    public void testTargetWithNoEntrapmentCoverageIsReported() {
        // The quieter half: nothing crashes, but this target's entrapment ratio is zero, so it
        // biases any entrapment-derived FDP downward. 45,537 of these shipped unnoticed.
        List<String> libEntrapment = Arrays.asList("MENTAAAK");
        List<String> libTargets = Arrays.asList("TGTAAAK", "MTGTBBBK");

        List<EntrapmentPairingValidator.Violation> v =
                EntrapmentPairingValidator.validateLibraryAgainstManifest(
                        libEntrapment, libTargets, pairs(), types());

        Assert.assertTrue(EntrapmentPairingValidator.describe(v).contains("no entrapment coverage"),
                "an uncovered target must be reported, got: " + EntrapmentPairingValidator.describe(v));
    }
}
