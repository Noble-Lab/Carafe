package test.java.koina;

import main.java.koina.KoinaClient;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

/**
 * Live tests against the public Koina service. These make network calls; when Koina is
 * unreachable (e.g. offline CI) each test throws {@link SkipException} so the build is not failed.
 *
 * <p>Includes the {@code ELVISLIVESR} comparison: how similar are two different Koina predictors
 * (AlphaPepDeep vs Prosit) for the same precursor.</p>
 */
public class KoinaLiveTest {

    private static final String URL = "https://koina.wilhelmlab.org";
    private static final String PEPTIDE = "ELVISLIVESR";

    private KoinaClient client() {
        return new KoinaClient(URL);
    }

    /**
     * Skip every live test unless explicitly opted in (env {@code KOINA_LIVE_TESTS} or system
     * property {@code koina.live}). Keeps networked calls out of the default {@code mvn test} run so
     * the suite stays offline and deterministic; the per-test {@link SkipException} on a network
     * error remains as a second line of defense.
     */
    @BeforeMethod
    public void requireOptIn() {
        if (System.getenv("KOINA_LIVE_TESTS") == null
                && System.getProperty("koina.live") == null) {
            throw new SkipException(
                    "live Koina tests are opt-in; set KOINA_LIVE_TESTS=1 or -Dkoina.live=true to run");
        }
    }

    /** Build an annotation -> intensity map over fragments with positive intensity and m/z. */
    private static Map<String, Float> spectrum(KoinaClient.Ms2 ms2) {
        Map<String, Float> m = new HashMap<>();
        for (int i = 0; i < ms2.annotation.length; i++) {
            if (ms2.intensity[i] > 0 && ms2.mz[i] > 0) {
                m.merge(ms2.annotation[i], ms2.intensity[i], Float::sum);
            }
        }
        return m;
    }

    /** Cosine similarity of two spectra aligned by fragment annotation. */
    private static double cosine(Map<String, Float> a, Map<String, Float> b) {
        Set<String> keys = new TreeSet<>();
        keys.addAll(a.keySet());
        keys.addAll(b.keySet());
        double dot = 0, na = 0, nb = 0;
        for (String k : keys) {
            double x = a.getOrDefault(k, 0f);
            double y = b.getOrDefault(k, 0f);
            dot += x * y;
            na += x * x;
            nb += y * y;
        }
        return (na == 0 || nb == 0) ? 0 : dot / (Math.sqrt(na) * Math.sqrt(nb));
    }

    @Test
    public void elvislivesrPrositReturnsRealisticSpectrum() {
        try {
            KoinaClient c = client();
            String model = "Prosit_2020_intensity_HCD";
            Set<String> in = c.getModelInputNames(model);
            List<KoinaClient.Ms2> ms2 = c.inferMs2(model, List.of(PEPTIDE), List.of(2),
                    List.of(25.0f), null, in);
            Map<String, Float> spec = spectrum(ms2.get(0));
            Assert.assertTrue(spec.size() >= 5, "expected several fragments, got " + spec.size());
            boolean hasY = spec.keySet().stream().anyMatch(k -> k.startsWith("y"));
            boolean hasB = spec.keySet().stream().anyMatch(k -> k.startsWith("b"));
            Assert.assertTrue(hasY && hasB, "expected both y and b ions");

            float[] irt = c.inferRt("Prosit_2019_irt", List.of(PEPTIDE), c.getModelInputNames("Prosit_2019_irt"));
            Assert.assertEquals(irt.length, 1);
            Assert.assertTrue(Float.isFinite(irt[0]), "iRT should be finite");
            System.out.println("[KoinaLive] " + PEPTIDE + " Prosit: " + spec.size()
                    + " fragments, iRT=" + irt[0]);
        } catch (java.io.IOException | InterruptedException e) {
            throw new SkipException("Koina unreachable: " + e.getMessage());
        }
    }

    @Test
    public void alphaPepDeepVsPrositSimilarityForElvislivesr() {
        try {
            KoinaClient c = client();
            String ap = "AlphaPeptDeep_ms2_generic";
            String pr = "Prosit_2020_intensity_HCD";
            Set<String> apIn = c.getModelInputNames(ap);
            Set<String> prIn = c.getModelInputNames(pr);
            // Same precursor + collision energy; AlphaPepDeep also takes an instrument.
            Map<String, Float> apSpec = spectrum(c.inferMs2(ap, List.of(PEPTIDE), List.of(2),
                    List.of(25.0f), List.of("LUMOS"), apIn).get(0));
            Map<String, Float> prSpec = spectrum(c.inferMs2(pr, List.of(PEPTIDE), List.of(2),
                    List.of(25.0f), null, prIn).get(0));
            double cos = cosine(apSpec, prSpec);
            System.out.printf("[KoinaLive] %s AlphaPepDeep vs Prosit cosine = %.3f "
                    + "(AP frags=%d, Prosit frags=%d)%n", PEPTIDE, cos, apSpec.size(), prSpec.size());
            // Different models, but the same peptide's dominant y/b ions should align well.
            Assert.assertTrue(cos > 0.5,
                    "AlphaPepDeep and Prosit should be reasonably similar; cosine=" + cos);
        } catch (java.io.IOException | InterruptedException e) {
            throw new SkipException("Koina unreachable: " + e.getMessage());
        }
    }
}
