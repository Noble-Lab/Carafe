package test.java.koina;

import main.java.koina.KoinaClient;
import main.java.koina.KoinaLibraryGenerator;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * Tests for {@link KoinaLibraryGenerator}'s peptidoform enumeration (no network). The private
 * {@code generatePeptidoforms} method and the private {@code Peptidoform} fields are accessed via
 * reflection.
 */
public class KoinaLibraryGeneratorTest {

    private static List<?> forms(String pep, boolean fixC, boolean varM, int maxVar) throws Exception {
        KoinaLibraryGenerator.Config cfg = new KoinaLibraryGenerator.Config();
        cfg.fixedCarbamidomethylC = fixC;
        cfg.variableOxidationM = varM;
        cfg.maxVarMods = maxVar;
        Method m = KoinaLibraryGenerator.class.getDeclaredMethod(
                "generatePeptidoforms", String.class, KoinaLibraryGenerator.Config.class);
        m.setAccessible(true);
        return (List<?>) m.invoke(null, pep, cfg);
    }

    private static String field(Object form, String name) throws Exception {
        Field f = form.getClass().getDeclaredField(name);
        f.setAccessible(true);
        return String.valueOf(f.get(form));
    }

    private static double mass(Object form) throws Exception {
        Field f = form.getClass().getDeclaredField("neutralMass");
        f.setAccessible(true);
        return (double) f.get(form);
    }

    @Test
    public void unmodifiedPeptideYieldsOneForm() throws Exception {
        List<?> forms = forms("PEPTIDEK", true, true, 1);
        Assert.assertEquals(forms.size(), 1);
        Assert.assertEquals(field(forms.get(0), "proforma"), "PEPTIDEK");
        Assert.assertEquals(field(forms.get(0), "diann"), "PEPTIDEK");
    }

    @Test
    public void cysIsAlwaysCarbamidomethylatedAndMetOxidationIsVariable() throws Exception {
        // ELVISCMK has one C and one M -> 2 forms (no Met-ox, Met-ox), both with fixed Cys mod.
        List<?> forms = forms("ELVISCMK", true, true, 1);
        Assert.assertEquals(forms.size(), 2);
        int oxCount = 0;
        for (Object f : forms) {
            String pf = field(f, "proforma");
            String dn = field(f, "diann");
            Assert.assertTrue(pf.contains("C[UNIMOD:4]"), "Cys must be carbamidomethyl: " + pf);
            Assert.assertTrue(dn.contains("C(UniMod:4)"), "DIA-NN Cys mod: " + dn);
            if (pf.contains("M[UNIMOD:35]")) {
                oxCount++;
            }
        }
        Assert.assertEquals(oxCount, 1, "exactly one form should carry Met oxidation");
    }

    @Test
    public void variableMaxModsControlsCombinations() throws Exception {
        // MEMK has two M. With maxVar=2 -> {none, M1, M2, M1+M2} = 4 forms.
        Assert.assertEquals(forms("MEMK", false, true, 2).size(), 4);
        // With maxVar=1 -> {none, M1, M2} = 3 forms.
        Assert.assertEquals(forms("MEMK", false, true, 1).size(), 3);
        // Variable oxidation disabled -> 1 form.
        Assert.assertEquals(forms("MEMK", false, false, 2).size(), 1);
    }

    @Test
    public void oxidationAddsExpectedMass() throws Exception {
        List<?> noVar = forms("PEPMK", false, false, 1); // no mods
        List<?> withVar = forms("PEPMK", false, true, 1); // {none, ox}
        double base = mass(noVar.get(0));
        double oxMass = -1;
        for (Object f : withVar) {
            if (field(f, "proforma").contains("M[UNIMOD:35]")) {
                oxMass = mass(f);
            }
        }
        Assert.assertTrue(oxMass > 0);
        Assert.assertEquals(oxMass - base, 15.994915, 1e-4, "oxidation should add ~15.9949 Da");
    }

    // ---- End-to-end: FASTA -> enumerate -> infer (stubbed) -> write DIA-NN library TSV ----

    private static void invokeRun(KoinaLibraryGenerator.Config cfg, KoinaClient client) throws Exception {
        Method m = KoinaLibraryGenerator.class.getDeclaredMethod(
                "run", KoinaLibraryGenerator.Config.class, KoinaClient.class);
        m.setAccessible(true);
        m.invoke(null, cfg, client);
    }

    /**
     * A {@link KoinaClient} that returns canned predictions instead of hitting the network, so the
     * full library-generation pipeline (FASTA read, peptidoform/charge enumeration, precursor m/z
     * filtering, fragment sorting/normalization, TSV writing) can be exercised offline. Each
     * precursor gets three fragments; the top (y2+1) carries the maximum intensity so it normalizes
     * to RelativeIntensity 1.0.
     */
    private static final class StubKoinaClient extends KoinaClient {
        StubKoinaClient() {
            super("http://stub.invalid");
        }

        @Override
        public Set<String> getModelInputNames(String model) {
            return new HashSet<>(Arrays.asList("peptide_sequences", "precursor_charges"));
        }

        @Override
        public List<Ms2> inferMs2(String model, List<String> sequences, List<Integer> charges,
                List<Float> ces, List<String> instruments, Set<String> availableInputs) {
            List<Ms2> out = new ArrayList<>();
            for (int i = 0; i < sequences.size(); i++) {
                out.add(new Ms2(
                        new String[] { "y1+1", "y2+1", "b1+1" },
                        new float[] { 250.1f, 400.2f, 300.3f },
                        new float[] { 0.5f, 1.0f, 0.2f }));
            }
            return out;
        }

        @Override
        public float[] inferRt(String model, List<String> sequences, Set<String> availableInputs) {
            float[] rt = new float[sequences.size()];
            Arrays.fill(rt, 12.34f);
            return rt;
        }
    }

    @Test
    public void endToEnd_writesDiannLibraryFromInjectedClient() throws Exception {
        // A target + a decoy_ entry (anagram). Both ~927 Da: z=2 m/z ~465 is in [400,900], z=3 is
        // out, so each peptide yields exactly one precursor.
        Path fasta = Files.createTempFile("koina_in", ".fasta");
        Files.write(fasta,
                (">sp|P1|TEST_HUMAN\nPEPTIDEK\n>decoy_sp|P1|TEST_HUMAN\nKEDITPEP\n")
                        .getBytes(StandardCharsets.UTF_8));
        Path outTsv = Files.createTempFile("koina_lib", ".tsv");

        KoinaLibraryGenerator.Config cfg = new KoinaLibraryGenerator.Config();
        cfg.peptideFasta = fasta.toString();
        cfg.outputTsv = outTsv.toString();
        cfg.ms2Model = "stub_ms2";
        cfg.rtModel = "stub_rt";
        cfg.minCharge = 2;
        cfg.maxCharge = 3;

        invokeRun(cfg, new StubKoinaClient());

        List<String> lines = Files.readAllLines(outTsv, StandardCharsets.UTF_8);
        Assert.assertTrue(lines.get(0).startsWith(
                "ModifiedPeptide\tStrippedPeptide\tPrecursorMz\tPrecursorCharge\tTr_recalibrated\t"
                        + "ProteinID\tDecoy\tFragmentMz\tRelativeIntensity\tFragmentType"),
                "header: " + lines.get(0));

        List<String> rows = lines.subList(1, lines.size());
        // 2 precursors (z=2 only) x 3 stub fragments = 6 fragment rows.
        Assert.assertEquals(rows.size(), 6, "expected 6 fragment rows");

        int targetRows = 0, decoyRows = 0, topFragRows = 0;
        for (String r : rows) {
            String[] c = r.split("\t", -1);
            Assert.assertEquals(c.length, 13, "13 columns: " + r);
            Assert.assertEquals(c[3], "2", "precursor charge");
            Assert.assertEquals(c[4], "12.3400", "iRT from stub");
            String stripped = c[1];
            String decoy = c[6];
            if ("PEPTIDEK".equals(stripped)) {
                Assert.assertEquals(decoy, "0", "target must not be flagged decoy");
                Assert.assertEquals(c[5], "sp|P1|TEST_HUMAN", "target protein id");
                targetRows++;
            } else if ("KEDITPEP".equals(stripped)) {
                Assert.assertEquals(decoy, "1", "decoy_ accession must be flagged decoy");
                decoyRows++;
            } else {
                Assert.fail("unexpected stripped peptide: " + stripped);
            }
            // Top fragment (y2+1, stub intensity 1.0) normalizes to RelativeIntensity 1.0.
            if ("1.000000".equals(c[8])) {
                Assert.assertEquals(c[9], "y", "top fragment type");
                Assert.assertEquals(c[10], "2", "top fragment number");
                Assert.assertEquals(c[11], "1", "top fragment charge");
                topFragRows++;
            }
        }
        Assert.assertEquals(targetRows, 3, "3 target fragment rows");
        Assert.assertEquals(decoyRows, 3, "3 decoy fragment rows");
        Assert.assertEquals(topFragRows, 2, "exactly one normalized top fragment per precursor");
    }

    @Test
    public void endToEnd_writesDotDecimalsUnderACommaLocale() throws Exception {
        // The library TSV must use '.' as the decimal separator regardless of the JVM default
        // locale; under a comma locale, unlocalized String.format would emit "12,3400" and break
        // every downstream parser.
        Locale prev = Locale.getDefault();
        try {
            Locale.setDefault(Locale.GERMANY); // comma decimal separator
            Path fasta = Files.createTempFile("koina_loc", ".fasta");
            Files.write(fasta, ">sp|P1|TEST_HUMAN\nPEPTIDEK\n".getBytes(StandardCharsets.UTF_8));
            Path outTsv = Files.createTempFile("koina_loc_lib", ".tsv");

            KoinaLibraryGenerator.Config cfg = new KoinaLibraryGenerator.Config();
            cfg.peptideFasta = fasta.toString();
            cfg.outputTsv = outTsv.toString();
            cfg.ms2Model = "stub_ms2";
            cfg.rtModel = "stub_rt";
            cfg.minCharge = 2;
            cfg.maxCharge = 3;

            invokeRun(cfg, new StubKoinaClient());

            for (String line : Files.readAllLines(outTsv, StandardCharsets.UTF_8)) {
                Assert.assertFalse(line.matches(".*\\d,\\d.*"),
                        "a decimal comma leaked into the TSV under a comma locale: " + line);
            }
        } finally {
            Locale.setDefault(prev);
        }
    }
}
