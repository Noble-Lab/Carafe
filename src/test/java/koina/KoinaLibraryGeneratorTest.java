package test.java.koina;

import main.java.koina.KoinaLibraryGenerator;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.List;

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
}
