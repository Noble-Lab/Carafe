package test.java.ai;

import main.java.ai.AIGear;
import main.java.input.CParameter;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.reflect.Method;

public class ConvertModificationTest {

    @Test
    public void testConvertModificationNTermAcetyl() throws Exception {
        AIGear aiGear = new AIGear();
        CParameter.terminal_char = "-";
        
        Method method = AIGear.class.getDeclaredMethod("convert_modification", String.class, String.class);
        method.setAccessible(true);

        String peptide = "ADPEVCCFITK";
        String modification = "Acetyl of protein N-term@0[42.010565];Carbamidomethyl of C@6[57.02146372057];Carbamidomethyl of C@7[57.02146372057]";

        String result = (String) method.invoke(aiGear, peptide, modification);
        
        System.out.println("Input Peptide: " + peptide);
        System.out.println("Input Modification: " + modification);
        System.out.println("Result: " + result);
        
        // Result: 5ADPEVCCFITK-
        Assert.assertEquals(result, "5ADPEVCCFITK-", "Should keep first residue if N-term acetylated and add '5' prefix");
    }

    @Test
    public void testConvertModificationOxidation() throws Exception {
        AIGear aiGear = new AIGear();
        CParameter.terminal_char = "-";

        Method method = AIGear.class.getDeclaredMethod("convert_modification", String.class, String.class);
        method.setAccessible(true);

        String peptide = "MAME";
        String modification = "Oxidation of M@1[15.99];Oxidation of M@3[15.99]";

        String result = (String) method.invoke(aiGear, peptide, modification);
        System.out.println("Result: " + result);
        Assert.assertEquals(result, "-1A1E-");
    }

    @Test
    public void testConvertModificationNTermAcetylAndOxidation() throws Exception {
        AIGear aiGear = new AIGear();
        CParameter.terminal_char = "-";

        Method method = AIGear.class.getDeclaredMethod("convert_modification", String.class, String.class);
        method.setAccessible(true);

        String peptide = "MADAEKNAVAEK";
        String modification = "Acetyl of protein N-term@0[42.0105646837];Oxidation of M@1[15.99491461956]";

        String result = (String) method.invoke(aiGear, peptide, modification);
        System.out.println("Result: " + result);
        
        // Expected: 51ADAEKNAVAEK-
        // '5' prefix for N-term acetyl
        // '1' for Oxidation of M at pos 1
        Assert.assertEquals(result, "51ADAEKNAVAEK-", "Should handle both N-term acetyl and Oxidation correctly");
    }

    @Test
    public void testSkylineProteinNTermAcetylFormatting() throws Exception {
        AIGear aiGear = new AIGear();

        Method peptideMethod = AIGear.class.getDeclaredMethod("get_modified_peptide_skyline", String.class, String.class, String.class);
        peptideMethod.setAccessible(true);

        String peptide = "AANSGLDSK";
        String mods = "Acetyl@Protein_N-term";
        String modSites = "0";

        String result = (String) peptideMethod.invoke(aiGear, peptide, mods, modSites);

        Assert.assertEquals(result, "A[42.0105646837]ANSGLDSK", "Skyline peptideModSeq should place protein N-term acetylation on the first residue.");
    }

    @Test
    public void testSkylineProteinNTermAcetylAndFirstResidueOxidationFormatting() throws Exception {
        AIGear aiGear = new AIGear();

        Method peptideMethod = AIGear.class.getDeclaredMethod("get_modified_peptide_skyline", String.class, String.class, String.class);
        peptideMethod.setAccessible(true);

        String peptide = "MAAAAAAAAAGAAGGR";
        String mods = "Acetyl@Protein_N-term;Oxidation@M";
        String modSites = "0;1";

        String result = (String) peptideMethod.invoke(aiGear, peptide, mods, modSites);
        System.out.println("Result: " + result);

        Assert.assertEquals(result, "M[58.00547930326]AAAAAAAAAGAAGGR", "Skyline peptideModSeq should combine protein N-term acetylation with first-residue oxidation.");
    }

    @Test
    public void testSkylineProteinNTermAcetylPosition() throws Exception {
        AIGear aiGear = new AIGear();

        Method positionMethod = AIGear.class.getDeclaredMethod("get_skyline_modification_position", String.class, int.class);
        positionMethod.setAccessible(true);

        int result = (int) positionMethod.invoke(aiGear, "Acetyl@Protein_N-term", 0);

        Assert.assertEquals(result, 1, "Skyline Modifications.position should store protein N-term acetylation at position 1.");
    }
}
