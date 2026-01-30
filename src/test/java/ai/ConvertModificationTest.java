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
}
