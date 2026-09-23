package test.java.ai;

import main.java.ai.AIGear;
import main.java.input.CParameter;
import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.lang.reflect.Method;

/**
 * Modification parsing on the training path, where Carafe builds training data from a PSM table
 * and MS data.
 * <p>
 * Every TMT peptide carries a peptide N-terminal label, and neither parser handled one: the DIA-NN
 * parser read modification codes back one character at a time (codes are modification ids, so any
 * id of 10 or more, including TMT on K, ended in "Unrecognized modification" and System.exit), and
 * the RT-data encoder indexed aa[pos - 1] for a position-0 modification. Fine-tuning on TMT data was
 * therefore impossible through Carafe, from both the command line and the GUI.
 * <p>
 * The "unchanged" tests pin the output for inputs that already worked, recorded before the fix,
 * so that the fix provably changes nothing but the inputs that used to crash or exit.
 */
public class NTermTrainingModificationTest {

    private AIGear aiGear;
    private Method rtEncode;

    @BeforeClass
    public void setUp() throws Exception {
        CParameter.init();
        CParameter.varMods = "5"; // protein N-term acetyl as a variable modification
        aiGear = new AIGear();
        aiGear.load_mod_map();
        rtEncode = AIGear.class.getDeclaredMethod("convert_modification", String.class, String.class);
        rtEncode.setAccessible(true);
    }

    private String diann(String modSeq, String peptide) {
        return aiGear.get_modification_diann(modSeq, peptide);
    }

    private String alphabase(String modification) {
        String[] r = aiGear.convert_modification(modification);
        return r[0] + " | " + r[1];
    }

    private String rt(String peptide, String modification) throws Exception {
        return (String) rtEncode.invoke(aiGear, peptide, modification);
    }

    @Test
    public void testDiannUnchangedForInputsThatAlreadyWorked() {
        Assert.assertEquals(diann("AAAAC(UniMod:4)LDK", "AAAACLDK"), "Carbamidomethyl of C@5[57.02146372057]");
        Assert.assertEquals(diann("AGEVLNQPM(UniMod:35)MMAAR", "AGEVLNQPMMMAAR"), "Oxidation of M@9[15.99491461956]");
        Assert.assertEquals(diann("AAAAAAAATMALAAPS(UniMod:21)SPTPESPTMLTK", "AAAAAAAATMALAAPSSPTPESPTMLTK"),
                "Phospho of S@16[79.96633052074999]");
        Assert.assertEquals(diann("(UniMod:1)AAVTLHLR", "AAVTLHLR"), "Acetyl of protein N-term@0[42.010565]");
        Assert.assertEquals(diann("(UniMod:1)AAC(UniMod:4)M(UniMod:35)K", "AACMK"),
                "Acetyl of protein N-term@0[42.010565];Carbamidomethyl of C@3[57.02146372057];"
                        + "Oxidation of M@4[15.99491461956]");
        Assert.assertEquals(diann("PEPTIDEK", "PEPTIDEK"), "-");
    }

    @Test
    public void testDiannTmtLabels() {
        String tmt = diann("(UniMod:737)AAK(UniMod:737)", "AAK");
        Assert.assertEquals(tmt, "TMT6plex of N-term@0[229.162932];TMT 6-plex of K@3[229.16293213472]");
        Assert.assertEquals(alphabase(tmt), "TMT6plex@Any N-term;TMT6plex@K | 0;3");

        String pro = diann("(UniMod:2016)AC(UniMod:4)K(UniMod:2016)", "ACK");
        Assert.assertEquals(alphabase(pro), "TMTpro@Any N-term;Carbamidomethyl@C;TMTpro@K | 0;2;3");

        // Residue positions stay 1-based with the N-terminal label in front.
        Assert.assertEquals(alphabase(diann("(UniMod:737)M(UniMod:35)PEPK(UniMod:737)", "MPEPK")),
                "TMT6plex@Any N-term;Oxidation@M;TMT6plex@K | 0;1;5");

        // Protein N-term acetyl keeps its route when a residue carries a TMT label.
        Assert.assertEquals(alphabase(diann("(UniMod:1)AAK(UniMod:737)", "AAK")),
                "Acetyl@Protein_N-term;TMT6plex@K | 0;3");
    }

    @Test
    public void testRtEncodingUnchangedForInputsThatAlreadyWorked() throws Exception {
        Assert.assertEquals(rt("PEPMTIDEK", "Oxidation of M@4[15.9949]"), "-PEP1TIDEK-");
        Assert.assertEquals(rt("AACLDK", "Carbamidomethyl of C@3[57.0215]"), "-AACLDK-");
        Assert.assertEquals(rt("AASTK", "Phospho of S@3[79.9663]"), "-AA2TK-");
        Assert.assertEquals(rt("AAVTLHLR", "Acetyl of protein N-term@0[42.0106]"), "5AAVTLHLR-");
        Assert.assertEquals(rt("PEPTIDEK", "-"), "-PEPTIDEK-");
    }

    @Test
    public void testRtEncodingPeptideNTermLabel() throws Exception {
        // Used to throw ArrayIndexOutOfBoundsException: Index -1.
        Assert.assertEquals(rt("AAK", "TMT6plex of N-term@0[229.1629];TMT 6-plex of K@3[229.1629]"), "1015AA15-");
        Assert.assertEquals(rt("AAK", "TMT 6-plex of peptide N-term@0[229.1629];TMT 6-plex of K@3[229.1629]"),
                "16AA15-");
        Assert.assertEquals(rt("ACK", "TMTpro of peptide N-term@0[304.2071];Carbamidomethyl of C@2[57.0215];"
                + "TMTpro of K@3[304.2071]"), "20AC19-");
    }
}
