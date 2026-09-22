package test.java.gui;

import main.java.gui.CarafeGUI;
import main.java.input.CModification;
import main.java.input.CParameter;
import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.lang.reflect.Method;

/**
 * The GUI's isobaric labelling preset maps a labelling choice to Carafe modification names.
 * <p>
 * The names are resolved to numeric {@code -fixMod} ids at run time rather than hard-coded,
 * because the ids depend on the contents of {@code top_modifications.tsv} and {@code unimod.xml}
 * (see {@code ModificationIdStabilityTest}). This checks the names the GUI asks for actually
 * exist, so the preset cannot silently produce an empty or wrong selection.
 */
public class LabelingPresetTest {

    private Method namesFor;

    @BeforeClass
    public void setUp() throws Exception {
        CParameter.init();
        CModification.getInstance();
        // Reached by reflection so the GUI does not have to widen its API for a test.
        namesFor = CarafeGUI.class.getDeclaredMethod("labelingModificationNames", String.class);
        namesFor.setAccessible(true);
    }

    private String[] names(String labeling) throws Exception {
        return (String[]) namesFor.invoke(null, labeling);
    }

    @Test
    public void testUnlabeledChoiceSelectsNothing() throws Exception {
        Assert.assertEquals(names("None").length, 0);
        Assert.assertEquals(names(null).length, 0);
    }

    @Test
    public void testEveryTmtChoiceLabelsBothLysineAndTheNTerminus() throws Exception {
        for (String choice : new String[]{
                "TMT 6-plex / 10-plex / 11-plex", "TMT 2-plex", "TMTpro (16/18/32/35-plex)"}) {
            String[] names = names(choice);
            Assert.assertEquals(names.length, 2, choice + " should set two modifications");
            Assert.assertTrue(names[0].endsWith(" of K"), choice + ": first should be the K label");
            Assert.assertTrue(names[1].endsWith(" of peptide N-term"),
                    choice + ": second should be the peptide N-terminal label. TMT labels both, "
                            + "and every TMT peptide carries the N-terminal one.");
        }
    }

    /**
     * The decisive check: every name the preset uses must resolve to a modification id, otherwise
     * the GUI would quietly fail to set the fixed modifications.
     */
    @Test
    public void testEveryPresetNameResolvesToAModificationId() throws Exception {
        for (String choice : new String[]{
                "TMT 6-plex / 10-plex / 11-plex", "TMT 2-plex", "TMTpro (16/18/32/35-plex)"}) {
            for (String name : names(choice)) {
                Integer id = CModification.getInstance().ptm_name2id.get(name);
                Assert.assertNotNull(id, "No modification id for " + name
                        + " (used by the '" + choice + "' preset)");
            }
        }
    }

    /**
     * The presets must be distinct: TMT 2-plex and TMTpro have different masses from the
     * 6/10/11-plex family, so selecting one must not select another's modifications.
     */
    @Test
    public void testPresetsDoNotOverlap() throws Exception {
        String[] choices = {"TMT 6-plex / 10-plex / 11-plex", "TMT 2-plex", "TMTpro (16/18/32/35-plex)"};
        for (int i = 0; i < choices.length; i++) {
            for (int j = i + 1; j < choices.length; j++) {
                for (String a : names(choices[i])) {
                    for (String b : names(choices[j])) {
                        Assert.assertNotEquals(a, b,
                                choices[i] + " and " + choices[j] + " must not share " + a);
                    }
                }
            }
        }
    }
}
