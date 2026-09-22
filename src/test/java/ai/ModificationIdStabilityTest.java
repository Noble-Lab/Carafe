package test.java.ai;

import main.java.input.CModification;
import main.java.input.CParameter;
import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Pins the numeric modification ids used by {@code -fixMod} / {@code -varMod} and by the GUI
 * modification dropdowns.
 * <p>
 * These numbers appear verbatim in users' saved command lines, scripts and published methods, so
 * renumbering them silently invalidates earlier analyses: a command that meant
 * "Carbamidomethyl of C" would quietly come to mean something else. The ids are assigned
 * sequentially in the {@link CModification} constructor from {@code top_modifications.tsv},
 * then the modification factory defaults, then the UniMod-derived modifications, so editing
 * {@code top_modifications.tsv} or {@code unimod.xml} shifts every id after the edit.
 * <p>
 * If this test fails, the change is not safe to ship as-is even if everything else passes.
 */
public class ModificationIdStabilityTest {

    /** The ids shown in the GUI dropdowns, which calls get_top_mod_list(26). */
    private static final Map<Integer, String> EXPECTED = new LinkedHashMap<>();

    static {
        EXPECTED.put(1, "Carbamidomethyl of C");
        EXPECTED.put(2, "Oxidation of M");
        EXPECTED.put(3, "Deamidated of N");
        EXPECTED.put(4, "Deamidated of Q");
        EXPECTED.put(5, "Acetyl of protein N-term");
        EXPECTED.put(6, "Acetyl of K");
        EXPECTED.put(7, "Phospho of S");
        EXPECTED.put(8, "Phospho of T");
        EXPECTED.put(9, "Phospho of Y");
        EXPECTED.put(10, "GG of K");
        EXPECTED.put(11, "TMT 10-plex of K");
        EXPECTED.put(12, "TMT 10-plex of peptide N-term");
        EXPECTED.put(13, "TMT 11-plex of K");
        EXPECTED.put(14, "TMT 11-plex of peptide N-term");
        EXPECTED.put(15, "TMT 6-plex of K");
        EXPECTED.put(16, "TMT 6-plex of peptide N-term");
        EXPECTED.put(17, "TMT 2-plex of K");
        EXPECTED.put(18, "TMT 2-plex of peptide N-term");
        EXPECTED.put(19, "TMTpro of K");
        EXPECTED.put(20, "TMTpro of peptide N-term");
        EXPECTED.put(21, "iTRAQ 4-plex of K");
        EXPECTED.put(22, "iTRAQ 4-plex of peptide N-term");
        EXPECTED.put(23, "iTRAQ 4-plex of Y");
        EXPECTED.put(24, "iTRAQ 8-plex of K");
        EXPECTED.put(25, "iTRAQ 8-plex of peptide N-term");
        EXPECTED.put(26, "iTRAQ 8-plex of Y");
    }

    private CModification modifications;

    @BeforeClass
    public void setUp() {
        CParameter.init();
        modifications = CModification.getInstance();
    }

    @Test
    public void testGuiVisibleModificationIdsAreUnchanged() {
        for (Map.Entry<Integer, String> entry : EXPECTED.entrySet()) {
            Assert.assertEquals(modifications.id2ptmname.get(entry.getKey()), entry.getValue(),
                    "Modification id " + entry.getKey() + " changed meaning. Saved -fixMod/-varMod "
                            + "command lines would silently select a different modification.");
        }
    }

    @Test
    public void testTopModListMatchesTheSameIds() {
        // The GUI builds its dropdowns from this call, so it must agree with the ids above.
        LinkedHashMap<Integer, String> topMods = modifications.get_top_mod_list(EXPECTED.size());
        Assert.assertEquals(topMods.size(), EXPECTED.size());
        for (Map.Entry<Integer, String> entry : EXPECTED.entrySet()) {
            String shown = topMods.get(entry.getKey());
            Assert.assertNotNull(shown, "id " + entry.getKey() + " missing from the GUI dropdown list");
            Assert.assertTrue(shown.startsWith(entry.getValue()),
                    "GUI dropdown entry for id " + entry.getKey() + " is " + shown
                            + ", expected it to start with " + entry.getValue());
        }
    }

    /**
     * The total count is a cheap tripwire for a change to unimod.xml or the modification factory:
     * any addition or removal shifts the ids of everything after it.
     */
    @Test
    public void testTotalModificationCountIsUnchanged() {
        Assert.assertEquals(modifications.id2ptmname.size(), 1069,
                "The number of registered modifications changed, which renumbers ids. "
                        + "If this is intentional, every saved -fixMod/-varMod value must be migrated.");
    }
}
