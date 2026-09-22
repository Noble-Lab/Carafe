package test.java.ai;

import main.java.ai.AIGear;
import main.java.input.CModification;
import main.java.input.CParameter;
import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

/**
 * Tests for the Carafe modification name to AlphaBase/AlphaPeptDeep modification name mapping,
 * focused on isobaric labels (TMT/iTRAQ) and peptide terminal modifications.
 * <p>
 * AlphaBase names peptide terminal modifications "Any N-term"/"Any C-term". UniMod reports the
 * site of those modifications as "N-term"/"C-term", so the mapping has to translate them. Every
 * TMT-labeled peptide carries a peptide N-terminal label, so this translation is required before
 * any TMT training data can be generated.
 */
public class TMTModificationTest {

    private AIGear aiGear;

    @BeforeClass
    public void setUp() {
        CParameter.init();
        aiGear = new AIGear();
        aiGear.load_mod_map();
    }

    private String convert(String modification) {
        return aiGear.convert_modification(modification)[0];
    }

    private String sites(String modification) {
        return aiGear.convert_modification(modification)[1];
    }

    @Test
    public void testTmt6plexOnLysine() {
        Assert.assertEquals(convert("TMT6plex of K@6[229.1629]"), "TMT6plex@K");
    }

    @Test
    public void testTmt6plexOnPeptideNTerm() {
        // UniMod site is "N-term"; AlphaBase requires "Any N-term".
        Assert.assertEquals(convert("TMT6plex of N-term@0[229.1629]"), "TMT6plex@Any N-term");
    }

    @Test
    public void testTmtPeptideNTermFactoryNames() {
        // Names offered by -fixMod/-varMod through top_modifications.tsv.
        Assert.assertEquals(convert("TMT 6-plex of peptide N-term@0[229.1629]"), "TMT6plex@Any N-term");
        Assert.assertEquals(convert("TMT 10-plex of peptide N-term@0[229.1629]"), "TMT6plex@Any N-term");
        Assert.assertEquals(convert("TMT 11-plex of peptide N-term@0[229.1629]"), "TMT6plex@Any N-term");
        Assert.assertEquals(convert("TMT 2-plex of peptide N-term@0[225.1558]"), "TMT2plex@Any N-term");
        Assert.assertEquals(convert("TMTpro of peptide N-term@0[304.2071]"), "TMTpro@Any N-term");
    }

    @Test
    public void testTmtLysineFactoryNames() {
        Assert.assertEquals(convert("TMT 6-plex of K@3[229.1629]"), "TMT6plex@K");
        Assert.assertEquals(convert("TMT 10-plex of K@3[229.1629]"), "TMT6plex@K");
        Assert.assertEquals(convert("TMT 11-plex of K@3[229.1629]"), "TMT6plex@K");
        Assert.assertEquals(convert("TMT 2-plex of K@3[225.1558]"), "TMT2plex@K");
        Assert.assertEquals(convert("TMTpro of K@3[304.2071]"), "TMTpro@K");
    }

    @Test
    public void testItraqNames() {
        Assert.assertEquals(convert("iTRAQ 4-plex of K@3[144.1021]"), "iTRAQ4plex@K");
        Assert.assertEquals(convert("iTRAQ 8-plex of peptide N-term@0[304.1990]"), "iTRAQ8plex@Any N-term");
    }

    /**
     * A fully TMT 6-plex labeled peptide with an oxidized methionine: the shape of essentially
     * every row produced from the Prosit-TMT ProteomeTools table.
     */
    @Test
    public void testFullyLabeledPeptide() {
        String modification = "TMT6plex of N-term@0[229.1629];"
                + "Oxidation of M@4[15.9949];"
                + "TMT6plex of K@8[229.1629]";
        Assert.assertEquals(convert(modification), "TMT6plex@Any N-term;Oxidation@M;TMT6plex@K");
        Assert.assertEquals(sites(modification), "0;4;8");
    }

    /**
     * The terminal-site fix also registers peptide N-terminal labels in the psi_name_site maps
     * that the DIA-NN/Skyline/EncyclopeDIA writers consult. Before the fix these were filed under
     * the unusable key "TMT6plex@N-term".
     * <p>
     * The lysine label needs separate handling: "TMT6plex of K" is deduplicated out of
     * id2ptmname (its UNIMOD:737_K key collides with the factory name "TMT 6-plex of K"), so
     * load_UniMods cannot register "TMT6plex@K" and CModification.load_isobaric_label_sites()
     * registers it from the factory name instead.
     */
    @Test
    public void testPeptideNTermLabelRegisteredForLibraryOutput() {
        CModification modifications = CModification.getInstance();
        Assert.assertTrue(
                modifications.psi_name_site2site_unimod_acc.containsKey("TMT6plex@Any N-term"),
                "TMT6plex@Any N-term should be registered for library output");
        Assert.assertFalse(
                modifications.psi_name_site2site_unimod_acc.containsKey("TMT6plex@N-term"),
                "the unusable @N-term key should no longer be produced");
    }

    @Test
    public void testToAlphabaseSite() {
        Assert.assertEquals(CModification.to_alphabase_site("N-term"), "Any N-term");
        Assert.assertEquals(CModification.to_alphabase_site("C-term"), "Any C-term");
        Assert.assertEquals(CModification.to_alphabase_site("K"), "K");
        Assert.assertEquals(CModification.to_alphabase_site("Protein N-term"), "Protein N-term");
    }

    // ------------------------------------------------------------------ library output

    private String writer(String method, String peptide, String mods, String sites) throws Exception {
        java.lang.reflect.Method m = AIGear.class.getDeclaredMethod(
                method, String.class, String.class, String.class);
        m.setAccessible(true);
        return (String) m.invoke(aiGear, peptide, mods, sites);
    }

    /**
     * A fully TMT-labeled tryptic peptide must survive every library writer. Before the isobaric
     * label sites were registered these exited with
     * "modification TMT6plex@K is not supported yet", and the peptide N-terminal label separately
     * indexed aa[-1].
     */
    @Test
    public void testTmtPeptideRendersInDiannLibrary() throws Exception {
        Assert.assertEquals(
                writer("get_modified_peptide_diann", "AACLDK",
                        "TMT6plex@Any N-term;Carbamidomethyl@C;TMT6plex@K", "0;3;6"),
                "_(UniMod:737)AAC[UniMod:4]LDK(UniMod:737)_");
        // Note the bracket style differs: Carafe hard-codes "C[UniMod:4]" for the common
        // modifications while the generic path emits "(UniMod:737)". Both are pre-existing.
    }

    @Test
    public void testTmtPeptideRendersInEncyclopediaLibrary() throws Exception {
        Assert.assertEquals(
                writer("get_modified_peptide_encyclopedia", "AACLDK",
                        "TMT6plex@Any N-term;Carbamidomethyl@C;TMT6plex@K", "0;3;6"),
                "_[TMT6plex]AAC[Carbamidomethyl (C)]LDK[TMT6plex (K)]_");
    }

    @Test
    public void testTmtPeptideRendersInMzSpecLib() throws Exception {
        Assert.assertEquals(
                writer("get_modified_peptide", "AACLDK",
                        "TMT6plex@Any N-term;TMT6plex@K", "0;6"),
                "_[TMT6plex]AACLDK[TMT6plex]_");
    }

    /**
     * A peptide N-terminal label and a label on residue 1 both belong at the start of the
     * sequence. Both used to be written to aa[0], so whichever was processed second silently
     * discarded the other while the precursor m/z still counted both, producing a library row
     * whose sequence and mass disagree. The result must not depend on the listing order, which
     * varies with the order modifications were added (fixMod order, or variable before fixed).
     */
    @Test
    public void testTerminalAndFirstResidueLabelsBothSurvive() throws Exception {
        String[][] writers = {
                {"get_modified_peptide_diann", "_(UniMod:737)K(UniMod:737)AACLD_"},
                {"get_modified_peptide_encyclopedia", "_[TMT6plex]K[TMT6plex (K)]AACLD_"},
                {"get_modified_peptide", "_[TMT6plex]K[TMT6plex]AACLD_"},
        };
        for (String[] w : writers) {
            Assert.assertEquals(
                    writer(w[0], "KAACLD", "TMT6plex@Any N-term;TMT6plex@K", "0;1"), w[1],
                    w[0] + ": N-terminal label listed first");
            Assert.assertEquals(
                    writer(w[0], "KAACLD", "TMT6plex@K;TMT6plex@Any N-term", "1;0"), w[1],
                    w[0] + ": N-terminal label listed second - must give the same result");
        }
    }

    @Test
    public void testSkylinePlacesTerminalLabelOnFirstResidue() throws Exception {
        java.lang.reflect.Method m = AIGear.class.getDeclaredMethod(
                "get_skyline_modification_position", String.class, int.class);
        m.setAccessible(true);
        Assert.assertEquals((int) m.invoke(aiGear, "TMT6plex@Any N-term", 0), 1);
        // unchanged for the case that already worked
        Assert.assertEquals((int) m.invoke(aiGear, "Acetyl@Protein_N-term", 0), 1);
        Assert.assertEquals((int) m.invoke(aiGear, "TMT6plex@K", 6), 6);
    }

    @Test
    public void testTmtMassIsResolvable() {
        CModification modifications = CModification.getInstance();
        Assert.assertEquals(modifications.get_mod_mass_by_psi_name_site("TMT6plex@K"),
                229.16293213472, 1e-9);
        Assert.assertEquals(modifications.get_mod_mass_by_psi_name_site("TMT6plex@Any N-term"),
                229.16293213472, 1e-6);
    }

    /**
     * Guard against regressions: the mappings Carafe2 already relied on must be untouched.
     */
    @Test
    public void testExistingMappingsUnchanged() {
        Assert.assertEquals(convert("Oxidation of M@1[15.9949]"), "Oxidation@M");
        Assert.assertEquals(convert("Carbamidomethyl of C@5[57.0215]"), "Carbamidomethyl@C");
        Assert.assertEquals(convert("Carbamidomethylation of C@5[57.0215]"), "Carbamidomethyl@C");
        Assert.assertEquals(convert("Phospho of S@2[79.9663]"), "Phospho@S");
        Assert.assertEquals(convert("Phosphorylation of S@2[79.9663]"), "Phospho@S");
        Assert.assertEquals(convert("Acetyl of protein N-term@0[42.0106]"), "Acetyl@Protein_N-term");
        Assert.assertEquals(convert("GG of K@4[114.0429]"), "GG@K");
        Assert.assertEquals(convert("-"), "");
    }
}
