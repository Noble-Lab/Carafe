package test.java.db;

import com.compomics.util.experiment.biology.enzymes.Enzyme;
import main.java.db.DBGear;
import main.java.input.CParameter;
import main.java.input.PeptideUtils;
import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

import java.util.Set;

/**
 * Tests for {@link DBGear#get_mz}, the precursor m/z calculation, and for N-terminal methionine
 * clipping in {@link DBGear#digest_protein}. get_mz is foundational -- every library precursor,
 * isolation-window assignment and apex lookup depends on it -- and a regression here would be
 * silent, so the mass/charge relation is pinned. The clip tests guard the fix for issue #1: the
 * clip must fire only for a true protein initiator methionine during a real digest, and never in
 * NoCut mode (where each record is an already-digested peptide).
 */
public class DBGearTest {

    // The clip tests mutate global CParameter digest settings; save/restore so they don't leak.
    private int savedEnzyme, savedMissed, savedMinLen, savedMaxLen;
    private boolean savedClipM;

    @BeforeMethod
    public void saveCParameter() {
        savedEnzyme = CParameter.enzyme;
        savedMissed = CParameter.maxMissedCleavages;
        savedMinLen = CParameter.minPeptideLength;
        savedMaxLen = CParameter.maxPeptideLength;
        savedClipM = CParameter.clip_nTerm_M;
    }

    @AfterMethod
    public void restoreCParameter() {
        CParameter.enzyme = savedEnzyme;
        CParameter.maxMissedCleavages = savedMissed;
        CParameter.minPeptideLength = savedMinLen;
        CParameter.maxPeptideLength = savedMaxLen;
        CParameter.clip_nTerm_M = savedClipM;
    }

    private void permissiveDigestParams() {
        CParameter.clip_nTerm_M = true;
        CParameter.maxMissedCleavages = 0;
        CParameter.minPeptideLength = 5;
        CParameter.maxPeptideLength = 35;
    }

    @Test
    public void noCutEnzymeNeverClipsNTerminalMet() {
        DBGear.init_enzymes();
        permissiveDigestParams();
        DBGear g = new DBGear();
        Enzyme noCut = DBGear.getEnzymeByIndex(DBGear.getEnzymeIndexByName("NoCut"));
        // In NoCut mode the record IS the peptide; it must be taken as-is with the leading M kept.
        Set<String> peptides = g.digest_protein(noCut, "MPEPTIDEK");
        Assert.assertTrue(peptides.contains("MPEPTIDEK"), "NoCut must keep the peptide as-is");
        Assert.assertFalse(peptides.contains("PEPTIDEK"),
                "NoCut must not clip the N-terminal methionine");
        // Stronger than "the clipped form is absent": nothing extra was minted at all.
        Assert.assertEquals(peptides.size(), 1,
                "NoCut on a single peptide must yield exactly that peptide");
    }

    /**
     * The sibling of the clip guard, one line below it. PeptideUtils.protein_n_term_peptides gates
     * modn_protein variable mods, and its "peptide is a prefix of the protein" filter is vacuous
     * under NoCut - the entry IS the protein - so every peptide in a peptide-level FASTA would be
     * recorded as a protein N-terminus and become eligible for e.g. Acetyl (protein N-term).
     */
    @Test
    public void noCutRegistersNoProteinNTermPeptides() {
        DBGear.init_enzymes();
        permissiveDigestParams();
        PeptideUtils.protein_n_term_peptides.clear();
        DBGear g = new DBGear();
        Enzyme noCut = DBGear.getEnzymeByIndex(DBGear.getEnzymeIndexByName("NoCut"));

        g.digest_protein(noCut, "MPEPTIDEK");
        g.digest_protein(noCut, "SAMPLERPEPTIDEK");

        Assert.assertFalse(PeptideUtils.protein_n_term_peptides.contains("MPEPTIDEK"),
                "NoCut has no protein context and must claim no protein N-terminus");
        Assert.assertFalse(PeptideUtils.protein_n_term_peptides.contains("SAMPLERPEPTIDEK"),
                "an interior tryptic peptide must never be registered as a protein N-terminus");
    }

    /**
     * The other half: a real digest must still register its genuine protein N-terminal peptides,
     * or the guard above could be satisfied by never populating the set at all - which would
     * silently drop protein N-terminal modifications from every ordinary run.
     */
    @Test
    public void realEnzymeStillRegistersProteinNTermPeptides() {
        DBGear.init_enzymes();
        permissiveDigestParams();
        PeptideUtils.protein_n_term_peptides.clear();
        DBGear g = new DBGear();
        Enzyme trypsin = DBGear.getEnzymeByIndex(DBGear.getEnzymeIndexByName("Trypsin"));

        g.digest_protein(trypsin, "MAAAAAKMPEPTIDEK");

        Assert.assertTrue(PeptideUtils.protein_n_term_peptides.contains("MAAAAAK"),
                "the protein's leading peptide is a genuine protein N-terminus");
        Assert.assertFalse(PeptideUtils.protein_n_term_peptides.contains("MPEPTIDEK"),
                "an internal M-starting peptide is not a protein N-terminus");
    }

    @Test
    public void trypsinClipsOnlyTheProteinInitiatorMet() {
        DBGear.init_enzymes();
        permissiveDigestParams();
        DBGear g = new DBGear();
        Enzyme trypsin = DBGear.getEnzymeByIndex(DBGear.getEnzymeIndexByName("Trypsin"));
        // Protein starts with M (real initiator) and also contains an INTERNAL M-starting peptide.
        Set<String> peptides = g.digest_protein(trypsin, "MAAAAAKMPEPTIDEK");
        Assert.assertTrue(peptides.contains("MAAAAAK"), "leading peptide retained");
        Assert.assertTrue(peptides.contains("AAAAAK"), "initiator Met of the leading peptide clipped");
        Assert.assertTrue(peptides.contains("MPEPTIDEK"), "internal M-starting peptide retained");
        Assert.assertFalse(peptides.contains("PEPTIDEK"),
                "internal M (not a protein initiator) must not be clipped");
    }

    @Test
    public void mzFollowsTheMassChargeRelation() {
        DBGear g = new DBGear();
        double proton = g.get_mz(0.0, 1); // (0 + 1*proton)/1
        Assert.assertTrue(proton > 1.0 && proton < 1.02, "proton mass should be ~1.007, was " + proton);
        // m/z = mass/charge + proton
        Assert.assertEquals(g.get_mz(1000.0, 2), 1000.0 / 2 + proton, 1e-9);
        Assert.assertEquals(g.get_mz(2400.0, 3), 2400.0 / 3 + proton, 1e-9);
        Assert.assertEquals(g.get_mz(927.44, 1), 927.44 + proton, 1e-9);
    }

    @Test
    public void higherChargeGivesLowerMz() {
        DBGear g = new DBGear();
        double m = 2000.0;
        Assert.assertTrue(g.get_mz(m, 3) < g.get_mz(m, 2));
        Assert.assertTrue(g.get_mz(m, 2) < g.get_mz(m, 1));
    }

    @Test
    public void neutralMassIsRecoverableFromMz() {
        DBGear g = new DBGear();
        double proton = g.get_mz(0.0, 1);
        int z = 2;
        double mz = g.get_mz(1234.56, z);
        // mass = mz*z - z*proton
        Assert.assertEquals(mz * z - z * proton, 1234.56, 1e-9);
    }
}
