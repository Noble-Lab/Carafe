package test.java.ai;

import main.java.ai.PSMConfig;
import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.Test;

/**
 * Tests for {@link PSMConfig#use_osprey_blib_column_names()}. The Osprey path reuses the
 * DIA-NN column names (since {@code OspreyBlibReader} emits a DIA-NN-style TSV) but tags the
 * search engine as "Osprey".
 *
 * <p>TestNG style (argument order {@code assertEquals(actual, expected, message)}) to match the
 * project's other tests so these run under {@code mvn test}.</p>
 */
public class PSMConfigTest {

    // PSMConfig is static, mutable global state; reset to the DIA-NN defaults after each test so
    // column-name changes here can't leak into other tests.
    @AfterMethod
    public void resetColumnConfig() {
        PSMConfig.use_diann_report_column_names();
    }

    @Test
    public void ospreyColumnsMatchDiannNamesButTagOspreyEngine() {
        PSMConfig.use_osprey_blib_column_names();
        Assert.assertEquals(PSMConfig.search_engine_name, "Osprey");
        Assert.assertEquals(PSMConfig.stripped_peptide_sequence_column_name, "Stripped.Sequence");
        Assert.assertEquals(PSMConfig.peptide_modification_column_name, "Modified.Sequence");
        Assert.assertEquals(PSMConfig.precursor_charge_column_name, "Precursor.Charge");
        Assert.assertEquals(PSMConfig.rt_column_name, "RT");
        Assert.assertEquals(PSMConfig.qvalue_column_name, "Q.Value");
        Assert.assertEquals(PSMConfig.ms_file_column_name, "File.Name");
    }

    @Test
    public void diannAndOspreyShareIdentificationColumnNames() {
        PSMConfig.use_diann_report_column_names();
        String diannStripped = PSMConfig.stripped_peptide_sequence_column_name;
        String diannMod = PSMConfig.peptide_modification_column_name;
        String diannRt = PSMConfig.rt_column_name;

        PSMConfig.use_osprey_blib_column_names();
        // The OspreyBlibReader writes DIA-NN-named columns so the existing reader works unchanged.
        Assert.assertEquals(PSMConfig.stripped_peptide_sequence_column_name, diannStripped);
        Assert.assertEquals(PSMConfig.peptide_modification_column_name, diannMod);
        Assert.assertEquals(PSMConfig.rt_column_name, diannRt);
    }

    /**
     * Regression for the Osprey finetune "Spectrum not found" bug. The Osprey {@code .blib} has no
     * usable DIA-NN MS2 scan index, so {@code MS2.Scan} is written as 0 for every precursor and
     * {@code add_ms2spectrum_index} synthesizes the true ordinal into an {@code ms2index} column.
     * The finetune path resets to DIA-NN column names afterward, which points the reader back at
     * the all-zero {@code MS2.Scan} placeholder -- so it must re-point at {@code ms2index} via
     * {@link PSMConfig#use_added_ms2index_columns()}. If that re-point is dropped, every precursor
     * resolves to MS2 scan 0 and MS2 training data collapses to zero valid PSMs.
     */
    @Test
    public void ospreyFinetuneReadsSynthesizedMs2IndexNotZeroPlaceholder() {
        // Reproduce the exact sequence in AIGear's Osprey finetune branch.
        PSMConfig.use_osprey_blib_column_names();
        Assert.assertEquals(PSMConfig.ms2_index_column_name, "MS2.Scan",
                "blib placeholder column before correction");

        PSMConfig.use_diann_report_column_names();
        Assert.assertEquals(PSMConfig.ms2_index_column_name, "MS2.Scan",
                "DIA-NN reset still points at the placeholder");

        PSMConfig.use_added_ms2index_columns();
        // The fix: the reader must use the column add_ms2spectrum_index actually populated.
        Assert.assertEquals(PSMConfig.ms2_index_column_name, "ms2index",
                "reader must use the synthesized MS2 ordinal, not MS2.Scan=0");
        Assert.assertEquals(PSMConfig.rt_column_name, "apex_rt",
                "reader must use the matched apex RT column");

        // Restore DIA-NN defaults so static config does not leak into other tests.
        PSMConfig.use_diann_report_column_names();
    }
}
