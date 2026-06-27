package test.java.ai;

import main.java.ai.PSMConfig;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * Tests for {@link PSMConfig#use_osprey_blib_column_names()}. The OspreySharp path reuses the
 * DIA-NN column names (since {@code OspreyBlibReader} emits a DIA-NN-style TSV) but tags the
 * search engine as "OspreySharp".
 *
 * <p>TestNG style (argument order {@code assertEquals(actual, expected, message)}) to match the
 * project's other tests so these run under {@code mvn test}.</p>
 */
public class PSMConfigTest {

    @Test
    public void ospreyColumnsMatchDiannNamesButTagOspreyEngine() {
        PSMConfig.use_osprey_blib_column_names();
        Assert.assertEquals(PSMConfig.search_engine_name, "OspreySharp");
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
}
