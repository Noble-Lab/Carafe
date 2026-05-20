package test.java.rank;

import org.testng.Assert;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

public class RankLabelGeneratorTest {

    private Class<?> generatorClass;
    private Class<?> rParameterClass;
    private Class<?> psmConfigClass;

    @BeforeMethod
    public void setUp() throws Exception {
        generatorClass = Class.forName("main.java.rank.RankLabelGenerator");
        rParameterClass = Class.forName("main.java.rank.RParameter");
        psmConfigClass = Class.forName("main.java.rank.PSMConfig");

        // Initialize enzymes
        Method initEnzymes = generatorClass.getDeclaredMethod("init_enzymes");
        initEnzymes.setAccessible(true);
        initEnzymes.invoke(null);

        // Reset configuration parameters to standard defaults
        setStaticField(rParameterClass, "enzyme", 1); // Trypsin
        setStaticField(rParameterClass, "maxMissedCleavages", 0);
        setStaticField(rParameterClass, "minPeptideLength", 7);
        setStaticField(rParameterClass, "maxPeptideLength", 35);
        setStaticField(rParameterClass, "minPeptideCharge", 2);
        setStaticField(rParameterClass, "maxPeptideCharge", 4);

        // Reset column names
        setStaticField(psmConfigClass, "ms_file_column_name", "File.Name");
        setStaticField(psmConfigClass, "protein_group_column_name", "Protein.Group");
        setStaticField(psmConfigClass, "stripped_peptide_sequence_column_name", "Stripped.Sequence");
        setStaticField(psmConfigClass, "peptide_modification_column_name", "Modified.Sequence");
        setStaticField(psmConfigClass, "precursor_charge_column_name", "Precursor.Charge");
        setStaticField(psmConfigClass, "precursor_intensity_column_name", "Precursor.Normalised");
        setStaticField(psmConfigClass, "qvalue_column_name", "Q.Value");
        setStaticField(psmConfigClass, "global_qvalue_column_name", "Global.Q.Value");
        setStaticField(psmConfigClass, "lib_qvalue_column_name", "Lib.Q.Value");
        setStaticField(psmConfigClass, "global_pg_column_name", "Global.PG.Q.Value");
        setStaticField(psmConfigClass, "lib_pg_column_name", "Lib.PG.Q.Value");
    }

    private void setStaticField(Class<?> clazz, String fieldName, Object value) throws Exception {
        Field field = clazz.getDeclaredField(fieldName);
        field.setAccessible(true);
        field.set(null, value);
    }

    @Test
    public void testLabelGenerationAndDetectionFlags() throws Exception {
        // 1. Create a temporary mock FASTA file
        Path tempDir = Files.createTempDirectory("carafe_test");
        File fastaFile = new File(tempDir.toFile(), "mock_db.fasta");
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(fastaFile))) {
            writer.write(">PROT_A\n");
            writer.write("AEPTIDEAKAEPTIDEBK\n"); // Digests to AEPTIDEAK and AEPTIDEBK with Trypsin
        }

        // 2. Create a temporary mock report TSV file
        File reportFile = new File(tempDir.toFile(), "mock_report.tsv");
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(reportFile))) {
            // Write Header
            writer.write("File.Name\tProtein.Group\tStripped.Sequence\tModified.Sequence\tPrecursor.Charge\tPrecursor.Normalised\tQ.Value\tGlobal.Q.Value\tLib.Q.Value\tGlobal.PG.Q.Value\tLib.PG.Q.Value\n");
            
            // Run 1: AEPTIDEAK (100.0) > AEPTIDEBK (50.0)
            writer.write("Run1\tPROT_A\tAEPTIDEAK\tAEPTIDEAK\t2\t100.0\t0.001\t0.001\t0.001\t0.001\t0.001\n");
            writer.write("Run1\tPROT_A\tAEPTIDEBK\tAEPTIDEBK\t2\t50.0\t0.001\t0.001\t0.001\t0.001\t0.001\n");

            // Run 2: AEPTIDEAK (0.0 / not detected) < AEPTIDEBK (80.0)
            // Note: We omit AEPTIDEAK in Run 2 to simulate it not being detected in this run.
            writer.write("Run2\tPROT_A\tAEPTIDEBK\tAEPTIDEBK\t2\t80.0\t0.001\t0.001\t0.001\t0.001\t0.001\n");
        }

        // 3. Instantiate RankLabelGenerator and configure
        java.lang.reflect.Constructor<?> constructor = generatorClass.getDeclaredConstructor();
        constructor.setAccessible(true);
        Object generator = constructor.newInstance();
        
        // Set generator instance fields
        setInstanceField(generator, "db", fastaFile.getAbsolutePath());
        setInstanceField(generator, "min_detected_peptides", 2);
        setInstanceField(generator, "ratio_undetected_peptides", 0.0);

        // 4. Run generate_train_data
        Method generateMethod = generatorClass.getDeclaredMethod("generate_train_data", String.class, String.class, String.class);
        generateMethod.setAccessible(true);
        generateMethod.invoke(generator, reportFile.getAbsolutePath(), tempDir.toString(), "test_run");

        // 5. Read the generated consensus label file
        File consensusFile = new File(tempDir.toFile(), "consensus_label.txt");
        Assert.assertTrue(consensusFile.exists(), "consensus_label.txt should be generated");

        List<String> lines = Files.readAllLines(consensusFile.toPath());
        Assert.assertFalse(lines.isEmpty(), "consensus_label.txt should not be empty");

        boolean pairVerified = false;
        for (String line : lines) {
            System.out.println("Consensus output: " + line);
            String[] parts = line.split("\t");
            if (parts.length >= 10 && parts[1].equals("AEPTIDEAK|2:AEPTIDEBK|2")) {
                // Verify votes:
                // Run 1: AEPTIDEAK (100) > AEPTIDEBK (50) -> 1 pos vote
                // Run 2: AEPTIDEAK (not detected = 0) < AEPTIDEBK (80) -> 1 neg vote
                // Total: n_pos = 1, n_neg = 1
                int n_pos = Integer.parseInt(parts[4]);
                int n_neg = Integer.parseInt(parts[5]);
                Assert.assertEquals(n_pos, 1, "n_pos should be 1");
                Assert.assertEquals(n_neg, 1, "n_neg should be 1");

                // Verify detection flags:
                // AEPTIDEAK was detected in Run 1 (so a_detected should be 1)
                // AEPTIDEBK was detected in Run 1 and Run 2 (so b_detected should be 1)
                int a_detected = Integer.parseInt(parts[7]);
                int b_detected = Integer.parseInt(parts[8]);

                Assert.assertEquals(a_detected, 1, "AEPTIDEAK was detected in Run 1, so a_detected should be 1");
                Assert.assertEquals(b_detected, 1, "AEPTIDEBK was detected in both runs, so b_detected should be 1");
                pairVerified = true;
            }
        }
        Assert.assertTrue(pairVerified, "AEPTIDEAK|2:AEPTIDEBK|2 pair should be present and verified");
    }

    private void setInstanceField(Object instance, String fieldName, Object value) throws Exception {
        Field field = instance.getClass().getDeclaredField(fieldName);
        field.setAccessible(true);
        field.set(instance, value);
    }
}
