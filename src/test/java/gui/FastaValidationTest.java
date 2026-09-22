package test.java.gui;

import main.java.gui.CarafeGUI;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.File;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;

/**
 * Validation of the protein database fields.
 * <p>
 * Checking only that the file exists let a saved settings file be accepted as the database: it
 * exists, so validation passed, the run produced zero peptides and then failed much later with an
 * unrelated error. Validation therefore checks the extension against what the backend actually
 * dispatches on, and additionally checks the content of anything claiming to be FASTA.
 * <p>
 * The extensions matter: the training database field also accepts a DIA-NN {@code .speclib}
 * (offered by its own Browse filter and passed to DIA-NN as its library), and the library
 * database field accepts a peptide-forms or Skyline precursor table. Rejecting either would
 * break a supported workflow outright.
 */
public class FastaValidationTest {

    /** @return the reason the file is unusable, or null when acceptable */
    private static String problem(File f, boolean allowSpeclib) throws Exception {
        Method m = CarafeGUI.class.getDeclaredMethod("proteinDbProblem", File.class, boolean.class);
        m.setAccessible(true);
        return (String) m.invoke(null, f, allowSpeclib);
    }

    private static File write(String name, String suffix, String content) throws Exception {
        File f = Files.createTempFile("carafe_" + name, suffix).toFile();
        f.deleteOnExit();
        Files.write(f.toPath(), content.getBytes(StandardCharsets.UTF_8));
        return f;
    }

    @Test
    public void testAcceptsFasta() throws Exception {
        Assert.assertNull(problem(write("plain", ".fasta", ">sp|P12345|TEST\nMKAIL\n"), false));
        Assert.assertNull(problem(write("fa", ".fa", ">sp|P12345|TEST\nMKAIL\n"), false));
        // Leading blank lines are tolerated.
        Assert.assertNull(problem(write("blank", ".fasta", "\n\n>sp|P1|T\nMKAIL\n"), false));
    }

    @Test
    public void testAcceptsFastaWithByteOrderMark() throws Exception {
        // A BOM'd FASTA reads its first line as "\uFEFF>sp|...", which a naive startsWith(">")
        // rejects even though the file is perfectly valid.
        Assert.assertNull(problem(write("bom", ".fasta", "\uFEFF>sp|P12345|TEST\nMKAIL\n"), false));
    }

    @Test
    public void testAcceptsTheAlternativesEachFieldSupports() throws Exception {
        // Training database: DIA-NN spectral library.
        Assert.assertNull(problem(write("lib", ".speclib", "binary-ish content"), true));
        // Library database: peptide forms / Skyline precursor table.
        for (String ext : new String[]{".tsv", ".txt", ".csv"}) {
            Assert.assertNull(problem(write("table", ext, "peptide\tcharge\nPEPTIDER\t2\n"), false),
                    ext + " is a supported library database input");
        }
    }

    @Test
    public void testRejectsTheWrongKindOfFile() throws Exception {
        // The case that reached Carafe: a saved settings file in the database field.
        Assert.assertNotNull(problem(write("json", ".json", "{\n  \"workflow\": 3\n}\n"), false));
        // A .speclib is not a library database, and a peptide table is not a training database.
        Assert.assertNotNull(problem(write("lib2", ".speclib", "x"), false));
        Assert.assertNotNull(problem(write("table2", ".tsv", "a\tb\n"), true));
    }

    @Test
    public void testRejectsAMisnamedFasta() throws Exception {
        // Claims to be FASTA but is not, which the extension alone cannot catch.
        Assert.assertNotNull(problem(write("fake", ".fasta", "{\"workflow\": 3}\n"), false));
        Assert.assertNotNull(problem(write("emptyfasta", ".fasta", ""), false));
    }

    @Test
    public void testUnreadableFileIsLeftToCarafe() throws Exception {
        // Not the GUI's job to report a path problem; Carafe gives the real error.
        File missing = new File("Z:/nope/x_" + System.nanoTime() + ".fasta");
        Assert.assertNull(problem(missing, false));
    }
}
