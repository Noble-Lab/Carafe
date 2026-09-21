package test.java.ai;

import main.java.ai.AIGear;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;

/**
 * Tests for {@link AIGear#resolveRunMsFile}, the per-row MS-file resolution used by the finetune
 * MS2-index step. Regression coverage for the multi-file (folder) training crash: when {@code -ms}
 * is a folder of MS files (e.g. gas-phase fractionation), each PSM's run must resolve to its own
 * mzML by File.Name, instead of the whole folder being opened as a single mzML (which NPE'd in
 * {@code DIAMeta.load_ms_data}).
 *
 * <p>TestNG style (argument order {@code assertEquals(actual, expected, message)}) to match the
 * project's other tests so these run under {@code mvn test}. File.Name fixtures use forward-slash
 * foreign paths so the base-name resolution is exercised identically on Windows and Linux.</p>
 */
public class AIGearMsFileResolutionTest {

    private Path touch(Path dir, String name) throws Exception {
        Path p = dir.resolve(name);
        Files.createFile(p);
        return p;
    }

    private void deleteRecursively(Path dir) throws Exception {
        if (dir == null || !Files.exists(dir)) {
            return;
        }
        Files.walk(dir).sorted(Comparator.reverseOrder()).map(Path::toFile).forEach(File::delete);
    }

    /** Single-file -ms: the file is used regardless of a foreign, non-existent File.Name value. */
    @Test
    public void singleFileArgIsUsedDirectly() throws Exception {
        Path dir = Files.createTempDirectory("carafe_ms_single");
        try {
            Path f = touch(dir, "run1.mzML");
            String resolved = AIGear.resolveRunMsFile("/foreign/box/run1.mzML", f.toString());
            Assert.assertEquals(resolved, f.toString(),
                    "with a single -ms file, that file should be used regardless of File.Name");
        } finally {
            deleteRecursively(dir);
        }
    }

    /** Folder -ms: each run resolves to its own mzML inside the folder, by File.Name base name. */
    @Test
    public void folderArgResolvesEachRunByBaseName() throws Exception {
        Path dir = Files.createTempDirectory("carafe_ms_folder");
        try {
            touch(dir, "400to500.mzML");
            touch(dir, "500to600.mzML");
            String a = AIGear.resolveRunMsFile("/foreign/box/400to500.mzML", dir.toString());
            String b = AIGear.resolveRunMsFile("/foreign/box/500to600.mzML", dir.toString());
            Assert.assertEquals(new File(a).getName(), "400to500.mzML",
                    "first run should resolve to its file in the folder");
            Assert.assertEquals(new File(b).getName(), "500to600.mzML",
                    "second run should resolve to its file in the folder");
            Assert.assertNotEquals(a, b, "distinct runs must resolve to distinct files (the multi-file fix)");
            Assert.assertTrue(new File(a).isFile() && new File(b).isFile(),
                    "resolved paths must be real files, never the folder itself");
        } finally {
            deleteRecursively(dir);
        }
    }

    /** An existing full path in File.Name is used as-is (Osprey ran on the same machine). */
    @Test
    public void existingFullPathIsUsedAsIs() throws Exception {
        Path dir = Files.createTempDirectory("carafe_ms_fullpath");
        try {
            Path f = touch(dir, "run2.mzML");
            String resolved = AIGear.resolveRunMsFile(f.toString(), dir.toString());
            Assert.assertEquals(resolved, f.toString(),
                    "a File.Name that already points to an existing file should be used directly");
        } finally {
            deleteRecursively(dir);
        }
    }

    /** A run missing from the folder fails loudly rather than handing a directory to the mzML loader. */
    @Test(expectedExceptions = IllegalArgumentException.class)
    public void missingRunInFolderThrows() throws Exception {
        Path dir = Files.createTempDirectory("carafe_ms_missing");
        try {
            touch(dir, "present.mzML");
            AIGear.resolveRunMsFile("/foreign/box/absent.mzML", dir.toString());
            Assert.fail("expected IllegalArgumentException for a run not present in the -ms folder");
        } finally {
            deleteRecursively(dir);
        }
    }

    /** No File.Name and a folder -ms is unresolvable and must throw (not open the folder as mzML). */
    @Test(expectedExceptions = IllegalArgumentException.class)
    public void folderArgWithNoFileNameThrows() throws Exception {
        Path dir = Files.createTempDirectory("carafe_ms_nocol");
        try {
            touch(dir, "present.mzML");
            AIGear.resolveRunMsFile(null, dir.toString());
            Assert.fail("expected IllegalArgumentException when -ms is a folder and no File.Name is available");
        } finally {
            deleteRecursively(dir);
        }
    }
}
