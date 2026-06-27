package test.java.util;

import main.java.util.ReuseSignature;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

/**
 * Tests for {@link ReuseSignature}: the per-step reuse signature must be deterministic, change when
 * parameters or input files change, ignore machine/run-volatile command tokens, and round-trip
 * through the sidecar.
 */
public class ReuseSignatureTest {

    @Test
    public void computeIsDeterministic() {
        List<String> cmd = Arrays.asList("java", "-jar", "carafe.jar", "-fdr", "0.01", "-db", "x.fasta");
        String a = ReuseSignature.compute(cmd, null, List.of());
        String b = ReuseSignature.compute(cmd, null, List.of());
        Assert.assertEquals(a, b);
        Assert.assertEquals(a.length(), 64, "SHA-256 hex");
    }

    @Test
    public void changingAParameterChangesTheSignature() {
        List<String> base = Arrays.asList("java", "-jar", "carafe.jar", "-fdr", "0.01", "-db", "x.fasta");
        List<String> diff = Arrays.asList("java", "-jar", "carafe.jar", "-fdr", "0.05", "-db", "x.fasta");
        Assert.assertNotEquals(ReuseSignature.compute(base, null, List.of()),
                ReuseSignature.compute(diff, null, List.of()));
    }

    @Test
    public void normalizationIgnoresVolatileTokens() {
        // Differ only in executable path, -Xmx, -jar path, -python path, and --threads.
        List<String> a = Arrays.asList("/usr/bin/java", "-Xmx16G", "-jar", "/a/carafe.jar",
                "-python", "/p/python", "-fdr", "0.01", "-db", "x.fasta");
        List<String> b = Arrays.asList("C:\\jdk\\java.exe", "-Xmx8G", "-jar", "D:\\carafe.jar",
                "-python", "C:\\py\\python.exe", "--threads", "8", "-fdr", "0.01", "-db", "x.fasta");
        Assert.assertEquals(ReuseSignature.compute(a, null, List.of()),
                ReuseSignature.compute(b, null, List.of()),
                "RAM / thread / executable-path differences must not invalidate the cache");
    }

    @Test
    public void editingAnInputFileChangesTheSignature() throws Exception {
        Path in = Files.createTempFile("reuse_in", ".fasta");
        Files.writeString(in, ">a\nPEPTIDEK\n", StandardCharsets.UTF_8);
        List<String> cmd = Arrays.asList("java", "-jar", "carafe.jar", "-db", in.toString());
        String before = ReuseSignature.compute(cmd, null, List.of(in.toString()));
        // Append bytes -> size (and mtime) change.
        Files.writeString(in, ">b\nSAMPLERK\n", StandardCharsets.UTF_8, java.nio.file.StandardOpenOption.APPEND);
        String after = ReuseSignature.compute(cmd, null, List.of(in.toString()));
        Assert.assertNotEquals(before, after, "an edited input must invalidate the signature");
    }

    @Test
    public void unchangedInputKeepsTheSignatureStable() throws Exception {
        Path in = Files.createTempFile("reuse_stable", ".fasta");
        Files.writeString(in, ">a\nPEPTIDEK\n", StandardCharsets.UTF_8);
        List<String> cmd = Arrays.asList("java", "-jar", "carafe.jar", "-db", in.toString());
        String a = ReuseSignature.compute(cmd, null, List.of(in.toString()));
        String b = ReuseSignature.compute(cmd, null, List.of(in.toString()));
        Assert.assertEquals(a, b);
    }

    @Test
    public void directoryInputDetectsChangedContents() throws Exception {
        Path dir = Files.createTempDirectory("reuse_dir");
        Files.writeString(dir.resolve("a.mzML"), "x", StandardCharsets.UTF_8);
        List<String> cmd = Arrays.asList("java", "-jar", "carafe.jar", "-ms", dir.toString());
        String before = ReuseSignature.compute(cmd, null, List.of(dir.toString()));
        // Add another file in the directory.
        Files.writeString(dir.resolve("b.mzML"), "y", StandardCharsets.UTF_8);
        String after = ReuseSignature.compute(cmd, null, List.of(dir.toString()));
        Assert.assertNotEquals(before, after, "a changed directory listing must invalidate the signature");
    }

    @Test
    public void tokenizesRawCmdWhenArgsEmpty() {
        // MSConvert tasks carry only a cmd string (with quoted paths).
        String raw = "msconvert --filter \"peakPicking true 1-2\" --mzML \"in.raw\" -o \"out\"";
        String a = ReuseSignature.compute(List.of(), raw, List.of());
        String b = ReuseSignature.compute(List.of(), raw, List.of());
        Assert.assertEquals(a, b);
        // A different filter changes the signature.
        String raw2 = "msconvert --filter \"peakPicking false 1-2\" --mzML \"in.raw\" -o \"out\"";
        Assert.assertNotEquals(a, ReuseSignature.compute(List.of(), raw2, List.of()));
    }

    @Test
    public void sidecarRoundTrip() throws Exception {
        Path out = Files.createTempFile("reuse_out", ".tsv");
        String sig = "deadbeef";
        ReuseSignature.writeSidecar(out.toString(), sig, "-fdr 0.01 -db x.fasta", List.of());
        Assert.assertTrue(new File(ReuseSignature.sidecarPath(out.toString())).isFile());
        Assert.assertTrue(ReuseSignature.matches(out.toString(), sig));
        Assert.assertFalse(ReuseSignature.matches(out.toString(), "other"));
    }

    @Test
    public void matchesIsFalseWhenSidecarAbsent() throws Exception {
        Path out = Files.createTempFile("reuse_nosig", ".tsv");
        // No sidecar written.
        Assert.assertFalse(ReuseSignature.matches(out.toString(), "anything"));
    }
}
