package main.java.util;

import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;
import org.apache.tools.ant.types.Commandline;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.List;

/**
 * Computes and persists a per-step "reuse signature" so the GUI's <em>Reuse existing results</em>
 * feature only skips a step when its output already exists <b>and</b> the step would produce the
 * same result — i.e. the analysis parameters and the input files are unchanged.
 *
 * <p>A signature is a SHA-256 of:</p>
 * <ol>
 *   <li>the step's command with machine/run-volatile tokens removed (executable, {@code -jar},
 *       {@code -python}, {@code -Xmx}/{@code -Xms}, {@code -Djava.security.manager*},
 *       {@code --threads}, {@code --verbose}), so it reflects analysis parameters rather than the
 *       machine; and</li>
 *   <li>a fingerprint of each input file: {@code basename|size|lastModified} (directories are
 *       expanded to their top-level files), so re-converted / edited / swapped inputs are
 *       detected.</li>
 * </ol>
 *
 * <p>The signature is stored next to the output as {@code <output>.carafe.sig}. Computing it at
 * run time (just before a step would be skipped) makes it cascade: when an upstream output changes,
 * its fingerprint changes, so downstream signatures no longer match and those steps re-run.</p>
 */
public final class ReuseSignature {

    /** Bump if the signature scheme changes (invalidates old sidecars). */
    private static final int SCHEME_VERSION = 1;

    /** Flags whose following token is also volatile and should be dropped together. */
    private static final List<String> VOLATILE_FLAG_WITH_VALUE = List.of(
            "-jar", "-python", "--threads", "-threads", "--verbose", "-verbose");

    private ReuseSignature() {
    }

    public static String sidecarPath(String outputFile) {
        return outputFile + ".carafe.sig";
    }

    /**
     * Compute the signature for a step.
     *
     * @param commandTokens the step's argument list (preferred); may be empty
     * @param rawCmd        the step's command string (used when {@code commandTokens} is empty)
     * @param inputFiles    the files the step reads (directories are expanded to their files)
     * @return SHA-256 hex signature
     */
    public static String compute(List<String> commandTokens, String rawCmd, List<String> inputFiles) {
        String canonical = "v" + SCHEME_VERSION + "\n"
                + "cmd:" + String.join(" ", normalizeCommand(tokenize(commandTokens, rawCmd))) + "\n"
                + fingerprintInputs(inputFiles);
        return sha256Hex(canonical);
    }

    /** Argument tokens: {@code commandTokens} when non-empty, else {@code rawCmd} tokenized. */
    static List<String> tokenize(List<String> commandTokens, String rawCmd) {
        if (commandTokens != null && !commandTokens.isEmpty()) {
            return new ArrayList<>(commandTokens);
        }
        if (rawCmd == null || rawCmd.isBlank()) {
            return new ArrayList<>();
        }
        return new ArrayList<>(List.of(Commandline.translateCommandline(rawCmd)));
    }

    /** Drop machine/run-volatile tokens so the signature reflects analysis parameters only. */
    static List<String> normalizeCommand(List<String> tokens) {
        List<String> out = new ArrayList<>();
        for (int i = 0; i < tokens.size(); i++) {
            String t = tokens.get(i);
            if (i == 0) {
                continue; // the executable (java / msconvert / diann / osprey / Carafe.exe)
            }
            if (t.startsWith("-Xmx") || t.startsWith("-Xms") || t.startsWith("-Djava.security.manager")) {
                continue;
            }
            if (VOLATILE_FLAG_WITH_VALUE.contains(t)) {
                i++; // also skip this flag's value
                continue;
            }
            out.add(t);
        }
        return out;
    }

    /** Canonical fingerprint string for the inputs (sorted by basename; directories expanded). */
    static String fingerprintInputs(List<String> inputFiles) {
        List<String> lines = new ArrayList<>();
        if (inputFiles != null) {
            for (String path : inputFiles) {
                if (path == null || path.isBlank()) {
                    continue;
                }
                File f = new File(path);
                if (f.isDirectory()) {
                    File[] kids = f.listFiles(File::isFile);
                    if (kids != null) {
                        for (File k : kids) {
                            lines.add(fingerprintFile(k));
                        }
                    }
                } else {
                    lines.add(fingerprintFile(f));
                }
            }
        }
        lines.sort(String::compareTo);
        return String.join("\n", lines);
    }

    private static String fingerprintFile(File f) {
        if (f.exists()) {
            return "in:" + f.getName() + "|" + f.length() + "|" + f.lastModified();
        }
        return "in:" + f.getName() + "|MISSING";
    }

    /**
     * Write the sidecar JSON next to {@code outputFile}. {@code normalizedCmd} and {@code inputs}
     * are stored for human inspection; only {@code signature} is compared on reuse.
     */
    public static void writeSidecar(String outputFile, String signature, String normalizedCmd,
            List<String> inputs) {
        try {
            JSONObject root = new JSONObject();
            root.put("version", SCHEME_VERSION);
            root.put("sig", signature);
            root.put("command", normalizedCmd);
            JSONArray arr = new JSONArray();
            if (inputs != null) {
                for (String path : inputs) {
                    if (path == null || path.isBlank()) {
                        continue;
                    }
                    File f = new File(path);
                    JSONObject in = new JSONObject();
                    in.put("name", f.getName());
                    in.put("path", path);
                    in.put("size", f.exists() ? f.length() : -1);
                    in.put("mtime", f.exists() ? f.lastModified() : -1);
                    arr.add(in);
                }
            }
            root.put("inputs", arr);
            Files.writeString(new File(sidecarPath(outputFile)).toPath(),
                    JSON.toJSONString(root), StandardCharsets.UTF_8);
        } catch (Exception ignore) {
            // Best effort: a missing/unwritable sidecar just means the step re-runs next time.
        }
    }

    /**
     * True if a sidecar exists next to {@code outputFile} and its stored signature equals
     * {@code currentSignature}. False (so the step re-runs) when the sidecar is absent/unreadable.
     */
    public static boolean matches(String outputFile, String currentSignature) {
        File sc = new File(sidecarPath(outputFile));
        if (!sc.isFile()) {
            return false;
        }
        try {
            JSONObject root = JSON.parseObject(Files.readString(sc.toPath(), StandardCharsets.UTF_8));
            return currentSignature != null && currentSignature.equals(root.getString("sig"));
        } catch (Exception e) {
            return false;
        }
    }

    private static String sha256Hex(String s) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] digest = md.digest(s.getBytes(StandardCharsets.UTF_8));
            StringBuilder sb = new StringBuilder(digest.length * 2);
            for (byte b : digest) {
                sb.append(Character.forDigit((b >> 4) & 0xf, 16));
                sb.append(Character.forDigit(b & 0xf, 16));
            }
            return sb.toString();
        } catch (Exception e) {
            // SHA-256 is guaranteed present; fall back to a stable hash if not.
            return Integer.toHexString(s.hashCode());
        }
    }
}
