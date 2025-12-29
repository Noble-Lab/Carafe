package main.java.util;

import java.io.*;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.time.Duration;
import java.util.*;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

public class PyInstaller {

    /**
     * Single-function installer (Windows + Linux).
     *
     * Installs:
     *  - uv (downloaded locally)
     *  - Python 3.9.18 in a .venv under installRoot
     *  - torch 2.1.2 GPU wheels (cu121 or cu118) if NVIDIA driver supports it, else CPU wheels
     *  - pinned pip deps listed in-code
     *  - alphapeptdeep_dia from GitHub ZIP
     */
    public static String installAll(Path installRoot) throws Exception {
        // ---------------- OS guard ----------------
        String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        boolean isWindows = os.contains("win");
        boolean isLinux = os.contains("linux");
        if (!isWindows && !isLinux) {
            throw new IllegalStateException("This installer supports Windows and Linux only. Detected: " + os);
        }

        // ---------------- Your pinned deps here ----------------
        // NOTE: Do NOT include torch/torchvision/torchaudio here because we install them with a special index URL.
        final List<String> PIP_PINS = List.of(
                "alphabase==1.2.1",
                "alpharaw==0.4.3",
                "alphatims==1.0.8",
                "numpy<2",
                "pandas==2.2.3",
                "transformers==4.47.0",
                "pip"
        );

        // Choose a reproducible ZIP (tag/commit) if you want.
        final String ALPHAPEPTDEEP_DIA_ZIP = "https://github.com/wenbostar/alphapeptdeep_dia/archive/refs/heads/main.zip";

        // ---------------- Prep dirs ----------------
        Files.createDirectories(installRoot);
        Path uvDir = installRoot.resolve("uv");
        Path logsDir = installRoot.resolve("logs");
        Files.createDirectories(uvDir);
        Files.createDirectories(logsDir);

        Path logFile = logsDir.resolve("install.log");

        // ---------------- Logger ----------------
        class Log {
            synchronized void info(String msg) {
                String line = "[" + java.time.Instant.now() + "] " + msg;
                System.out.println(line);
                try {
                    Files.writeString(logFile, line + System.lineSeparator(), StandardCharsets.UTF_8,
                            StandardOpenOption.CREATE, StandardOpenOption.APPEND);
                } catch (IOException ignored) {}
            }
        }
        Log log = new Log();

        // ---------------- Command runner ----------------
        class Cmd {
            String run(List<String> command, Path workDir) throws Exception {
                log.info("RUN: " + String.join(" ", command));
                ProcessBuilder pb = new ProcessBuilder(command);
                pb.directory(workDir.toFile());
                pb.redirectErrorStream(true);

                Process p = pb.start();
                ByteArrayOutputStream baos = new ByteArrayOutputStream();
                try (InputStream is = p.getInputStream()) {
                    is.transferTo(baos);
                }
                int code = p.waitFor();
                String out = baos.toString(StandardCharsets.UTF_8);

                log.info("EXIT: " + code);
                if (!out.isBlank()) log.info("OUTPUT:\n" + out);

                if (code != 0) {
                    throw new RuntimeException("Command failed (" + code + "): " + String.join(" ", command) + "\n" + out);
                }
                return out;
            }
        }
        Cmd cmd = new Cmd();

        // ---------------- Download uv ----------------
        String uvUrl = isWindows
                ? "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
                : "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz";
        Path uvArchive = uvDir.resolve(isWindows ? "uv.zip" : "uv.tar.gz");

        log.info("Downloading uv: " + uvUrl);
        HttpClient http = HttpClient.newBuilder()
                .followRedirects(HttpClient.Redirect.ALWAYS)
                .connectTimeout(Duration.ofSeconds(30))
                .build();

        HttpRequest req = HttpRequest.newBuilder()
                .uri(URI.create(uvUrl))
                .timeout(Duration.ofMinutes(5))
                .GET()
                .build();

        HttpResponse<Path> resp = http.send(req, HttpResponse.BodyHandlers.ofFile(uvArchive));
        if (resp.statusCode() < 200 || resp.statusCode() >= 300) {
            throw new IOException("Failed to download uv zip. HTTP " + resp.statusCode());
        }

        // Unpack uv
        if (isWindows) {
            log.info("Extracting uv zip...");
            try (ZipInputStream zis = new ZipInputStream(Files.newInputStream(uvArchive))) {
                ZipEntry entry;
                while ((entry = zis.getNextEntry()) != null) {
                    if (entry.isDirectory()) continue;
                    Path outPath = uvDir.resolve(entry.getName()).normalize();
                    if (!outPath.startsWith(uvDir)) {
                        throw new SecurityException("Zip slip attempt: " + entry.getName());
                    }
                    Files.createDirectories(outPath.getParent());
                    try (OutputStream osOut = Files.newOutputStream(outPath, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {
                        zis.transferTo(osOut);
                    }
                    zis.closeEntry();
                }
            }
        } else {
            log.info("Extracting uv tar.gz...");
            cmd.run(List.of("tar", "-xzf", uvArchive.toString(), "-C", uvDir.toString()), installRoot);
        }

        // Find uv binary (may be nested depending on archive layout)
        String uvFileName = isWindows ? "uv.exe" : "uv";
        Path uvExe = uvDir.resolve(uvFileName);
        if (!Files.isRegularFile(uvExe)) {
            try (var walk = Files.walk(uvDir)) {
                uvExe = walk
                        .filter(Files::isRegularFile)
                        .filter(p -> p.getFileName().toString().equalsIgnoreCase(uvFileName))
                        .findFirst()
                        .orElseThrow(() -> new FileNotFoundException(uvFileName + " not found in " + uvDir));
            }
        }
        // Ensure executable bit on *nix
        try {
            uvExe.toFile().setExecutable(true);
        } catch (SecurityException ignored) {}
        log.info("uv: " + uvExe.toAbsolutePath());

        // ---------------- Create venv (Python 3.9.18) ----------------
        // Creates: installRoot\.venv
        cmd.run(List.of(uvExe.toString(), "venv", "--python", "3.9.18", ".venv"), installRoot);

        // ---------------- Detect NVIDIA driver (nvidia-smi) ----------------
        String driverVersion = null;
        try {
            List<String> driverCmd = isWindows
                    ? List.of("cmd.exe", "/c", "nvidia-smi --query-gpu=driver_version --format=csv,noheader")
                    : List.of("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader");
            String out = cmd.run(driverCmd, installRoot);
            for (String line : out.split("\\R")) {
                line = line.trim();
                if (!line.isEmpty()) {
                    driverVersion = line;
                    break;
                }
            }
        } catch (Exception e) {
            log.info("nvidia-smi not available; will install CPU-only PyTorch wheels.");
        }

        // Compare version strings like "546.33"
        java.util.function.BiPredicate<String, String> geVersion = (a, b) -> {
            if (a == null) return false;
            String[] as = a.trim().split("\\.");
            String[] bs = b.trim().split("\\.");
            int n = Math.max(as.length, bs.length);
            for (int i = 0; i < n; i++) {
                int ai = i < as.length ? Integer.parseInt(as[i].replaceAll("\\D+", "")) : 0;
                int bi = i < bs.length ? Integer.parseInt(bs[i].replaceAll("\\D+", "")) : 0;
                if (ai != bi) return ai > bi;
            }
            return true;
        };

        // Driver thresholds (CUDA):
        // - CUDA 12.1 baseline ~ 531.14
        // - CUDA 11.8 baseline ~ 522.06
        String torchIndexUrl;
        if (driverVersion != null && geVersion.test(driverVersion, "531.14")) {
            torchIndexUrl = "https://download.pytorch.org/whl/cu121";
            log.info("Driver " + driverVersion + " => installing torch cu121.");
        } else if (driverVersion != null && geVersion.test(driverVersion, "522.06")) {
            torchIndexUrl = "https://download.pytorch.org/whl/cu118";
            log.info("Driver " + driverVersion + " => installing torch cu118.");
        } else {
            torchIndexUrl = "https://download.pytorch.org/whl/cpu";
            log.info("Driver " + driverVersion + " => installing torch CPU wheels.");
        }

        // ---------------- Install torch 2.5.1 (+ matching torchvision/torchaudio) ----------------
        cmd.run(List.of(
                uvExe.toString(), "pip", "install",
                "torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1",
                "--index-url", torchIndexUrl
        ), installRoot);

        // ---------------- Install your pinned deps (in-code list) ----------------
        // Do one big install command. If you have a LOT of pins, you can chunk it.
        {
            List<String> pipCmd = new ArrayList<>();
            pipCmd.add(uvExe.toString());
            pipCmd.add("pip");
            pipCmd.add("install");
            pipCmd.addAll(PIP_PINS);
            cmd.run(pipCmd, installRoot);
        }

        // ---------------- Install alphapeptdeep_dia from GitHub ZIP ----------------
        cmd.run(List.of(
                uvExe.toString(), "pip", "install",
                ALPHAPEPTDEEP_DIA_ZIP
        ), installRoot);

        // ---------------- Sanity checks ----------------
        cmd.run(List.of(
                uvExe.toString(), "run", "python",
                "-c",
                "import sys, torch; print(sys.version); print('torch', torch.__version__); "
                        + "print('cuda available', torch.cuda.is_available()); print('torch cuda', torch.version.cuda)"
        ), installRoot);

        log.info("Done. Venv: " + installRoot.resolve(".venv").toAbsolutePath());
        log.info("To run: " + uvExe.toAbsolutePath() + " run python");
        Path pyPath = isWindows
                ? installRoot.resolve(".venv/Scripts/python.exe")
                : installRoot.resolve(".venv/bin/python3");
        String py_path = pyPath.toAbsolutePath().toString();
        return py_path;
    }
}
