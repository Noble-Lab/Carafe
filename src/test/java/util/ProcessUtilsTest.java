package test.java.util;

import main.java.util.ProcessUtils;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Tests for {@link ProcessUtils#terminateAll}: a tracked process and its descendants (the real
 * converter spawned by the shell wrapper) must be killed, and empty/dead inputs handled cleanly.
 *
 * <p>Cross-platform: a long-running child is spawned via {@code ping} on Windows or {@code sleep}
 * on Unix, wrapped in a shell so there is a parent + child to verify descendant termination.</p>
 */
public class ProcessUtilsTest {

    private static boolean isWindows() {
        return System.getProperty("os.name").toLowerCase().contains("win");
    }

    /** A shell wrapper (parent) whose child is a ~60s sleeper, so descendants() is non-empty. */
    private static Process spawnWrappedSleeper() throws IOException {
        if (isWindows()) {
            return new ProcessBuilder("cmd", "/c", "ping -n 60 127.0.0.1 > NUL").start();
        }
        // A compound command so bash does NOT exec-optimize into the sleeper; it stays the parent
        // with `sleep` as a child, giving us a descendant to verify.
        return new ProcessBuilder("bash", "-c", "sleep 60; echo done").start();
    }

    @Test
    public void terminatesRunningProcess() throws Exception {
        Process p = spawnWrappedSleeper();
        Assert.assertTrue(p.isAlive(), "process should be alive after start");
        ProcessUtils.terminateAll(List.of(p));
        Assert.assertFalse(p.isAlive(), "process should be terminated");
    }

    @Test
    public void terminatesDescendants() throws Exception {
        Process parent = spawnWrappedSleeper();
        // Give the shell a moment to spawn its child (ping/sleep).
        for (int i = 0; i < 20 && parent.descendants().findAny().isEmpty(); i++) {
            Thread.sleep(100);
        }
        List<ProcessHandle> kids = parent.descendants().collect(Collectors.toList());
        Assert.assertFalse(kids.isEmpty(), "expected a child process under the shell wrapper");

        ProcessUtils.terminateAll(List.of(parent));

        Assert.assertFalse(parent.isAlive(), "parent should be terminated");
        // Allow a brief moment for the OS to reap the killed children.
        for (int i = 0; i < 20; i++) {
            boolean anyAlive = kids.stream().anyMatch(ProcessHandle::isAlive);
            if (!anyAlive) {
                break;
            }
            Thread.sleep(100);
        }
        for (ProcessHandle h : kids) {
            Assert.assertFalse(h.isAlive(), "descendant process should be killed (pid " + h.pid() + ")");
        }
    }

    @Test
    public void shutdownHookKillsChildrenWhenJvmExits() throws Exception {
        // Run a separate JVM that registers a shutdown hook (like the GUI) and spawns a child,
        // then exits; verify the child is killed when that JVM goes away.
        String javaBin = System.getProperty("java.home") + java.io.File.separator + "bin"
                + java.io.File.separator + "java";
        String cp = System.getProperty("java.class.path");
        Process helper = new ProcessBuilder(javaBin, "-cp", cp, "test.java.util.ShutdownHookHelper")
                .redirectErrorStream(true)
                .start();

        long childPid = -1;
        try (java.io.BufferedReader r = new java.io.BufferedReader(
                new java.io.InputStreamReader(helper.getInputStream()))) {
            String line;
            while ((line = r.readLine()) != null) {
                if (line.startsWith("CHILD_PID=")) {
                    childPid = Long.parseLong(line.substring("CHILD_PID=".length()).trim());
                }
            }
        }
        helper.waitFor(30, java.util.concurrent.TimeUnit.SECONDS);
        Assert.assertTrue(childPid > 0, "helper should have reported a child PID");

        // After the helper JVM exited, its shutdown hook should have killed the child.
        boolean alive = true;
        for (int i = 0; i < 30; i++) {
            alive = ProcessHandle.of(childPid).map(ProcessHandle::isAlive).orElse(false);
            if (!alive) {
                break;
            }
            Thread.sleep(100);
        }
        Assert.assertFalse(alive,
                "shutdown hook should have killed the child process on JVM exit (pid " + childPid + ")");
    }

    @Test
    public void handlesEmptyAndAlreadyDeadProcesses() throws Exception {
        // Empty collection: no exception.
        ProcessUtils.terminateAll(List.of());
        // Already-finished process: no exception.
        Process quick = new ProcessBuilder(isWindows() ? List.of("cmd", "/c", "echo done")
                : List.of("true")).start();
        quick.waitFor();
        Assert.assertFalse(quick.isAlive());
        ProcessUtils.terminateAll(List.of(quick)); // must not throw
    }
}
