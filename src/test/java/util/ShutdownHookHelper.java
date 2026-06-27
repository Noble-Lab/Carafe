package test.java.util;

import main.java.util.ProcessUtils;

import java.util.List;

/**
 * Test support (NOT a test class): a tiny program that mirrors what the GUI does — it spawns a
 * long-running child process and registers a JVM shutdown hook that terminates it via
 * {@link ProcessUtils#terminateAll}. It prints the child's PID, then exits normally, which fires
 * the hook. {@code ProcessUtilsTest} runs this as a subprocess and verifies the child was killed
 * when this JVM exited (the orphaned-MSConvert scenario).
 */
public final class ShutdownHookHelper {

    private ShutdownHookHelper() {
    }

    public static void main(String[] args) throws Exception {
        boolean win = System.getProperty("os.name").toLowerCase().contains("win");
        Process sleeper = win
                ? new ProcessBuilder("cmd", "/c", "ping -n 60 127.0.0.1 > NUL").start()
                : new ProcessBuilder("bash", "-c", "sleep 60; echo done").start();

        Runtime.getRuntime().addShutdownHook(
                new Thread(() -> ProcessUtils.terminateAll(List.of(sleeper))));

        // Wait for the shell to spawn its child, then report that child's PID.
        for (int i = 0; i < 30 && sleeper.descendants().findAny().isEmpty(); i++) {
            Thread.sleep(100);
        }
        long childPid = sleeper.descendants().findFirst().map(ProcessHandle::pid).orElse(sleeper.pid());
        System.out.println("CHILD_PID=" + childPid);
        System.out.flush();

        Thread.sleep(300);
        System.exit(0); // triggers the shutdown hook -> terminateAll -> child is killed
    }
}
