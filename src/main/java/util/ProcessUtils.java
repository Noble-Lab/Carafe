package main.java.util;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * Helpers for managing external processes launched by the GUI (MSConvert, DIA-NN, Osprey,
 * Carafe, Python). Extracted from the GUI so the termination logic is unit-testable.
 */
public final class ProcessUtils {

    private ProcessUtils() {
    }

    /**
     * Forcibly terminate each process and <em>all of its descendants</em>, then wait briefly for
     * them to exit. Used by the Stop button and the JVM shutdown hook so that closing the GUI
     * mid-run never leaves orphaned converter processes behind. Best effort: null/dead entries and
     * any per-process errors are ignored.
     *
     * @param processes the processes to terminate (e.g. {@code cmd /c msconvert ...} wrappers whose
     *                  child is the real converter)
     */
    public static void terminateAll(Collection<Process> processes) {
        List<Process> list = new ArrayList<>(processes);
        // First pass: kill descendants (the actual converter spawned by the shell), then the
        // process itself.
        for (Process p : list) {
            try {
                if (p != null && p.isAlive()) {
                    p.descendants().forEach(ProcessHandle::destroyForcibly);
                    p.destroyForcibly();
                }
            } catch (Exception ignore) {
                // best effort
            }
        }
        // Second pass: give them a moment to actually die.
        for (Process p : list) {
            if (p == null) {
                continue;
            }
            try {
                p.waitFor(2, TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            } catch (Exception ignore) {
                // best effort
            }
        }
    }
}
