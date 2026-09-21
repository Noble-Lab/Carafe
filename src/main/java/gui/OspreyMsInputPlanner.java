package main.java.gui;

import java.io.File;
import java.util.List;

/**
 * Small pure helpers for validating the MS inputs of the Osprey workflows, split out of
 * {@code CarafeGUI.runCarafe()} so they can be unit-tested without the Swing layer.
 */
public final class OspreyMsInputPlanner {

    private OspreyMsInputPlanner() {
    }

    /**
     * Whether every path shares the same (non-null) parent directory. Carafe's {@code -ms} argument
     * takes a single file or ONE folder, so when multiple training files are handed to it as their
     * parent directory they must all live in that same directory — otherwise files from other folders
     * are silently dropped. A single path (or empty list) is trivially "same directory".
     *
     * @param paths the resolved MS file paths
     * @return true if all paths share one parent directory
     */
    public static boolean allInSameDirectory(List<String> paths) {
        if (paths == null || paths.size() <= 1) {
            return true;
        }
        String parent = new File(paths.get(0)).getParent();
        if (parent == null) {
            return false;
        }
        for (String p : paths) {
            if (!parent.equals(new File(p).getParent())) {
                return false;
            }
        }
        return true;
    }
}
