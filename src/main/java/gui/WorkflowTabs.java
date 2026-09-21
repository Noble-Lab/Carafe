package main.java.gui;

import javax.swing.JTabbedPane;
import java.util.ArrayList;
import java.util.List;

/**
 * Tab-selection helper for the Carafe GUI, split out of {@code CarafeGUI} so the "which tabs freeze
 * during a run" rule can be unit-tested without the full Swing frame.
 */
public final class WorkflowTabs {

    /** The one tab that stays interactive while a workflow runs. */
    public static final String CONSOLE_TAB = "Console";

    private WorkflowTabs() {
    }

    /**
     * Indices of the input/settings tabs to freeze during a run: every tab except the Console.
     * Selecting by "not Console" rather than a fixed count keeps every workflow tab locked as tabs are
     * added (e.g. the Osprey tab), instead of silently leaving a new settings tab editable mid-run.
     *
     * @param pane the workflow tabbed pane
     * @return the indices of the tabs that should be frozen
     */
    public static List<Integer> freezableTabIndices(JTabbedPane pane) {
        List<Integer> indices = new ArrayList<>();
        if (pane == null) {
            return indices;
        }
        for (int i = 0; i < pane.getTabCount(); i++) {
            if (!CONSOLE_TAB.equals(pane.getTitleAt(i))) {
                indices.add(i);
            }
        }
        return indices;
    }
}
