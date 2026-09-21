package test.java.gui;

import main.java.gui.WorkflowTabs;
import org.testng.Assert;
import org.testng.annotations.Test;

import javax.swing.JPanel;
import javax.swing.JTabbedPane;
import java.util.List;

/**
 * Coverage for {@link WorkflowTabs#freezableTabIndices}: every input/settings tab (including the
 * Osprey tab that regressed the old fixed-count logic) must be frozen during a run, and only the
 * Console left interactive.
 */
public class WorkflowTabsTest {

    private JTabbedPane paneWithTabs(String... titles) {
        JTabbedPane pane = new JTabbedPane();
        for (String t : titles) {
            pane.addTab(t, new JPanel());
        }
        return pane;
    }

    @Test
    public void freezesEveryTabExceptConsole() {
        // The real Carafe tab order, with the Osprey tab (index 4) that the old code left editable.
        JTabbedPane pane = paneWithTabs("Workflow", "Training Data Generation", "Model Training",
                "Library Generation", "Osprey", "Console");
        Assert.assertEquals(WorkflowTabs.freezableTabIndices(pane), List.of(0, 1, 2, 3, 4),
                "all settings tabs including Osprey are frozen; Console (index 5) is not");
    }

    @Test
    public void consoleInAnyPositionIsExcluded() {
        JTabbedPane pane = paneWithTabs("Console", "Workflow", "Osprey");
        Assert.assertEquals(WorkflowTabs.freezableTabIndices(pane), List.of(1, 2));
    }

    @Test
    public void nullPaneYieldsNoIndices() {
        Assert.assertTrue(WorkflowTabs.freezableTabIndices(null).isEmpty());
    }
}
