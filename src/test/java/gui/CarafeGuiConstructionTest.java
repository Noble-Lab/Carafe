package test.java.gui;

import main.java.gui.CarafeGUI;
import main.java.input.CParameter;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

import java.awt.GraphicsEnvironment;
import java.lang.reflect.Field;
import javax.swing.JComboBox;
import javax.swing.JComponent;
import javax.swing.JLabel;
import javax.swing.SwingUtilities;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Builds the whole Carafe window once, without showing it.
 * <p>
 * Every panel is assembled in the constructor, so a layout mistake in a newly added row -- a
 * duplicated grid position, a component used before it is created, a null in a combo model --
 * throws here rather than in front of a user. The constructor does not call setVisible, so no
 * window appears; only {@code main} does that.
 */
public class CarafeGuiConstructionTest {

    private static Object field(CarafeGUI gui, String name) throws Exception {
        Field f = CarafeGUI.class.getDeclaredField(name);
        f.setAccessible(true);
        return f.get(gui);
    }

    @Test
    public void testWindowBuildsAndExposesTheTmtControls() throws Exception {
        if (GraphicsEnvironment.isHeadless()) {
            throw new SkipException("no display available");
        }

        AtomicReference<CarafeGUI> built = new AtomicReference<>();
        AtomicReference<Throwable> failure = new AtomicReference<>();
        CParameter.init();
        SwingUtilities.invokeAndWait(() -> {
            try {
                built.set(new CarafeGUI());
            } catch (Throwable t) {
                failure.set(t);
            }
        });

        if (failure.get() != null) {
            throw new AssertionError("Building the Carafe window failed", failure.get());
        }
        CarafeGUI gui = built.get();
        Assert.assertNotNull(gui, "the window was not built");

        try {
            @SuppressWarnings("unchecked")
            JComboBox<String> workflow = (JComboBox<String>) field(gui, "workflowCombo");
            Assert.assertEquals(workflow.getItemCount(), 5,
                    "expected the prediction-only (4th) and DDA-report (5th) workflows to be offered");
            Assert.assertTrue(workflow.getItemAt(3).contains("pretrained"),
                    "the fourth workflow should be the pretrained-model one, got: " + workflow.getItemAt(3));

            @SuppressWarnings("unchecked")
            JComboBox<String> labeling = (JComboBox<String>) field(gui, "labelingCombo");
            Assert.assertNotNull(labeling, "the labelling control is missing");
            Assert.assertEquals(labeling.getItemAt(0), "None", "unlabelled must be the default");
            Assert.assertEquals(labeling.getSelectedIndex(), 0,
                    "the GUI must open unlabelled so existing workflows are unchanged");
            Assert.assertEquals(labeling.getItemCount(), 4);

            @SuppressWarnings("unchecked")
            JComboBox<String> ms2Model = (JComboBox<String>) field(gui, "ms2ModelCombo");
            Assert.assertNotNull(ms2Model, "the MS2 base model control is missing");
            Assert.assertEquals(ms2Model.getItemCount(), 3, "expected three explicit choices");
            Assert.assertEquals(ms2Model.getSelectedItem(), "Generic (AlphaPeptDeep)",
                    "the default must be the generic model, i.e. existing behaviour");
            Assert.assertEquals(ms2Model.getItemAt(1), "TMT base model (bundled)",
                    "the bundled TMT model must be offered explicitly");
        } finally {
            SwingUtilities.invokeAndWait(gui::dispose);
        }
    }

    /**
     * Both new controls must take part in the "changed from default" label highlight that every
     * other analysis setting uses, otherwise a non-default labelling or model looks untouched.
     */
    @Test
    public void testNewControlsHighlightWhenChangedFromDefault() throws Exception {
        if (GraphicsEnvironment.isHeadless()) {
            throw new SkipException("no display available");
        }
        CParameter.init();
        AtomicReference<CarafeGUI> ref = new AtomicReference<>();
        SwingUtilities.invokeAndWait(() -> ref.set(new CarafeGUI()));
        CarafeGUI gui = ref.get();
        try {
            SwingUtilities.invokeAndWait(() -> {
                try {
                    java.lang.reflect.Method findLabelFor =
                            CarafeGUI.class.getDeclaredMethod("findLabelFor", JComponent.class);
                    findLabelFor.setAccessible(true);
                    java.lang.reflect.Field colorField =
                            CarafeGUI.class.getDeclaredField("MODIFIED_LABEL_COLOR");
                    colorField.setAccessible(true);
                    java.awt.Color modified = (java.awt.Color) colorField.get(null);

                    @SuppressWarnings("unchecked")
                    JComboBox<String> labeling = (JComboBox<String>) field(gui, "labelingCombo");
                    @SuppressWarnings("unchecked")
                    JComboBox<String> ms2Model = (JComboBox<String>) field(gui, "ms2ModelCombo");

                    JLabel labelingLabel = (JLabel) findLabelFor.invoke(gui, labeling);
                    JLabel ms2ModelLabel = (JLabel) findLabelFor.invoke(gui, ms2Model);
                    Assert.assertNotNull(labelingLabel, "no label resolved for Isobaric Labelling");
                    Assert.assertNotNull(ms2ModelLabel, "no label resolved for MS2 Base Model");

                    Assert.assertNotEquals(labelingLabel.getForeground(), modified,
                            "the default selection must not look modified");
                    Assert.assertNotEquals(ms2ModelLabel.getForeground(), modified,
                            "the default selection must not look modified");

                    // Selecting a TMT reagent changes both controls away from their defaults.
                    labeling.setSelectedItem("TMT 6-plex / 10-plex / 11-plex");

                    Assert.assertEquals(labelingLabel.getForeground(), modified,
                            "Isobaric Labelling should be highlighted once it is not None");
                    Assert.assertEquals(ms2ModelLabel.getForeground(), modified,
                            "MS2 Base Model should be highlighted once it is not the generic model");

                    labeling.setSelectedItem("None");
                    Assert.assertNotEquals(labelingLabel.getForeground(), modified,
                            "returning to None should clear the highlight");
                    Assert.assertNotEquals(ms2ModelLabel.getForeground(), modified,
                            "returning to the generic model should clear the highlight");
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            });
        } finally {
            SwingUtilities.invokeAndWait(gui::dispose);
        }
    }

    /**
     * Choosing a TMT reagent must move the MS2 Base Model control to the TMT model, so the
     * dropdown always states which model will be used, and going back to unlabelled must restore
     * the generic one. A custom file the user picked is never overwritten.
     */
    @Test
    public void testLabelingKeepsTheModelControlHonest() throws Exception {
        if (GraphicsEnvironment.isHeadless()) {
            throw new SkipException("no display available");
        }
        CParameter.init();
        AtomicReference<CarafeGUI> ref = new AtomicReference<>();
        SwingUtilities.invokeAndWait(() -> ref.set(new CarafeGUI()));
        CarafeGUI gui = ref.get();
        try {
            SwingUtilities.invokeAndWait(() -> {
                try {
                    @SuppressWarnings("unchecked")
                    JComboBox<String> labeling = (JComboBox<String>) field(gui, "labelingCombo");
                    @SuppressWarnings("unchecked")
                    JComboBox<String> ms2Model = (JComboBox<String>) field(gui, "ms2ModelCombo");

                    labeling.setSelectedItem("TMTpro (16/18/32/35-plex)");
                    Assert.assertEquals(ms2Model.getSelectedItem(), "TMT base model (bundled)",
                            "selecting a TMT reagent should select the TMT base model");

                    labeling.setSelectedItem("None");
                    Assert.assertEquals(ms2Model.getSelectedItem(), "Generic (AlphaPeptDeep)",
                            "going back to unlabelled should restore the generic model");

                    ms2Model.setSelectedItem("Custom file...");
                    labeling.setSelectedItem("TMT 2-plex");
                    Assert.assertEquals(ms2Model.getSelectedItem(), "Custom file...",
                            "a custom model chosen by the user must not be overwritten");
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            });
        } finally {
            SwingUtilities.invokeAndWait(gui::dispose);
        }
    }
}
