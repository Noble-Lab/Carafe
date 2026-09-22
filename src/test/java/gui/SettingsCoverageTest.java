package test.java.gui;

import main.java.gui.CarafeGUI;
import main.java.input.CParameter;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JComponent;
import javax.swing.JSpinner;
import javax.swing.JTextField;
import javax.swing.SwingUtilities;
import java.awt.GraphicsEnvironment;
import java.io.File;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Guards against a new GUI control being added without being wired into the places that have to
 * know about it.
 * <p>
 * A control has to be registered in three separate lists, and nothing but this test connects
 * them: the command assembly (which builds the Carafe command line), {@code settingsRegistry()}
 * (save/load), and {@code initSettingChangeIndicators()} (the orange "changed from default"
 * label). Missing the second means a saved run silently comes back with the wrong value; missing
 * the third means a non-default setting looks untouched. Both have happened.
 * <p>
 * The check is behavioural rather than structural: every input is changed, saved, clobbered and
 * reloaded, and must come back with the value that was saved. That is the property users care
 * about, and it cannot be satisfied by registering a control under the wrong key.
 */
public class SettingsCoverageTest {

    /**
     * Inputs that deliberately are not saved to the settings file, with the reason. Anything not
     * listed here must round-trip.
     */
    private static final Set<String> NOT_SAVED = new HashSet<>(Arrays.asList(
            // Machine-specific executable locations, persisted separately via java Preferences.
            "pythonPathCombo",
            "diannPathCombo",
            "msConvertPathCombo",
            // Pickers, not state: choosing an entry appends its id to the "...Selected" text
            // field, and that field is what gets saved.
            "fixModAvailableCombo",
            "varModAvailableCombo",
            // A per-invocation choice about skipping completed steps, not an analysis parameter.
            "reuseResultsCheckbox"
    ));

    private static Object field(CarafeGUI gui, String name) throws Exception {
        Field f = CarafeGUI.class.getDeclaredField(name);
        f.setAccessible(true);
        return f.get(gui);
    }

    private static void invoke(CarafeGUI gui, String name, File file) throws Exception {
        Method m = CarafeGUI.class.getDeclaredMethod(name, File.class);
        m.setAccessible(true);
        m.invoke(gui, file);
    }

    /** Every input component declared on the form, by field name. */
    private static List<Field> inputFields() {
        List<Field> inputs = new ArrayList<>();
        for (Field f : CarafeGUI.class.getDeclaredFields()) {
            Class<?> t = f.getType();
            if (JComboBox.class.isAssignableFrom(t) || JTextField.class.isAssignableFrom(t)
                    || JSpinner.class.isAssignableFrom(t) || JCheckBox.class.isAssignableFrom(t)) {
                f.setAccessible(true);
                inputs.add(f);
            }
        }
        return inputs;
    }

    /** Current value of an input, comparable across a save/load cycle. */
    private static Object valueOf(JComponent c) {
        if (c instanceof JTextField tf) {
            return tf.getText();
        }
        if (c instanceof JSpinner sp) {
            return sp.getValue();
        }
        if (c instanceof JComboBox<?> cb) {
            return cb.getSelectedItem();
        }
        if (c instanceof JCheckBox ck) {
            return ck.isSelected();
        }
        return null;
    }

    /** Change an input to something different, or return false when it has no other value. */
    private static boolean perturb(JComponent c) {
        if (c instanceof JTextField tf) {
            tf.setText(tf.getText() + "_x");
            return true;
        }
        if (c instanceof JSpinner sp) {
            Object next = sp.getModel().getNextValue();
            if (next == null) {
                next = sp.getModel().getPreviousValue();
            }
            if (next == null) {
                return false;
            }
            sp.setValue(next);
            return true;
        }
        if (c instanceof JComboBox<?> cb) {
            if (cb.getItemCount() < 2) {
                return false;
            }
            cb.setSelectedIndex((cb.getSelectedIndex() + 1) % cb.getItemCount());
            return true;
        }
        if (c instanceof JCheckBox ck) {
            ck.setSelected(!ck.isSelected());
            return true;
        }
        return false;
    }

    @Test
    public void testEveryInputIsSavedAndRestored() throws Exception {
        if (GraphicsEnvironment.isHeadless()) {
            throw new SkipException("no display available");
        }
        CParameter.init();
        File tmp = Files.createTempFile("carafe_coverage_", ".json").toFile();
        tmp.deleteOnExit();

        AtomicReference<CarafeGUI> ref = new AtomicReference<>();
        SwingUtilities.invokeAndWait(() -> ref.set(new CarafeGUI()));
        CarafeGUI gui = ref.get();

        List<String> notRestored = new ArrayList<>();
        List<String> unchangeable = new ArrayList<>();
        try {
            for (Field f : inputFields()) {
                String name = f.getName();
                if (NOT_SAVED.contains(name)) {
                    continue;
                }
                SwingUtilities.invokeAndWait(() -> {
                    try {
                        JComponent input = (JComponent) f.get(gui);
                        if (input == null) {
                            return;
                        }
                        if (!perturb(input)) {
                            unchangeable.add(name);
                            return;
                        }
                        Object expected = valueOf(input);
                        invoke(gui, "saveSettingsToFile", tmp);

                        // Clobber it, then restore from the file.
                        perturb(input);
                        invoke(gui, "loadSettingsFromFile", tmp);

                        if (!java.util.Objects.equals(valueOf(input), expected)) {
                            notRestored.add(name + " (saved " + expected + ", got " + valueOf(input) + ")");
                        }
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                });
            }
        } finally {
            SwingUtilities.invokeAndWait(gui::dispose);
        }

        Assert.assertTrue(notRestored.isEmpty(),
                "These inputs did not survive a settings save/load. Add them to settingsRegistry(), "
                        + "or to NOT_SAVED in this test with a reason:\n  "
                        + String.join("\n  ", notRestored));
        // Not a failure, but surfaces controls this test cannot exercise.
        if (!unchangeable.isEmpty()) {
            System.out.println("SettingsCoverageTest could not vary: " + unchangeable);
        }
    }

    /**
     * Inputs that legitimately do not appear in the Carafe command line, with the reason.
     * Anything not listed here must change the command when it changes.
     */
    private static final Set<String> NOT_IN_CARAFE_COMMAND = new HashSet<>(Arrays.asList(
            // These configure the DIA-NN search, which is a separate command.
            "trainDbFileField", "projectMsFileField", "diannPathCombo", "diannAdditionalOptionsField",
            // This one drives msconvert.
            "msConvertPathCombo",
            // Pipeline controls: they decide which commands run, not what Carafe is told.
            "workflowCombo", "benchmarkCheckbox", "reuseResultsCheckbox",
            // The isobaric label has no flag of its own. It is expressed through the fixed
            // modifications and the MS2 base model, both of which are checked in their own right.
            "labelingCombo"
    ));

    /**
     * Every analysis input must actually reach the Carafe command line.
     * <p>
     * This is the third list a control has to be wired into, alongside saving and the change
     * indicator, and it is the one with the worst failure mode: a control that is saved and
     * highlighted but never added to the command looks entirely correct in the GUI while the run
     * silently uses the default. Selecting a TMT reagent and still getting a library built with
     * the generic model would be exactly that bug.
     */
    @Test
    public void testEveryInputReachesTheCarafeCommandLine() throws Exception {
        if (GraphicsEnvironment.isHeadless()) {
            throw new SkipException("no display available");
        }
        CParameter.init();

        AtomicReference<CarafeGUI> ref = new AtomicReference<>();
        SwingUtilities.invokeAndWait(() -> ref.set(new CarafeGUI()));
        CarafeGUI gui = ref.get();

        Method build = CarafeGUI.class.getDeclaredMethod("buildCarafeCommand");
        build.setAccessible(true);

        // The labelling control also rewrites the fixed modifications and the MS2 base model,
        // and those DO reach the command line. Left enabled, that side effect would make
        // labelingCombo look wired up even if its own -labeling flag were missing, which is
        // precisely the bug this test exists to catch. Suppress it so each input has to change
        // the command on its own account.
        Field suppress = CarafeGUI.class.getDeclaredField("suppressLabelingPreset");
        suppress.setAccessible(true);
        suppress.setBoolean(gui, true);

        List<String> noEffect = new ArrayList<>();
        List<String> unchangeable = new ArrayList<>();
        try {
            for (Field f : inputFields()) {
                String name = f.getName();
                if (NOT_IN_CARAFE_COMMAND.contains(name)) {
                    continue;
                }
                SwingUtilities.invokeAndWait(() -> {
                    try {
                        JComponent input = (JComponent) f.get(gui);
                        if (input == null) {
                            return;
                        }
                        // The custom model path only reaches the command line when that source is
                        // selected, so set the precondition rather than depend on field order.
                        if ("ms2ModelField".equals(name)) {
                            ((JComboBox<?>) field(gui, "ms2ModelCombo")).setSelectedItem("Custom file...");
                        }
                        String before = commandOf(build, gui);
                        if (!perturb(input)) {
                            unchangeable.add(name);
                            return;
                        }
                        if (before.equals(commandOf(build, gui))) {
                            noEffect.add(name);
                        }
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                });
            }
        } finally {
            suppress.setBoolean(gui, false);
            SwingUtilities.invokeAndWait(gui::dispose);
        }

        Assert.assertTrue(noEffect.isEmpty(),
                "Changing these inputs does not change the Carafe command line, so their value "
                        + "never reaches Carafe. Add them to buildCarafeCommand(), or to "
                        + "NOT_IN_CARAFE_COMMAND in this test with a reason:\n  "
                        + String.join("\n  ", noEffect));
        if (!unchangeable.isEmpty()) {
            System.out.println("SettingsCoverageTest could not vary (command line): " + unchangeable);
        }
    }

    private static String commandOf(Method build, CarafeGUI gui) throws Exception {
        Object task = build.invoke(gui);
        return (String) task.getClass().getField("cmd").get(task);
    }

    /**
     * Every saved analysis input should also drive the "changed from default" label highlight.
     * File, path and free-text option inputs are excluded from that indicator by design.
     */
    @Test
    @SuppressWarnings("unchecked")
    public void testEveryAnalysisInputHasAChangeIndicator() throws Exception {
        if (GraphicsEnvironment.isHeadless()) {
            throw new SkipException("no display available");
        }
        CParameter.init();

        Set<String> noIndicator = new HashSet<>(NOT_SAVED);
        // Paths and file selections: the indicator is for analysis parameters only.
        noIndicator.addAll(Arrays.asList(
                "workflowCombo", "diannReportFileField", "trainMsFileField", "trainDbFileField",
                "projectMsFileField", "libraryDbFileField", "outputDirField", "ms2ModelField"));

        AtomicReference<CarafeGUI> ref = new AtomicReference<>();
        SwingUtilities.invokeAndWait(() -> ref.set(new CarafeGUI()));
        CarafeGUI gui = ref.get();
        List<String> missing = new ArrayList<>();
        try {
            Field tsField = CarafeGUI.class.getDeclaredField("trackedSettings");
            tsField.setAccessible(true);
            List<?> tracked = (List<?>) tsField.get(gui);

            Set<Object> trackedInputs = new HashSet<>();
            for (Object ts : tracked) {
                Field inputField = ts.getClass().getDeclaredField("input");
                inputField.setAccessible(true);
                trackedInputs.add(inputField.get(ts));
            }

            for (Field f : inputFields()) {
                if (noIndicator.contains(f.getName())) {
                    continue;
                }
                Object input = f.get(gui);
                if (input != null && !trackedInputs.contains(input)) {
                    missing.add(f.getName());
                }
            }
        } finally {
            SwingUtilities.invokeAndWait(gui::dispose);
        }

        Assert.assertTrue(missing.isEmpty(),
                "These inputs have no changed-from-default label indicator. Add them to "
                        + "initSettingChangeIndicators(), or to the exclusion list in this test:\n  "
                        + String.join("\n  ", missing));
    }
}
