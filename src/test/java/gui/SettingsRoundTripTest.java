package test.java.gui;

import main.java.gui.CarafeGUI;
import main.java.input.CParameter;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

import javax.swing.JComboBox;
import javax.swing.JTextField;
import javax.swing.SwingUtilities;
import java.awt.GraphicsEnvironment;
import java.io.File;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Saving and reloading a run must restore it exactly, including the TMT controls.
 * <p>
 * These two settings are easy to get wrong because the labelling control rewrites the fixed
 * modifications and the MS2 base model whenever it changes. Applied naively during a load, that
 * would overwrite the values being restored and silently discard a deliberate override, for
 * example "TMT labelling but predict with the generic model".
 */
public class SettingsRoundTripTest {

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

    @Test
    public void testTmtSettingsSurviveSaveAndLoad() throws Exception {
        if (GraphicsEnvironment.isHeadless()) {
            throw new SkipException("no display available");
        }
        CParameter.init();
        File tmp = Files.createTempFile("carafe_settings_", ".json").toFile();
        tmp.deleteOnExit();

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
                    JTextField fixMod = (JTextField) field(gui, "fixModSelectedField");

                    labeling.setSelectedItem("TMTpro (16/18/32/35-plex)");
                    String savedFixMod = fixMod.getText();
                    Assert.assertNotEquals(savedFixMod, "1", "the preset should have added the TMT labels");

                    // A deliberate override: TMT labelling, but predict with the generic model.
                    ms2Model.setSelectedItem("Generic (AlphaPeptDeep)");

                    invoke(gui, "saveSettingsToFile", tmp);

                    // Wipe the state, then restore it.
                    labeling.setSelectedItem("None");
                    ms2Model.setSelectedItem("TMT base model (bundled)");
                    fixMod.setText("1");

                    invoke(gui, "loadSettingsFromFile", tmp);

                    Assert.assertEquals(labeling.getSelectedItem(), "TMTpro (16/18/32/35-plex)",
                            "the labelling was not restored");
                    Assert.assertEquals(ms2Model.getSelectedItem(), "Generic (AlphaPeptDeep)",
                            "the deliberate generic-model override was not restored");
                    Assert.assertEquals(fixMod.getText(), savedFixMod,
                            "the fixed modifications were not restored");
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            });
        } finally {
            SwingUtilities.invokeAndWait(gui::dispose);
        }
    }

    @Test
    public void testCustomModelPathSurvivesSaveAndLoad() throws Exception {
        if (GraphicsEnvironment.isHeadless()) {
            throw new SkipException("no display available");
        }
        CParameter.init();
        File tmp = Files.createTempFile("carafe_settings_custom_", ".json").toFile();
        tmp.deleteOnExit();

        AtomicReference<CarafeGUI> ref = new AtomicReference<>();
        SwingUtilities.invokeAndWait(() -> ref.set(new CarafeGUI()));
        CarafeGUI gui = ref.get();
        try {
            SwingUtilities.invokeAndWait(() -> {
                try {
                    @SuppressWarnings("unchecked")
                    JComboBox<String> ms2Model = (JComboBox<String>) field(gui, "ms2ModelCombo");
                    JTextField ms2Field = (JTextField) field(gui, "ms2ModelField");

                    ms2Model.setSelectedItem("Custom file...");
                    ms2Field.setText("C:/models/my_tmt_model.pt");
                    invoke(gui, "saveSettingsToFile", tmp);

                    ms2Model.setSelectedItem("Generic (AlphaPeptDeep)");
                    ms2Field.setText("");
                    invoke(gui, "loadSettingsFromFile", tmp);

                    Assert.assertEquals(ms2Model.getSelectedItem(), "Custom file...");
                    Assert.assertEquals(ms2Field.getText(), "C:/models/my_tmt_model.pt",
                            "the custom model path was not restored");
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            });
        } finally {
            SwingUtilities.invokeAndWait(gui::dispose);
        }
    }
}
