package test.java.gui;

import main.java.gui.CarafeGUI;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.reflect.Method;

/**
 * The predict-only workflow has no MS data, so an NCE of "auto" cannot be resolved.
 * <p>
 * Left unchecked it falls through to Carafe's default of 27, which is outside the range the
 * bundled TMT model was trained on (30-38) and measurably degrades the library -- without the
 * user ever having chosen 27.
 */
public class PredictOnlyValidationTest {

    private static boolean usable(String nce) throws Exception {
        Method m = CarafeGUI.class.getDeclaredMethod("nceIsUsableWithoutMsData", String.class);
        m.setAccessible(true);
        return (boolean) m.invoke(null, nce);
    }

    @Test
    public void testAutoAndEmptyAreRejected() throws Exception {
        for (String nce : new String[]{"auto", "AUTO", " Auto ", "", "   ", null}) {
            Assert.assertFalse(usable(nce),
                    String.valueOf(nce) + " cannot be resolved without MS data");
        }
    }

    @Test
    public void testExplicitValuesAreAccepted() throws Exception {
        for (String nce : new String[]{"27", "30", "32.5", " 38 "}) {
            Assert.assertTrue(usable(nce), nce + " is an explicit collision energy");
        }
    }
}
