package test.java.ai;

import main.java.ai.SkylineIO;
import org.junit.Test;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.HashMap;

public class SkylineIOTest {

    @Test
    public void testGet_unimod_from_peptide() throws NoSuchMethodException, InvocationTargetException, IllegalAccessException {
        String mod_pep = "SSSFSC(unimod:4)PE";
        Method method = SkylineIO.class.getDeclaredMethod("get_unimod_from_peptide", String.class);
        method.setAccessible(true);
        HashMap<Integer, String> result = (HashMap<Integer, String>) method.invoke(null, mod_pep);
        org.junit.Assert.assertEquals(result.get(6), "C(unimod:4)");

        mod_pep = "SSSFSC(unimod:4)PE(unimod:35)";
        result = (HashMap<Integer, String>) method.invoke(null, mod_pep);
        org.junit.Assert.assertEquals(result.get(6), "C(unimod:4)");
        org.junit.Assert.assertEquals(result.get(8), "E(unimod:35)");

        mod_pep = "SSSFSC(unimod:4)PC";
        result = (HashMap<Integer, String>) method.invoke(null, mod_pep);
        org.junit.Assert.assertEquals(result.get(6), "C(unimod:4)");

        mod_pep = "SSSFSCPC(unimod:4)";
        result = (HashMap<Integer, String>) method.invoke(null, mod_pep);
        org.junit.Assert.assertEquals(result.get(8), "C(unimod:4)");

        mod_pep = "(unimod:41)SSSFSCPC(unimod:4)";
        result = (HashMap<Integer, String>) method.invoke(null, mod_pep);
        org.junit.Assert.assertEquals(result.get(0), "unimod:41");
        org.junit.Assert.assertEquals(result.get(8), "C(unimod:4)");
    }
}
