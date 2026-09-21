package test.java.koina;

import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;
import main.java.koina.KoinaClient;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Tests for {@link KoinaClient}'s static request-building and response-parsing (no network).
 * The static helpers are package-private, so they're invoked via reflection.
 */
public class KoinaClientTest {

    @SuppressWarnings("unchecked")
    private static <T> T call(String name, Class<?>[] sig, Object... args) throws Exception {
        Method m = KoinaClient.class.getDeclaredMethod(name, sig);
        m.setAccessible(true);
        return (T) m.invoke(null, args);
    }

    @Test
    public void buildInferBodyIncludesOnlyDeclaredInputs() throws Exception {
        // Prosit CID declares no collision_energies and no instrument_types.
        Set<String> avail = new HashSet<>(Arrays.asList("peptide_sequences", "precursor_charges"));
        String body = call("buildInferBody",
                new Class[] { List.class, List.class, List.class, List.class, Set.class },
                Arrays.asList("PEPTIDEK", "SAMPLEK"),
                Arrays.asList(2, 3),
                Arrays.asList(27.0f, 27.0f),
                Arrays.asList("QE", "QE"),
                avail);
        JSONObject root = JSON.parseObject(body);
        JSONArray inputs = root.getJSONArray("inputs");
        Set<String> sentNames = new HashSet<>();
        for (int i = 0; i < inputs.size(); i++) {
            sentNames.add(inputs.getJSONObject(i).getString("name"));
        }
        Assert.assertTrue(sentNames.contains("peptide_sequences"));
        Assert.assertTrue(sentNames.contains("precursor_charges"));
        Assert.assertFalse(sentNames.contains("collision_energies"), "CID model must not get CE");
        Assert.assertFalse(sentNames.contains("instrument_types"));
    }

    @Test
    public void buildInferBodyAddsCeAndInstrumentWhenDeclared() throws Exception {
        // AlphaPepDeep declares CE + instrument_types.
        Set<String> avail = new HashSet<>(Arrays.asList(
                "peptide_sequences", "precursor_charges", "collision_energies", "instrument_types"));
        String body = call("buildInferBody",
                new Class[] { List.class, List.class, List.class, List.class, Set.class },
                Arrays.asList("PEPTIDEK"),
                Arrays.asList(2),
                Arrays.asList(28.0f),
                Arrays.asList("LUMOS"),
                avail);
        JSONObject root = JSON.parseObject(body);
        JSONArray inputs = root.getJSONArray("inputs");
        Set<String> sentNames = new HashSet<>();
        for (int i = 0; i < inputs.size(); i++) {
            sentNames.add(inputs.getJSONObject(i).getString("name"));
        }
        Assert.assertTrue(sentNames.contains("collision_energies"));
        Assert.assertTrue(sentNames.contains("instrument_types"));
    }

    @Test
    public void parseMs2ResponseReshapesByRow() throws Exception {
        // n=2, f=3 flattened row-major.
        String json = "{\"outputs\":["
                + "{\"name\":\"intensities\",\"datatype\":\"FP32\",\"shape\":[2,3],\"data\":[0.1,0.2,0.3,0.4,0.5,0.6]},"
                + "{\"name\":\"mz\",\"datatype\":\"FP32\",\"shape\":[2,3],\"data\":[100,200,300,110,210,310]},"
                + "{\"name\":\"annotation\",\"datatype\":\"BYTES\",\"shape\":[2,3],"
                + "\"data\":[\"y1+1\",\"b2+1\",\"y3+1\",\"y1+1\",\"b2+1\",\"y3+1\"]}]}";
        List<KoinaClient.Ms2> res = call("parseMs2Response",
                new Class[] { String.class, int.class }, json, 2);
        Assert.assertEquals(res.size(), 2);
        Assert.assertEquals(res.get(0).intensity.length, 3);
        Assert.assertEquals(res.get(0).intensity[2], 0.3f, 1e-6);
        Assert.assertEquals(res.get(1).mz[0], 110f, 1e-6);
        Assert.assertEquals(res.get(1).annotation[2], "y3+1");
    }

    @Test
    public void parseRtResponseReadsIrt() throws Exception {
        String json = "{\"outputs\":[{\"name\":\"irt\",\"datatype\":\"FP32\",\"shape\":[3,1],"
                + "\"data\":[12.5,33.1,-5.0]}]}";
        float[] irt = call("parseRtResponse", new Class[] { String.class }, json);
        Assert.assertEquals(irt.length, 3);
        Assert.assertEquals(irt[1], 33.1f, 1e-5);
        Assert.assertEquals(irt[2], -5.0f, 1e-6);
    }

    @Test
    public void parseInputNamesReadsModelMetadata() throws Exception {
        String meta = "{\"name\":\"Prosit_2020_intensity_HCD\",\"inputs\":["
                + "{\"name\":\"peptide_sequences\",\"datatype\":\"BYTES\"},"
                + "{\"name\":\"precursor_charges\",\"datatype\":\"INT32\"},"
                + "{\"name\":\"collision_energies\",\"datatype\":\"FP32\"}]}";
        Set<String> names = call("parseInputNames", new Class[] { String.class }, meta);
        Assert.assertTrue(names.contains("peptide_sequences"));
        Assert.assertTrue(names.contains("collision_energies"));
        Assert.assertEquals(names.size(), 3);
    }
}
