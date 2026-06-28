package main.java.koina;

import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Minimal client for the <a href="https://koina.wilhelmlab.org">Koina</a> prediction service,
 * which hosts AlphaPepDeep / Prosit / ms2pip fragment-intensity models and Prosit / AlphaPepDeep
 * retention-time models behind the KServe/Triton v2 inference protocol.
 *
 * <p>Carafe uses Koina to generate the <em>initial</em> (non-finetuned) spectral library for the
 * Osprey workflows, as an alternative to running the local AlphaPepDeep model. Different
 * models declare different inputs (e.g. Prosit CID has no {@code collision_energies};
 * AlphaPepDeep adds {@code instrument_types}), so callers pass the set of inputs the model
 * declares (from {@link #getModelInputNames}) and this client sends only those.</p>
 *
 * <p>The JSON request/response building and parsing are factored into static methods so they can
 * be unit-tested without network access.</p>
 */
public class KoinaClient {

    /** Per-precursor MS2 prediction: parallel fragment annotation / m/z / intensity arrays. */
    public static final class Ms2 {
        public final String[] annotation;
        public final float[] mz;
        public final float[] intensity;

        public Ms2(String[] annotation, float[] mz, float[] intensity) {
            this.annotation = annotation;
            this.mz = mz;
            this.intensity = intensity;
        }
    }

    private final String baseUrl;
    private final HttpClient http;

    public KoinaClient(String baseUrl) {
        String b = baseUrl == null || baseUrl.isBlank() ? "https://koina.wilhelmlab.org" : baseUrl.trim();
        while (b.endsWith("/")) {
            b = b.substring(0, b.length() - 1);
        }
        this.baseUrl = b;
        this.http = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(30))
                .build();
    }

    /** Fetch a model's declared input names from {@code /v2/models/<model>}. */
    public Set<String> getModelInputNames(String model) throws IOException, InterruptedException {
        String body = httpGet(baseUrl + "/v2/models/" + model);
        return parseInputNames(body);
    }

    static Set<String> parseInputNames(String metadataJson) {
        Set<String> names = new HashSet<>();
        JSONObject o = JSON.parseObject(metadataJson);
        JSONArray inputs = o.getJSONArray("inputs");
        if (inputs != null) {
            for (int i = 0; i < inputs.size(); i++) {
                names.add(inputs.getJSONObject(i).getString("name"));
            }
        }
        return names;
    }

    /**
     * Run a fragment-intensity model. {@code charges}/{@code collisionEnergies}/{@code instruments}
     * may be null and are sent only when the model declares the corresponding input.
     *
     * @return one {@link Ms2} per input row (same order as {@code sequences})
     */
    public List<Ms2> inferMs2(String model, List<String> sequences, List<Integer> charges,
            List<Float> collisionEnergies, List<String> instruments, Set<String> availableInputs)
            throws IOException, InterruptedException {
        String reqBody = buildInferBody(sequences, charges, collisionEnergies, instruments, availableInputs);
        String respBody = httpPost(baseUrl + "/v2/models/" + model + "/infer", reqBody);
        return parseMs2Response(respBody, sequences.size());
    }

    /** Run a retention-time model (only {@code peptide_sequences} input); returns one iRT per row. */
    public float[] inferRt(String model, List<String> sequences, Set<String> availableInputs)
            throws IOException, InterruptedException {
        String reqBody = buildInferBody(sequences, null, null, null, availableInputs);
        String respBody = httpPost(baseUrl + "/v2/models/" + model + "/infer", reqBody);
        return parseRtResponse(respBody);
    }

    // ---- request building / response parsing (static, unit-testable) ----

    /** Build a KServe v2 infer request body, including only inputs the model declares. */
    static String buildInferBody(List<String> sequences, List<Integer> charges,
            List<Float> collisionEnergies, List<String> instruments, Set<String> availableInputs) {
        int n = sequences.size();
        JSONArray inputs = new JSONArray();

        // peptide_sequences (always required)
        inputs.add(stringInput("peptide_sequences", sequences));

        if (availableInputs.contains("precursor_charges") && charges != null) {
            JSONObject in = new JSONObject();
            in.put("name", "precursor_charges");
            in.put("shape", new int[] { n, 1 });
            in.put("datatype", "INT32");
            in.put("data", new JSONArray(new ArrayList<Object>(charges)));
            inputs.add(in);
        }
        if (availableInputs.contains("collision_energies") && collisionEnergies != null) {
            JSONObject in = new JSONObject();
            in.put("name", "collision_energies");
            in.put("shape", new int[] { n, 1 });
            in.put("datatype", "FP32");
            in.put("data", new JSONArray(new ArrayList<Object>(collisionEnergies)));
            inputs.add(in);
        }
        if (availableInputs.contains("instrument_types") && instruments != null) {
            inputs.add(stringInput("instrument_types", instruments));
        }

        JSONObject root = new JSONObject();
        root.put("id", "0");
        root.put("inputs", inputs);
        return JSON.toJSONString(root);
    }

    private static JSONObject stringInput(String name, List<String> data) {
        JSONObject in = new JSONObject();
        in.put("name", name);
        in.put("shape", new int[] { data.size(), 1 });
        in.put("datatype", "BYTES");
        in.put("data", new JSONArray(new ArrayList<Object>(data)));
        return in;
    }

    /** Parse intensities/mz/annotation outputs into {@code n} per-precursor {@link Ms2} records. */
    static List<Ms2> parseMs2Response(String json, int n) {
        JSONObject root = JSON.parseObject(json);
        float[] intensities = floatOutput(root, "intensities");
        float[] mz = floatOutput(root, "mz");
        String[] annotation = stringOutput(root, "annotation");
        if (intensities == null || mz == null || annotation == null) {
            throw new IllegalStateException("Koina MS2 response missing intensities/mz/annotation: "
                    + truncate(json));
        }
        int f = intensities.length / n;
        List<Ms2> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            String[] a = new String[f];
            float[] m = new float[f];
            float[] it = new float[f];
            for (int j = 0; j < f; j++) {
                int k = i * f + j;
                a[j] = annotation[k];
                m[j] = mz[k];
                it[j] = intensities[k];
            }
            out.add(new Ms2(a, m, it));
        }
        return out;
    }

    /** Parse the {@code irt} output into one value per input row. */
    static float[] parseRtResponse(String json) {
        JSONObject root = JSON.parseObject(json);
        float[] irt = floatOutput(root, "irt");
        if (irt == null) {
            // Some RT models name the output differently; fall back to the first FP32 output.
            JSONArray outputs = root.getJSONArray("outputs");
            if (outputs != null && !outputs.isEmpty()) {
                irt = toFloatArray(outputs.getJSONObject(0).getJSONArray("data"));
            }
        }
        if (irt == null) {
            throw new IllegalStateException("Koina RT response missing irt output: " + truncate(json));
        }
        return irt;
    }

    private static float[] floatOutput(JSONObject root, String name) {
        JSONArray data = outputData(root, name);
        return data == null ? null : toFloatArray(data);
    }

    private static String[] stringOutput(JSONObject root, String name) {
        JSONArray data = outputData(root, name);
        if (data == null) {
            return null;
        }
        String[] out = new String[data.size()];
        for (int i = 0; i < data.size(); i++) {
            out[i] = String.valueOf(data.get(i));
        }
        return out;
    }

    private static JSONArray outputData(JSONObject root, String name) {
        JSONArray outputs = root.getJSONArray("outputs");
        if (outputs == null) {
            return null;
        }
        for (int i = 0; i < outputs.size(); i++) {
            JSONObject o = outputs.getJSONObject(i);
            if (name.equals(o.getString("name"))) {
                return o.getJSONArray("data");
            }
        }
        return null;
    }

    private static float[] toFloatArray(JSONArray data) {
        float[] out = new float[data.size()];
        for (int i = 0; i < data.size(); i++) {
            out[i] = data.getFloatValue(i);
        }
        return out;
    }

    private static String truncate(String s) {
        return s.length() > 300 ? s.substring(0, 300) + "..." : s;
    }

    // ---- HTTP (with one retry on transient failures) ----

    private String httpGet(String url) throws IOException, InterruptedException {
        HttpRequest req = HttpRequest.newBuilder(URI.create(url))
                .timeout(Duration.ofSeconds(60))
                .GET()
                .build();
        return send(req);
    }

    private String httpPost(String url, String body) throws IOException, InterruptedException {
        HttpRequest req = HttpRequest.newBuilder(URI.create(url))
                .timeout(Duration.ofMinutes(5))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(body))
                .build();
        return send(req);
    }

    private String send(HttpRequest req) throws IOException, InterruptedException {
        IOException last = null;
        for (int attempt = 0; attempt < 2; attempt++) {
            try {
                HttpResponse<String> resp = http.send(req, HttpResponse.BodyHandlers.ofString());
                if (resp.statusCode() / 100 == 2) {
                    return resp.body();
                }
                throw new IOException("Koina request to " + req.uri() + " failed: HTTP "
                        + resp.statusCode() + " " + truncate(resp.body()));
            } catch (IOException e) {
                last = e;
            }
        }
        throw last;
    }
}
