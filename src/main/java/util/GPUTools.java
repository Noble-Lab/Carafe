package main.java.util;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class GPUTools {

    public String py_path = "python";

    public static class TorchGpuStatus {
        public boolean gpuAvailable = false;
        public boolean cudaAllocationOk;
        public long gpu_memory = -1;     // -1 if no CUDA
        public String torchVersion;
        public String cudaVersion;
        public String deviceName;
        public String rawOutput;

        public TorchGpuStatus(boolean gpuAvailable,
                              boolean cudaAllocationOk,
                              long gpu_memory,
                              String torchVersion,
                              String cudaVersion,
                              String deviceName,
                              String rawOutput) {
            this.gpuAvailable = gpuAvailable;
            this.cudaAllocationOk = cudaAllocationOk;
            this.gpu_memory = gpu_memory;
            this.torchVersion = torchVersion;
            this.cudaVersion = cudaVersion;
            this.deviceName = deviceName;
            this.rawOutput = rawOutput;
        }

        public TorchGpuStatus() {

        }

        @Override
        public String toString() {
            return "TorchGpuStatus{" +
                    "gpuAvailable=" + gpuAvailable +
                    ", gpu_memory=" + gpu_memory +
                    ", cudaAllocationOk=" + cudaAllocationOk +
                    ", torchVersion='" + torchVersion + '\'' +
                    ", cudaVersion='" + cudaVersion + '\'' +
                    ", deviceName='" + deviceName + '\'' +
                    '}';
        }
    }



    /**
     * Checks whether PyTorch can use CUDA GPU by calling Python from system PATH:
     *   python -c "<script>"
     *
     * It tries these commands in order:
     *   1) python
     *   2) py -3   (Windows launcher fallback)
     *   3) python3
     *
     * Returns a parsed TorchGpuStatus. Throws if no python candidate works or torch cannot be imported.
     */
    public TorchGpuStatus checkTorchGpu(){
        // Machine-parseable output + stronger check via CUDA allocation.
        String py = "import torch\n" +
                        "gpu = bool(torch.cuda.is_available())\n" +
                        "print('GPU_AVAILABLE=' + str(gpu))\n" +
                        "print('TORCH_VERSION=' + str(torch.__version__))\n" +
                        "print('TORCH_CUDA=' + str(torch.version.cuda))\n" +
                        "name=''\n" +
                        "gpu_memory=-1\n" +
                        "try:\n" +
                        "    if gpu:\n" +
                        "        name = torch.cuda.get_device_name(0)\n" +
                        "        p = torch.cuda.get_device_properties(0)\n" +
                        "        gpu_memory = int(p.total_memory)\n" +
                        "except Exception:\n" +
                        "    name=''\n" +
                        "print('DEVICE_NAME=' + name)\n" +
                        "print('GPU_MEMORY=' + str(gpu_memory))\n" +
                        "alloc_ok=False\n" +
                        "try:\n" +
                        "    if gpu:\n" +
                        "        x = torch.tensor([1.0], device='cuda')\n" +
                        "        alloc_ok = (x.device.type == 'cuda')\n" +
                        "except Exception:\n" +
                        "    alloc_ok=False\n" +
                        "print('CUDA_ALLOC_OK=' + str(alloc_ok))\n";

        try {
            String output = runAndCapture(List.of(py_path, "-c", py));
            return parseTorchGpuStatus(output);
        } catch (Exception e) {
            return new TorchGpuStatus();
        }
    }

    private static String runAndCapture(List<String> cmd) throws IOException, InterruptedException {
        ProcessBuilder pb = new ProcessBuilder(cmd);
        pb.redirectErrorStream(true);

        Process p = pb.start();

        String output;
        try (InputStream is = p.getInputStream()) {
            output = new String(is.readAllBytes(), StandardCharsets.UTF_8);
        }

        int code = p.waitFor();
        if (code != 0) {
            throw new RuntimeException("Command failed (" + code + "): " + String.join(" ", cmd) + "\nOutput:\n" + output);
        }
        return output;
    }

    private TorchGpuStatus parseTorchGpuStatus(String output) {
        Map<String, String> kv = new HashMap<>();
        for (String line : output.split("\\R")) {
            int idx = line.indexOf('=');
            if (idx > 0) {
                String k = line.substring(0, idx).trim();
                String v = line.substring(idx + 1).trim();
                kv.put(k, v);
            }
        }

        boolean gpu = Boolean.parseBoolean(kv.getOrDefault("GPU_AVAILABLE", "false"));
        boolean allocOk = Boolean.parseBoolean(kv.getOrDefault("CUDA_ALLOC_OK", "false"));
        String torchVer = kv.getOrDefault("TORCH_VERSION", "");
        String cudaVer = kv.getOrDefault("TORCH_CUDA", "");
        String devName = kv.getOrDefault("DEVICE_NAME", "");
        long gpu_memory = Long.parseLong(kv.getOrDefault("GPU_MEMORY", "-1"));

        return new TorchGpuStatus(gpu, allocOk, gpu_memory, torchVer, cudaVer, devName, output);
    }

    public static void main(String[] args) {
        try {
            GPUTools tools = new GPUTools();
            TorchGpuStatus st = tools.checkTorchGpu();
            System.out.println(st);
            System.out.println("\n--- Raw output ---\n" + st.rawOutput);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
