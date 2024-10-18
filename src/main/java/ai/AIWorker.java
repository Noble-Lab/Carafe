package main.java.ai;

import main.java.input.CParameter;
import main.java.util.StreamLog;
import main.java.util.Cloger;
import java.io.File;
import java.io.IOException;
import static main.java.ai.AIGear.get_jar_path;
import static main.java.ai.AIGear.get_py_path;

public final  class AIWorker implements Runnable{

    public static String python_bin;
    public String model_dir;
    public String input_file;
    public String out_dir;
    public String out_prefix;
    public String device;
    public String ms_instrument;
    public double nce;
    public String ai_mode = "-";
    public static boolean fast_mode = false;

    public AIWorker(String model_dir, String input_file, String out_dir, String out_prefix, String device, String ms_instrument, double nce, String ai_mode){

        this.model_dir = model_dir;
        this.input_file = input_file;
        this.out_dir = out_dir;
        this.out_prefix = out_prefix;
        this.device = device;
        this.ms_instrument = ms_instrument;
        this.nce = nce;
        this.ai_mode = ai_mode;
    }

    @Override
    public void run() {
        Cloger.getInstance().logger.info(Thread.currentThread().getName()+": predicting "+this.input_file);
        String mode = this.ai_mode.equals("-")? "general": this.ai_mode;
        String ai_pred = get_jar_path() + File.separator + "ai_pred.py";
        File F = new File(ai_pred);
        if(!F.exists()){
            ai_pred = get_py_path("/main/java/ai/ai_pred.py","carafe_ai_pred");
        }
        String cmd = python_bin +" " + ai_pred +
                " --model_dir "+ model_dir +
                " --in_file "+ this.input_file +
                " --out_dir " + this.out_dir +
                " --out_prefix "+ this.out_prefix +
                " --device " + this.device +
                " --instrument " + this.ms_instrument +
                " --tf_type " + CParameter.tf_type +
                " --nce " + this.nce+
                " --mode " + mode;
        if(fast_mode){
            cmd += " --fast";
        }
        run_cmd(cmd);
    }

    private boolean run_cmd(String cmd){
        boolean pass = true;
        Runtime rt = Runtime.getRuntime();
        Process p;
        try {
            p = rt.exec(cmd);
        } catch (IOException e) {
            pass = false;
            throw new RuntimeException(e);
        }

        StreamLog errorLog = new StreamLog(p.getErrorStream(), Thread.currentThread().getName()+": AI => Error:", true);
        StreamLog stdLog = new StreamLog(p.getInputStream(), Thread.currentThread().getName()+": AI => Message:", true);

        errorLog.start();
        stdLog.start();

        try {
            int exitValue = p.waitFor();
            if (exitValue != 0) {
                pass = false;
                Cloger.getInstance().logger.error(Thread.currentThread().getName()+": AI error:" + exitValue);
            }
        } catch (InterruptedException e) {
            pass = false;
            throw new RuntimeException(e);
        }

        try {
            errorLog.join();
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
        try {
            stdLog.join();
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }

        return pass;
    }
}

