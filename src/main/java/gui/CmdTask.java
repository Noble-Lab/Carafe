package main.java.gui;

import java.util.ArrayList;
import java.util.List;

public class CmdTask {
    public String cmd;
    public List<String> args = new ArrayList<>();
    public String task_name;
    public String task_description;
    public String time_start;
    public String time_end;
    public double time_used;
    public String out_dir;
    public List<String> out_files = new ArrayList<>();
    public List<String> out_files_description = new ArrayList<>();

    /**
     * Files this step READS (e.g. MS data, protein database, prior-step library/blib). Used to
     * compute the reuse signature so a step is only skipped when its inputs are unchanged.
     */
    public List<String> input_files = new ArrayList<>();

    /**
     * Primary output file used to decide whether this step can be skipped (reused).
     * When "Reuse existing results" is enabled and this file already exists, the step
     * is skipped instead of re-running. Null means the step is never auto-skipped.
     */
    public String skip_check_file = null;
    /**
     * Set to true when the step was skipped because its result was already present.
     */
    public boolean skipped = false;

    /**
     * Optional action run AFTER the step's process exits 0 and BEFORE its reuse signature
     * is written — e.g. move a locally-staged output to its final (possibly network) path.
     * Throwing aborts the workflow as a step failure. Never run when the step is skipped.
     */
    public ThrowingRunnable postAction = null;

    /** A {@link Runnable} that is allowed to throw a checked exception. */
    @FunctionalInterface
    public interface ThrowingRunnable {
        void run() throws Exception;
    }

    public CmdTask(String cmd, String task_name, String task_description) {
        this.cmd = cmd;
        this.task_name = task_name;
        this.task_description = task_description;
    }

    public CmdTask(List<String> args, String task_name, String task_description) {
        this.args = args;
        this.task_name = task_name;
        this.task_description = task_description;
    }

}
