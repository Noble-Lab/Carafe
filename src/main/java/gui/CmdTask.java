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
