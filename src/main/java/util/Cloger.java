package main.java.util;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class Cloger {

	private static Cloger instance = null;
	public Logger logger;
	private long job_start_time = 0;

	private Cloger(){
		logger = LogManager.getLogger(Cloger.class.getName());
	}

	public static Cloger getInstance() {
		if (instance == null) {
			instance = new Cloger();
		}
		return instance;
	}

	public void set_job_start_time(){
		job_start_time = System.currentTimeMillis();
	}

	public String get_job_run_time(){
		long ctime = System.currentTimeMillis();
		double t = 1.0*(ctime  - job_start_time)/1000.0/60.0;
        return(String.format("%.2f",t) + " min");
	}

}


