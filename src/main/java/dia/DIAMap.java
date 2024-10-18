package main.java.dia;

import java.util.*;

public class DIAMap {

    public DIAMeta meta = new DIAMeta();
    public HashSet<String> target_isolation_wins = new HashSet<>();
    public DIAMap(){

    }

    public String get_isolation_window(double mz){
        String isoWin = "";
        // boolean found = false;
        for(String id: this.target_isolation_wins){
            if(mz >= this.meta.isolationWindowMap.get(id).mz_lower && mz <= this.meta.isolationWindowMap.get(id).mz_upper){
                isoWin = id;
                // found = true;
                break;
            }
        }
        return isoWin;
    }

    public ArrayList<String> get_isolation_windows(double mz){
        ArrayList<String> isoWins = new ArrayList<>();
        // boolean found = false;
        for(String id: this.target_isolation_wins){
            if(mz >= this.meta.isolationWindowMap.get(id).mz_lower && mz <= this.meta.isolationWindowMap.get(id).mz_upper){
                // found = true;
                isoWins.add(id);
            }
        }
        return isoWins;
    }
}
