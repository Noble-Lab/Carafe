package test.java.db;

import main.java.db.DBGear;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * Tests for {@link DBGear#get_mz}, the precursor m/z calculation. It is foundational -- every
 * library precursor, isolation-window assignment and apex lookup depends on it -- and a regression
 * here would be silent, so the mass/charge relation is pinned. The proton mass is read back from
 * the method itself (mass 0, charge 1) to avoid hard-coding the exact constant.
 */
public class DBGearTest {

    @Test
    public void mzFollowsTheMassChargeRelation() {
        DBGear g = new DBGear();
        double proton = g.get_mz(0.0, 1); // (0 + 1*proton)/1
        Assert.assertTrue(proton > 1.0 && proton < 1.02, "proton mass should be ~1.007, was " + proton);
        // m/z = mass/charge + proton
        Assert.assertEquals(g.get_mz(1000.0, 2), 1000.0 / 2 + proton, 1e-9);
        Assert.assertEquals(g.get_mz(2400.0, 3), 2400.0 / 3 + proton, 1e-9);
        Assert.assertEquals(g.get_mz(927.44, 1), 927.44 + proton, 1e-9);
    }

    @Test
    public void higherChargeGivesLowerMz() {
        DBGear g = new DBGear();
        double m = 2000.0;
        Assert.assertTrue(g.get_mz(m, 3) < g.get_mz(m, 2));
        Assert.assertTrue(g.get_mz(m, 2) < g.get_mz(m, 1));
    }

    @Test
    public void neutralMassIsRecoverableFromMz() {
        DBGear g = new DBGear();
        double proton = g.get_mz(0.0, 1);
        int z = 2;
        double mz = g.get_mz(1234.56, z);
        // mass = mz*z - z*proton
        Assert.assertEquals(mz * z - z * proton, 1234.56, 1e-9);
    }
}
