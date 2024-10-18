package main.java.ai;

import com.compomics.util.experiment.biology.ions.Ion;
import com.compomics.util.experiment.biology.ions.IonFactory;
import com.compomics.util.experiment.biology.ions.NeutralLoss;
import com.compomics.util.experiment.biology.ions.impl.PeptideFragmentIon;
import com.compomics.util.experiment.biology.proteins.Peptide;
import com.compomics.util.experiment.identification.spectrum_annotation.AnnotationParameters;
import com.compomics.util.experiment.identification.spectrum_annotation.SpecificAnnotationParameters;
import com.compomics.util.experiment.identification.spectrum_annotation.SpectrumAnnotator;
import com.compomics.util.experiment.identification.spectrum_annotation.spectrum_annotators.PeptideSpectrumAnnotator;
import com.compomics.util.parameters.identification.advanced.SequenceMatchingParameters;
import com.compomics.util.parameters.identification.search.ModificationParameters;
import main.java.input.ModificationUtils;
import main.java.input.CParameter;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

import static main.java.ai.AIGear.getNeutralLossesMap;

class PeptideFrag {

    public static boolean lossWaterNH3 = true;
    public static int max_fragment_ion_charge = 2;

    /**
     * If true, only fragment ions with charge less than precursor charge will be considered.
     * It means that if precursor charge is 2, only fragment ions with charge 1+ will be considered.
     */
    public static boolean fragment_ion_charge_less_than_precursor_charge = false;
    PeptideSpectrumAnnotator peptideSpectrumAnnotator = new PeptideSpectrumAnnotator();
    SpecificAnnotationParameters specificAnnotationPreferences = new SpecificAnnotationParameters();
    ModificationParameters modificationParameters = new ModificationParameters();
    SequenceMatchingParameters sequenceMatchingParameters = new SequenceMatchingParameters();
    JSequenceProvider jSequenceProvider = new JSequenceProvider();

    public void init(int precursorCharge, Peptide objPeptide, String mod_ai){

        specificAnnotationPreferences.setSelectedCharges(getPossibleFragmentIonCharges(precursorCharge));
        specificAnnotationPreferences.addIonType(Ion.IonType.PEPTIDE_FRAGMENT_ION, PeptideFragmentIon.B_ION);
        specificAnnotationPreferences.addIonType(Ion.IonType.PEPTIDE_FRAGMENT_ION, PeptideFragmentIon.Y_ION);
        specificAnnotationPreferences.setFragmentIonAccuracy(CParameter.itol);
        specificAnnotationPreferences.setFragmentIonPpm(CParameter.itolu.startsWith("ppm"));
        specificAnnotationPreferences.setNeutralLossesAuto(false);
        specificAnnotationPreferences.clearNeutralLosses();
        specificAnnotationPreferences.setPrecursorCharge(precursorCharge);

        if(lossWaterNH3) {
            specificAnnotationPreferences.addNeutralLoss(NeutralLoss.H2O);
            specificAnnotationPreferences.addNeutralLoss(NeutralLoss.NH3);
        }

        AnnotationParameters annotationSettings = new AnnotationParameters();
        annotationSettings.setTiesResolution(SpectrumAnnotator.TiesResolution.mostIntense);
        annotationSettings.setFragmentIonAccuracy(CParameter.itol);
        annotationSettings.setFragmentIonPpm(CParameter.itolu.startsWith("ppm"));
        annotationSettings.setFragmentIonPpm(false);
        annotationSettings.setNeutralLossesSequenceAuto(false);

        annotationSettings.setIntensityThresholdType(AnnotationParameters.IntensityThresholdType.percentile);

        if(lossWaterNH3) {
            annotationSettings.addNeutralLoss(NeutralLoss.H2O);
            annotationSettings.addNeutralLoss(NeutralLoss.NH3);
        }

        // consider neutral loss of phosphorylation.
        if(mod_ai.equals("phosphorylation")) {
            if (ModificationUtils.getInstance().getModificationString(objPeptide).toLowerCase().contains("phosphorylation")) {
                specificAnnotationPreferences.setNeutralLossesMap(getNeutralLossesMap(objPeptide));
            }
        }
    }

    private HashSet<Integer> getPossibleFragmentIonCharges(int precursorCharge) {
        HashSet<Integer> charges = new HashSet<>(4);
        if (precursorCharge <= 1) {
            charges.add(precursorCharge);
        } else {
            int cur_max_fragment_ion_charge = Math.min(precursorCharge, max_fragment_ion_charge);
            if(fragment_ion_charge_less_than_precursor_charge) {
                if (precursorCharge == cur_max_fragment_ion_charge) {
                    cur_max_fragment_ion_charge = cur_max_fragment_ion_charge - 1;
                }
            }
            for (int c = 1; c <= cur_max_fragment_ion_charge; c++) {
                charges.add(c);
            }
        }
        return charges;
    }

    public HashMap<Integer, ArrayList<Ion>> getExpectedFragIons(Peptide peptide){
        IonFactory fragmentFactory = IonFactory.getInstance();
        // HashMap<Integer, HashMap<Integer, ArrayList<Ion>>> ions = fragmentFactory.getFragmentIons(peptide, specificAnnotationPreferences, modificationParameters, jSequenceProvider, sequenceMatchingParameters);
        HashMap<Integer,ArrayList<Ion>> ions = peptideSpectrumAnnotator.getExpectedIons(specificAnnotationPreferences,peptide, modificationParameters, jSequenceProvider, sequenceMatchingParameters);
        return ions;
    }

}
