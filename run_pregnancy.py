from loader import BasicLoader
from trainer import BinaryTuner

columns = ['ID',
 'Class_GDM',
 'Class_PTB',
 'Class_Macrosomia',
 'Age (years)',
 'BMI 1T (Kg/m2)',
 'BMI 2T (Kg/m2)',
 'BMI 3T (Kg/m2)',
 'Basal Glycaemia 1T (mg/dl)',
 'Basal Glycaemia 2T (mg/dl)',
 'Glycaemia 2h post 75 g (2T) (mg/dl)',
 'TSH 1T (mIU/L)',
 'FT4 1T (ng/dl)',
 'TT4 1T (ug/ml)',
 'TT3 1T (ng/ml)',
 'TSH 2T (mIU/L)',
 'FT4 2T (ng/dl)',
 'TT4 2T (ug/ml)',
 'TT3 2T (ng/ml)',
 'TSH 3T (mIU/L)',
 'FT4 3T (ng/dl)',
 'TT4 3T (ug/ml)',
 'TT3 3T (ng/ml)',
 'Gestational age at delivery (weeks)',
 'Newborn weight (g)',
 'Newborn height (cm)',
 'Newborn sex (F=1, M=2)',
 'Newborn TSH (mIU/L)',
 'Newborn FT4 (ng/dl)',
 'Newborn TT4 (ug/ml)',
 'Newborn TT3 (ng/ml)']

full_discrete=['Newborn sex (F=1, M=2)']

pred_continuo = ['Age (years)',
 'BMI 1T (Kg/m2)',
 'BMI 2T (Kg/m2)',
 'BMI 3T (Kg/m2)',
 'Basal Glycaemia 1T (mg/dl)',
 'Basal Glycaemia 2T (mg/dl)',
 'Glycaemia 2h post 75 g (2T) (mg/dl)',
 'TSH 1T (mIU/L)',
 'FT4 1T (ng/dl)',
 'TT4 1T (ug/ml)',
 'TT3 1T (ng/ml)',
 'TSH 2T (mIU/L)',
 'FT4 2T (ng/dl)',
 'TT4 2T (ug/ml)',
 'TT3 2T (ng/ml)',
 'TSH 3T (mIU/L)',
 'FT4 3T (ng/dl)',
 'TT4 3T (ug/ml)',
 'TT3 3T (ng/ml)']

noT3_pred_continuo = ['Age (years)',
 'BMI 1T (Kg/m2)',
 'BMI 2T (Kg/m2)',
 'Basal Glycaemia 1T (mg/dl)',
 'Basal Glycaemia 2T (mg/dl)',
 'Glycaemia 2h post 75 g (2T) (mg/dl)',
 'TSH 1T (mIU/L)',
 'FT4 1T (ng/dl)',
 'TT4 1T (ug/ml)',
 'TT3 1T (ng/ml)',
 'TSH 2T (mIU/L)',
 'FT4 2T (ng/dl)',
 'TT4 2T (ug/ml)',
 'TT3 2T (ng/ml)']


PTB_noGDM = BasicLoader('Dataset_Pregnancy_Outcomes_binary.csv',
    target=['Class_PTB'],
    continuous=pred_continuo, discrete=full_discrete,
    exclude={"Class_GDM": [1], })
PTB_noGDM.set_label('Class_PTB', 'PTB noGDM')

PTB_GDM = BasicLoader('Dataset_Pregnancy_Outcomes_binary.csv',
    target=['Class_PTB'],
    continuous=pred_continuo, discrete=full_discrete,
    exclude={"Class_GDM": [0], })
PTB_GDM.set_label('Class_PTB', 'PTB GDM')

NBM_GDM = BasicLoader('Dataset_Pregnancy_Outcomes_binary.csv',
    target=['Class_Macrosomia'],
    continuous=noT3_pred_continuo, discrete=full_discrete,
    exclude={"Class_GDM": [0], })
NBM_GDM.set_label('Class_Macrosomia', 'NBM GDM')

PTB_complete = BasicLoader('Dataset_Pregnancy_Outcomes_binary.csv',
    target=['Class_PTB'],
    continuous=pred_continuo, discrete=full_discrete)
PTB_complete.set_label('Class_PTB', 'PTB Complete')

NMB_complete = BasicLoader('Dataset_Pregnancy_Outcomes_binary.csv',
    target=['Class_Macrosomia'],
    continuous=noT3_pred_continuo, discrete=full_discrete)
NMB_complete.set_label('Class_Macrosomia', 'NBM Complete')

studies = [ ('PTB noGDM', PTB_noGDM), ('PTB GDM', PTB_GDM), ('NBM GDM', NBM_GDM), ('PTB Complete', PTB_complete), ('NBM Complete', NMB_complete) ]

for target, study in studies:
    study_df = study.get_dataset(target)
    trial = BinaryTuner(study_df, target, n_seeds=15, tuneScoring='roc_auc')
    trial.fit()
    trial.wrap_and_save()
