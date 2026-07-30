from loader import BasicLoader
from trainer import BinaryTuner

T3_pred_continuo = ['Age (years)',
 'BMI 1T (Kg/m2)',
 'BMI 2T (Kg/m2)',
 'BMI 3T (Kg/m2)',
 'Basal Glycaemia 1T (mg/dl)',
 'Basal Glycaemia 2T (mg/dl)',
 'Glycaemia 2h post 75 g (2T) (mg/dl)',
 'FT4 1T (ng/dl)',
 'FT4 2T (ng/dl)',
 'FT4 3T (ng/dl)',
 'TSH 1T (mIU/L)',
 'TSH 2T (mIU/L)',
 'TSH 3T (mIU/L)',
 'TT4 1T (ug/ml)',
 'TT4 2T (ug/ml)',
 'TT4 3T (ug/ml)',
 'TT3 1T (ng/ml)',
 'TT3 2T (ng/ml)',
 'TT3 3T (ng/ml)']

T2_pred_continuo = ['Age (years)',
 'BMI 1T (Kg/m2)',
 'BMI 2T (Kg/m2)',
 'Basal Glycaemia 1T (mg/dl)',
 'Basal Glycaemia 2T (mg/dl)',
 'Glycaemia 2h post 75 g (2T) (mg/dl)',
 'FT4 1T (ng/dl)',
 'FT4 2T (ng/dl)',
 'TSH 1T (mIU/L)',
 'TSH 2T (mIU/L)',
 'TT4 1T (ug/ml)',
 'TT4 2T (ug/ml)',
 'TT3 1T (ng/ml)',
 'TT3 2T (ng/ml)']


T1_pred_continuo = ['Age (years)',
 'BMI 1T (Kg/m2)',
 'Basal Glycaemia 1T (mg/dl)',
 'FT4 1T (ng/dl)',
 'TSH 1T (mIU/L)',
 'TT4 1T (ug/ml)',
 'TT3 1T (ng/ml)']

PTB_noGDM = BasicLoader('Dataset_Pregnancy_Outcomes_binary.csv',
    target=['Class_PTB'],
    continuous=T3_pred_continuo,
    exclude={"Class_GDM": [1], })
PTB_noGDM.set_label('Class_PTB', 'PTB noGDM T3')

PTB_GDM = BasicLoader('Dataset_Pregnancy_Outcomes_binary.csv',
    target=['Class_PTB'],
    continuous=T3_pred_continuo,
    exclude={"Class_GDM": [0], })
PTB_GDM.set_label('Class_PTB', 'PTB GDM T3')

NBM_GDM = BasicLoader('Dataset_Pregnancy_Outcomes_binary.csv',
    target=['Class_Macrosomia'],
    continuous=T2_pred_continuo,
    exclude={"Class_GDM": [0], })
NBM_GDM.set_label('Class_Macrosomia', 'NBM GDM T2')

PTB_T3 = BasicLoader('Dataset_Pregnancy_Outcomes_binary.csv',
    target=['Class_PTB'],
    continuous=T3_pred_continuo)
PTB_T3.set_label('Class_PTB', 'PTB T3')

NMB_T3 = BasicLoader('Dataset_Pregnancy_Outcomes_binary.csv',
    target=['Class_Macrosomia'],
    continuous=T3_pred_continuo)
NMB_T3.set_label('Class_Macrosomia', 'NBM T3')


PTB_T2 = BasicLoader('Dataset_Pregnancy_Outcomes_binary.csv',
    target=['Class_PTB'],
    continuous=T2_pred_continuo)
PTB_T2.set_label('Class_PTB', 'PTB T2')

NMB_T2 = BasicLoader('Dataset_Pregnancy_Outcomes_binary.csv',
    target=['Class_Macrosomia'],
    continuous=T2_pred_continuo)
NMB_T2.set_label('Class_Macrosomia', 'NBM T2')


PTB_T1 = BasicLoader('Dataset_Pregnancy_Outcomes_binary.csv',
    target=['Class_PTB'],
    continuous=T1_pred_continuo)
PTB_T1.set_label('Class_PTB', 'PTB T1')

NMB_T1 = BasicLoader('Dataset_Pregnancy_Outcomes_binary.csv',
    target=['Class_Macrosomia'],
    continuous=T1_pred_continuo)
NMB_T1.set_label('Class_Macrosomia', 'NBM T1')


studies = [ ('PTB noGDM T3', PTB_noGDM), ('PTB GDM T3', PTB_GDM), ('NBM GDM T2', NBM_GDM), ('PTB T3', PTB_T3), ('NBM T3', NMB_T3), ('PTB T2', PTB_T2), ('NBM T2', NMB_T2), ('PTB T1', PTB_T1), ('NBM T1', NMB_T1) ]

for target, study in studies:
    study_df = study.get_dataset(target)
    trial = BinaryTuner(study_df, target, n_seeds=25, tuneScoring='roc_auc')
    trial.fit()
    trial.wrap_and_save()
