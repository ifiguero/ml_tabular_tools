from loader import BasicLoader
from trainer import BinaryTuner

columns = ['ID',
 'Class_GDM (1=no, 2=yes)',
 'Class_PTB (1=no, 2=yes)',
 'Class_Macrosomia (1=no, 2=yes)',
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

full_cont = ['Age (years)',
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
 'Newborn TSH (mIU/L)',
 'Newborn FT4 (ng/dl)',
 'Newborn TT4 (ug/ml)',
 'Newborn TT3 (ng/ml)']

noT3_cont = ['Age (years)',
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
 'TT3 2T (ng/ml)',
 'Gestational age at delivery (weeks)',
 'Newborn weight (g)',
 'Newborn height (cm)',
 'Newborn TSH (mIU/L)',
 'Newborn FT4 (ng/dl)',
 'Newborn TT4 (ug/ml)']


sinGDMfull = BasicLoader('Dataset_Pregnancy_Outcomes.csv', csv_separator=';',
    target=['Class_PTB (1=no, 2=yes)'],
    continuous=full_cont, discrete=full_discrete,
    exclude={"Class_GDM (1=no, 2=yes)": [2], })

sinGDMfull.set_label('Class_PTB (1=no, 2=yes)', 'PTB noGDM Full')

conGDMfull = BasicLoader('Dataset_Pregnancy_Outcomes.csv', csv_separator=';',
    target=['Class_PTB (1=no, 2=yes)'],
    continuous=full_cont, discrete=full_discrete,
    exclude={"Class_GDM (1=no, 2=yes)": [1], })

conGDMfull.set_label('Class_PTB (1=no, 2=yes)', 'PTB GDM Full')

conGDMnoT3 = BasicLoader('Dataset_Pregnancy_Outcomes.csv', csv_separator=';',
    target=['Class_Macrosomia (1=no, 2=yes)'],
    continuous=noT3_cont, discrete=full_discrete,
    exclude={"Class_GDM (1=no, 2=yes)": [1], })

conGDMnoT3.set_label('Class_Macrosomia (1=no, 2=yes)', 'NBM GDM noT3')

todosFull = BasicLoader('Dataset_Pregnancy_Outcomes.csv', csv_separator=';',
    target=['Class_PTB (1=no, 2=yes)'],
    continuous=full_cont, discrete=full_discrete)

todosFull.set_label('Class_PTB (1=no, 2=yes)', 'PTB Todos Full')

todosNoT3 = BasicLoader('Dataset_Pregnancy_Outcomes.csv', csv_separator=';',
    target=['Class_Macrosomia (1=no, 2=yes)'],
    continuous=noT3_cont, discrete=full_discrete)

todosNoT3.set_label('Class_Macrosomia (1=no, 2=yes)', 'NBM Todos noT3')

studies = [
    ("PTB noGDM Full", sinGDMfull),
    ("PTB GDM Full", conGDMfull),
    ("NBM GDM noT3", conGDMnoT3),
    ("PTB Todos Full", todosFull),
    ("NBM Todos noT3", todosNoT3),
]

for target, study in studies:

    study.save_datasets(f'dataset_preprocesado_{target}.xlsx')
    study.save_univariate_analysis(f'analisis_univariado_{target}.xlsx')
    study.plot_all_pca()
    study.plot_all_heatmaps()
    study.plot_all_distributions()
    study.plot_all_boxplots()
#
