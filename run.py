from loader import BasicLoader
from trainer import BinaryTuner

study = BasicLoader('Dataset_Pregnancy_Outcomes.csv', csv_separator=';',
    target=['Class_GDM'],
    continuous=['Age (years)', 'BMI 1T (Kg/m2)', 'BMI 2T (Kg/m2)', 'Newborn weight (g)','Newborn height (cm)'],
    discrete=['Newborn sex (F=1, M=2)'])

# study.save_datasets('dataset_preprocesado.xlsx')
# study.save_univariate_analysis('analisis_univariado.xlsx')
# study.plot_all_pca()
# study.plot_all_heatmaps()
# study.plot_all_distributions()
# study.plot_all_boxplots()
#
trial_df = study.get_dataset('Class_GDM', remap_smaller_is_zero=True)

trial = BinaryTuner(trial_df, 'Class_GDM', n_seeds=10)

trial.fit()
