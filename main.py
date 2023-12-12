import streamlit as st
from joblib import load
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Prediccion", page_icon=":guardsman:", layout="centered", initial_sidebar_state="expanded")

le1_dfs = load('./tumor/le1.joblib')
le2_nhg = load('./tumor/le2.joblib')
le3_oss = load('./tumor/le3.joblib')
le4_tohs = load('./tumor/le4.joblib')
model = load('./tumor/modelo.joblib')
report_df1 = pd.read_csv('./tumor/report1.csv', sep=',', header=0, index_col=0)

le21_ms = load('./piel/le21.joblib')
le22_pg = load('./piel/le22.joblib')
le23_pts = load('./piel/le23.joblib')
le24_st = load('./piel/le24.joblib')
le25_ss = load('./piel/le25.joblib')
le26_tss = load('./piel/le26.joblib')
le27_tipo = load('./piel/le27tipo.joblib')
model2 = load('./piel/modelo2.joblib')
report_df2 = pd.read_csv('./piel/report2.csv', sep=',', header=0, index_col=0)

le31 = load('./brest cancer/le31.joblib')
model3 = load('./brest cancer/modelo3.joblib')
report_df3 = pd.read_csv('./brest cancer/report3.csv', sep=',', header=0, index_col=0)

le4_01 = load('./bladder cancer/le4_01.joblib')
le4_02 = load('./bladder cancer/le4_02.joblib')
le4_03 = load('./bladder cancer/le4_03.joblib')
le4_04 = load('./bladder cancer/le4_04.joblib')
le4_05 = load('./bladder cancer/le4_05.joblib')
le4_06_oss = load('./bladder cancer/le4_06_oss.joblib')
le4_07 = load('./bladder cancer/le4_07.joblib')
le4_08 = load('./bladder cancer/le4_08.joblib')
le4_09 = load('./bladder cancer/le4_09.joblib')
le4_10 = load('./bladder cancer/le4_10.joblib')
le4_11 = load('./bladder cancer/le4_11.joblib')
le4_12 = load('./bladder cancer/le4_12.joblib')
le4_13 = load('./bladder cancer/le4_13.joblib')
le4_14 = load('./bladder cancer/le4_14.joblib')
le4_15 = load('./bladder cancer/le4_15.joblib')
model4 = load('./bladder cancer/modelo4.joblib')
report_df4 = pd.read_csv('./bladder cancer/report4.csv', sep=',', header=0, index_col=0)

le5_01 = load('./glioblastoma/le5_01.joblib')
le5_03 = load('./glioblastoma/le5_03.joblib')
le5_04 = load('./glioblastoma/le5_04.joblib')
le5_06_oss = load('./glioblastoma/le5_06_oss.joblib')
le5_08 = load('./glioblastoma/le5_08.joblib')
le5_11 = load('./glioblastoma/le5_11.joblib')
le5_12 = load('./glioblastoma/le5_12.joblib')
model5 = load('./glioblastoma/modelo5.joblib')
report_df5 = pd.read_csv('./glioblastoma/report5.csv', sep=',', header=0, index_col=0)

le6 = load('./lung cancer/le6.joblib')
model6 = load('./lung cancer/modelo6.joblib')
report_df6 = pd.read_csv('./lung cancer/report6.csv', sep=',', header=0, index_col=0)

def main():
    st.title('Predicción de cáncer')
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Higado", "Piel", "Mama", "Vejiga", "Glioblastoma","Pulmon"])
    
    with tab1:
        st.header('Clasificación del carcinoma hepatocelular de hígado')
        
        with st.expander("Metricas del modelo 1"):
            st.write("El metodo usado es Gradient Boosting Classifier")
            fig, axes = plt.subplots(1, 3, figsize=(20, 5))

            # Heatmap para precision
            sns.heatmap(report_df1[['precision']].sort_values(by='precision', ascending=False), annot=True, cmap='Blues', ax=axes[0])
            axes[0].set_title('Heatmap de Precision')

            # Heatmap para recall
            sns.heatmap(report_df1[['recall']].sort_values(by='recall', ascending=False), annot=True, cmap='Blues', ax=axes[1])
            axes[1].set_title('Heatmap de Recall')

            # Heatmap para f1-score
            sns.heatmap(report_df1[['f1-score']].sort_values(by='f1-score', ascending=False), annot=True, cmap='Blues', ax=axes[2])
            axes[2].set_title('Heatmap de F1-Score')

            # Ajustes de diseño
            plt.tight_layout()

            # Guardar la figura para su uso en Streamlit
            st.pyplot(fig)

        cirrhosis = st.number_input('Cirrosis', value=3, min_value=1, max_value=4, help='1-4')
        diagnosis_age = st.number_input('Edad', value=55, min_value=26, max_value=80, help='26-80')
        disease_free_months = st.number_input('Sin enfermedad (meses)', value=43.5, min_value=0.2, max_value=86.9, step=0.1, format="%.1f", help='0.2-86.9')
        disease_free_status = st.selectbox('Estado libre de enfermedad', ('DiseaseFree','Recurred'))
        mutation_count = st.number_input('Recuento de mutaciones', value=506, min_value=2, max_value=1014, step=1, help='1-1014')
        neoplasm_histologic_grade = st.selectbox('Neoplasia Grado histológico', ('I','II','III','IIII','IV'))
        overall_survival_months = st.number_input('Supervivencia global (meses)', value=47.2, min_value=5.5, max_value=88.9, step=0.1, format="%.1f", help='5.5-88.9')
        tumor_other_histologic_subtype = st.selectbox('Otro subtipo histológico del tumor', ('HBV','NBNC','HCV'))
        
        col1, col2, col3 = st.columns(3)
        
        with col2:
            # Button to predict
            if st.button('Predecir',key='1'):
                diseaseFreeStatus = le1_dfs.transform([disease_free_status])
                neoplasmHistologicGrade = le2_nhg.transform([neoplasm_histologic_grade])
                tumorOtherHistologicSubtype = le4_tohs.transform([tumor_other_histologic_subtype])
                diseaseFreeStatus = int(diseaseFreeStatus)
                neoplasmHistologicGrade = int(neoplasmHistologicGrade)
                tumorOtherHistologicSubtype = int(tumorOtherHistologicSubtype)

                data = {'Cirrhosis': cirrhosis,
                    'Diagnosis Age': diagnosis_age,
                    'Disease Free (Months)': disease_free_months,
                    'dfs': diseaseFreeStatus,
                    'Mutation Count': mutation_count,
                    'nhg': neoplasmHistologicGrade,
                    'Overall Survival (Months)': overall_survival_months,
                    'tohs': tumorOtherHistologicSubtype}
                df = pd.DataFrame(data, index=[0])
                prediction = model.predict(df)  
                prediction = le3_oss.inverse_transform(prediction.tolist())
                    
                st.markdown(f"""
                            ### La predicción es: 
                            # **{prediction[0]}**
                            """)  # Access the first element of the prediction list
     
    with tab2:
        st.header('Clasificación del melanoma de piel')  
        
        with st.expander("Metricas del modelo 2"):
            st.write("El metodo usado es Gradient Boosting Classifier")
            fig2, axes2 = plt.subplots(1, 3, figsize=(20, 5))

            # Heatmap para precision
            sns.heatmap(report_df2[['precision']].sort_values(by='precision', ascending=False), annot=True, cmap='Blues', ax=axes2[0])
            axes2[0].set_title('Heatmap de Precision')

            # Heatmap para recall
            sns.heatmap(report_df2[['recall']].sort_values(by='recall', ascending=False), annot=True, cmap='Blues', ax=axes2[1])
            axes2[1].set_title('Heatmap de Recall')

            # Heatmap para f1-score
            sns.heatmap(report_df2[['f1-score']].sort_values(by='f1-score', ascending=False), annot=True, cmap='Blues', ax=axes2[2])
            axes2[2].set_title('Heatmap de F1-Score')

            # Ajustes de diseño
            plt.tight_layout()

            # Guardar la figura para su uso en Streamlit
            st.pyplot(fig2)
        
        age_at_procurement = st.number_input('Edad en la adquisición', value=60, min_value=25, max_value=94, help='Edad del paciente 25-94')
        mutation_count2 = st.number_input('Recuento de mutaciones', value=1821, min_value=1, max_value=3642, step=1, help='1-3642')
        mutation_status = st.selectbox('Estado de la mutación', ('WT', 'NRAS-Q61R', 'BRAF-V600K', 'BRAF-V600E', 'NRAS-Q61H','NRAS-Q61K', 'HRAS-Q61H', 'NRAS-Q61L', 
                                                                 '"NRAS-Q61K, KIT-L160V"','NRAS-G12V', 'KIT-V559D', 'KIT-N822Y', 'KIT-K642E', 'NRAS-G12D'))
        person_gender = st.selectbox('Genero', ('Male','Female'))
        primary_tumor_site = st.selectbox('Sitio del tumor primario', ('head/neck', 'extremity', 'choroid', 'heel', 'trunk', 'unknown','subungual', 'gingiva', 
                                                                       'sole', 'toe', 'nasal cavity','sun-exposed', 'finger', 'vulva'))
        sample_type = st.selectbox('Tipo de muestra', ('metastasis', 'primary'))
        specimen_site = st.selectbox('Sitio de la muestra', ('head/neck', 'extremity', 'trunk', 'lung', 'sole', 'choroid','lymph', 'subungual', 'brain', 'mucosal', 'pleural cavity','intestine', 'finger', 'gall bladder', 'small intestine'))
        tissue_source_site = st.selectbox('Sitio de origen del tejido', ('Tumor', 'Cell line'))
        
        col1, col2, col3 = st.columns(3)
        
        with col2:
            if st.button('Predecir',key='2'):
                mutationStatus = le21_ms.transform([mutation_status])
                personGender = le22_pg.transform([[person_gender]])
                primaryTumorSite = le23_pts.transform([primary_tumor_site])
                sampleType = le24_st.transform([sample_type])
                specimenSite = le25_ss.transform([specimen_site])
                tissueSourceSite = le26_tss.transform([tissue_source_site])
                
                data2 = {'Age At Procurement': age_at_procurement,
                        'Mutation Count': mutation_count2,
                        'ms': mutationStatus,
                        'pg': personGender,
                        'pts': primaryTumorSite,
                        'st': sampleType,
                        'ss': specimenSite,
                        'tss': tissueSourceSite}
                
                df2 = pd.DataFrame(data2, index=[0])
                prediction2 = model2.predict(df2)
                prediction2 = le27_tipo.inverse_transform(prediction2.tolist())
                
                st.markdown(f"""
                            ### La predicción es: 
                            # **{prediction2[0]}** """) 
                
    with tab3:
        st.header('Clasificación de Cancer de Mama') 
        
        with st.expander("Metricas del modelo 3"):
            st.write("El metodo usado es Random Forest Classifier")
            fig3, axes3 = plt.subplots(1, 3, figsize=(20, 5))

            # Heatmap para precision
            sns.heatmap(report_df3[['precision']].sort_values(by='precision', ascending=False), annot=True, cmap='Blues', ax=axes3[0])
            axes3[0].set_title('Heatmap de Precision')

            # Heatmap para recall
            sns.heatmap(report_df3[['recall']].sort_values(by='recall', ascending=False), annot=True, cmap='Blues', ax=axes3[1])
            axes3[1].set_title('Heatmap de Recall')

            # Heatmap para f1-score
            sns.heatmap(report_df3[['f1-score']].sort_values(by='f1-score', ascending=False), annot=True, cmap='Blues', ax=axes3[2])
            axes3[2].set_title('Heatmap de F1-Score')

            # Ajustes de diseño
            plt.tight_layout()

            # Guardar la figura para su uso en Streamlit
            st.pyplot(fig3)
        
        clump_thickness = st.number_input('Grosor del racimo', value=5, min_value=1, max_value=10, step=1, help='1-10')
        uniformity_of_cell_size = st.number_input('Uniformidad del tamaño de la célula', value=5, min_value=1, max_value=10, step=1, help='1-10')
        uniformity_of_cell_shape = st.number_input('Uniformidad de la forma de la célula', value=5, min_value=1, max_value=10, step=1, help='1-10')
        marginal_adhesion = st.number_input('Adhesión marginal', value=5, min_value=1, max_value=10, step=1, help='1-10')
        single_epithelial_cell_size = st.number_input('Tamaño de célula epitelial simple', value=5.0, min_value=1.0, max_value=10.0, step=1.0, format="%.1f", help='1-10')
        bare_nuclei = st.number_input('Núcleo desnudo', value=5, min_value=1, max_value=10, step=1, help='1-10')
        bland_chromatin = st.number_input('Cromatina suave', value=5, min_value=1, max_value=10, step=1, help='1-10')
        normal_nucleoli = st.number_input('Nucleoli normales', value=5, min_value=1, max_value=10, step=1, help='1-10')
        mitoses = st.number_input('Mitosis', value=5, min_value=1, max_value=10, step=1, help='1-10')
        
        col1, col2, col3 = st.columns(3)
        
        with col2:
            if st.button('Predecir',key='3'):
                data3 = {'Clump Thickness': clump_thickness,
                        'Uniformity of Cell Size': uniformity_of_cell_size,
                        'Uniformity of Cell Shape': uniformity_of_cell_shape,
                        'Marginal Adhesion': marginal_adhesion,
                        'Single Epithelial Cell Size': single_epithelial_cell_size,
                        'Bare Nuclei': bare_nuclei,
                        'Bland Chromatin': bland_chromatin,
                        'Normal Nucleoli': normal_nucleoli,
                        'Mitoses': mitoses}
                df3 = pd.DataFrame(data3, index=[0])
                prediction3 = model3.predict(df3)
                prediction3 = le31.inverse_transform(prediction3.tolist())
                st.markdown(f"""
                            #### Tienes cancer de tipo 
                            # **{prediction3[0]}** """) 
    
    with tab4:
        st.header('Clasificación del carcinoma de vejiga')
        
        with st.expander("Metricas del modelo 4"):
            st.write("El metodo usado es Gradient Boosting Classifier")
            fig4, axes4 = plt.subplots(1, 3, figsize=(20, 5))

            # Heatmap para precision
            sns.heatmap(report_df4[['precision']].sort_values(by='precision', ascending=False), annot=True, cmap='Blues', ax=axes4[0])
            axes4[0].set_title('Heatmap de Precision')

            # Heatmap para recall
            sns.heatmap(report_df4[['recall']].sort_values(by='recall', ascending=False), annot=True, cmap='Blues', ax=axes4[1])
            axes4[1].set_title('Heatmap de Recall')

            # Heatmap para f1-score
            sns.heatmap(report_df4[['f1-score']].sort_values(by='f1-score', ascending=False), annot=True, cmap='Blues', ax=axes4[2])
            axes4[2].set_title('Heatmap de F1-Score')

            # Ajustes de diseño
            plt.tight_layout()

            # Guardar la figura para su uso en Streamlit
            st.pyplot(fig4)
        
        cna = st.number_input('CNA',value=0.350, min_value=0.000, max_value=0.700, step=0.010, format="%.3f", help='0.000-0.700')
        concomitant_carcinoma_in_situ = st.selectbox('Carcinoma concomitante in situ', ('False','True'))
        death_due_to_disease = st.selectbox('Muerte por enfermedad', ('NO','YES'))
        diagnosis_age4 = st.number_input('Diagnóstico Edad',value=60, min_value=30, max_value=90, help='30-90')
        disease_free_months4 = st.number_input('Meses sin enfermedad', value=5.4, min_value=0.0, max_value=10.8, step=0.1, format="%.1f", help='0.0-10.8')
        disease_free_status4 = st.selectbox('Estado libre de enfermedad', ('DiseaseFree','Recurred/Progressed'))
        ln_status = st.selectbox('Estado de LN', ('Negative','Positive'))
        mutation_count4 = st.number_input('Recuento de mutaciones', value=22, min_value=2, max_value=46, step=1, help='2-46')
        neoadjuvant_chemotherapy = st.selectbox('Quimioterapia neoadyuvante', ('False','True'))
        overall_survival_months4 = st.number_input('Meses de supervivencia global', value=5.4, min_value=0.0, max_value=10.8, step=0.1, format="%.1f", help='0.0-10.8')
        pt_stage = st.selectbox('Etapa PT', ('pT0','pT1','pT2','pT3','pT4','Ta','pTa','T1','pTis'))
        race_category = st.selectbox('Categoría de raza', ('White','Non-Caucasian'))
        sex = st.selectbox("Sexo", ("MALE","FEMALE"))
        smoking_status = st.selectbox('Estado de fumador', ('Former','Never','Active')) #active
        surgical_treatment = st.selectbox('Tratamiento quirúrgico', ('Radical Cystectomy','TUR'))
        tissue_sequenced = st.selectbox('Tejido secuenciado', ('Radical Cystectomy','TUR'))
        variant_histology = st.selectbox('Histología variante', ('No','Yes'))
        # primary_histologic = st.selectbox('Diagnóstico histológico primario', ('Urothelial carcinoma','Squamous cell carcinoma','Adenocarcinoma','Small cell carcinoma','Neuroendocrine carcinoma'))
        # sample_type4 = st.selectbox('Tipo de muestra', ('Primary','Metastasis'))
        
        col1, col2, col3 = st.columns(3)
        
        with col2:
            if st.button('Predecir',key='4'):
                # ccis = le4_01.transform([concomitant_carcinoma_in_situ])
                ddtd = le4_02.transform([death_due_to_disease])
                dfs = le4_03.transform([disease_free_status4])
                lnstatus = le4_04.transform([ln_status])
                # neoadjuvant = le4_05.transform([neoadjuvant_chemotherapy])
                # p_histology = le4_07.transform([primary_histologic])
                pt_stage = le4_08.transform([pt_stage])
                # raceCategory = le4_09.transform([race_category])
                # sampleType4 = le4_10.transform([sample_type4])
                # sexo = le4_11.transform([sex])
                # smokingStatus = le4_12.transform([smoking_status])
                surgicalTreatment = le4_13.transform([surgical_treatment])
                tissueSequenced = le4_14.transform([tissue_sequenced])
                variantHistology = le4_15.transform([variant_histology])
                data4 = { 
                            'Disease Free (Months)': disease_free_months4,
                            'Overall Survival (Months)': overall_survival_months4,
                            'ddtd': ddtd,
                            'dfs': dfs,
                            'lnstatus': lnstatus,
                            'pt_stage': pt_stage,
                            'surgical_treatment': surgicalTreatment,
                            'tissue_sequenced': tissueSequenced,
                            'variant_histology': variantHistology
                            # 'CNA': cna,
                            # 'ccis': ccis,
                            # 'Diagnosis Age': diagnosis_age4,
                            # 'Mutation Count': mutation_count4,
                            # 'neoadjuvant': neoadjuvant,
                            # 'p_histology': p_histology,
                            # 'race_category': raceCategory,
                            # 'sexo': sexo,
                            # 'smoking_status': smokingStatus,
                            }
                df4 = pd.DataFrame(data4, index=[0])
                prediction4 = model4.predict(df4)
                prediction4 = le4_06_oss.inverse_transform(prediction4.tolist())
                st.markdown(f"""
                            #### El paciente esta 
                            # **{prediction4[0]}** """)
    
    with tab5:
        st.header('Clasificación del Tumor cerebral')
        
        with st.expander("Metricas del modelo 5"):
            st.write("El metodo usado es Gradient Boosting Classifier")
            fig5, axes5 = plt.subplots(1, 3, figsize=(20, 6))

            # Heatmap para precision
            sns.heatmap(report_df5[['precision']].sort_values(by='precision', ascending=False), annot=True, cmap='Blues', ax=axes5[0])
            axes5[0].set_title('Heatmap de Precision')

            # Heatmap para recall
            sns.heatmap(report_df5[['recall']].sort_values(by='recall', ascending=False), annot=True, cmap='Blues', ax=axes5[1])
            axes5[1].set_title('Heatmap de Recall')

            # Heatmap para f1-score
            sns.heatmap(report_df5[['f1-score']].sort_values(by='f1-score', ascending=False), annot=True, cmap='Blues', ax=axes5[2])
            axes5[2].set_title('Heatmap de F1-Score')

            # Ajustes de diseño
            plt.tight_layout()

            # Guardar la figura para su uso en Streamlit
            st.pyplot(fig5)
        
        acgh = st.selectbox('ACGH', ('YES','NO'))
        # cna = st.number_input('CNA', min_value=0.000, max_value=0.700, step=0.010, format="%.3f")
        complete_data = st.selectbox('Datos completos', ('YES','NO'))
        disease_free_months5 = st.number_input('Meses sin enfermedad', value=38.5, min_value=0.0, max_value=77.0, step=0.1, format="%.1f", help='0.0-77.0')
        disease_free_status5 = st.selectbox('Estado libre de enfermedad', ('DiseaseFree','Recurred'), key='sdfs5')
        # icd10_classification = st.selectbox('Clasificación ICD10', ('C71','C72'))
        karnofsky_performance_score = st.number_input('Puntuación de rendimiento de Karnofsky', value=80.0, min_value=40.0, max_value=100.0, step=20.0, help='40, 60, 80 o 100', format="%.1f")
        # mrna_data = st.selectbox('Datos de ARNm', ('YES','NO'))
        mutation_count5 = st.number_input('Recuento de mutaciones', value=34, min_value=0, max_value=68, step=1, help='0-68')
        overall_survival_months5 = st.number_input('Meses de supervivencia global', value=58, min_value=0, max_value=116, step=1, help='0-116')
        # person_gender5 = st.selectbox("Genero",("FEMALE","MALE"))
        pretreatment_history = st.selectbox('Historia de tratamiento previo', ('NO','YES'))
        # prior_glioma = st.selectbox('Glioma previo', ('NO','YES'))
        sequenced = st.selectbox('Secuenciado', ('YES','NO'))
        treatment_status = st.selectbox('Estado del tratamiento', ('Untreated','Treated'))
        
        col1, col2, col3 = st.columns(3)
        
        with col2:
            if st.button('Predecir',key='5'):
                acgh_data = le5_01.transform([acgh])
                completeData = le5_03.transform([complete_data])
                d_free_status = le5_04.transform([disease_free_status5])
                ptreatment_history = le5_08.transform([pretreatment_history])
                sequenceded = le5_11.transform([sequenced])
                treatmentStatus = le5_12.transform([treatment_status])
                data5 = { 
                            'Disease Free (Months)': disease_free_months5,
                            'Mutation Count': mutation_count5,
                            'Overall Survival (Months)': overall_survival_months5,
                            'acgh_data': acgh_data,
                            'complete_data': completeData,
                            'd_free_status': d_free_status,
                            'ptreatment_history': ptreatment_history,
                            'sequenced': sequenceded,
                            'treatment_status': treatmentStatus,
                            'Karnofsky Performance Score': karnofsky_performance_score,
                            }
                df5 = pd.DataFrame(data5, index=[0])
                prediction5 = model5.predict(df5)
                prediction5 = le5_06_oss.inverse_transform(prediction5.tolist())
                st.markdown(f"""
                            #### El paciente esta 
                            # **{prediction5[0]}** """)
                
    with tab6:
        st.header('Clasificación del carcinoma de pulmón')
        
        with st.expander("Metricas del modelo 6"):
            st.write("El metodo usado es Decision Tree Classifier")
            fig6, axes6 = plt.subplots(1, 3, figsize=(20, 6))

            # Heatmap para precision
            sns.heatmap(report_df6[['precision']].sort_values(by='precision', ascending=False), annot=True, cmap='Blues', ax=axes6[0])
            axes6[0].set_title('Heatmap de Precision')

            # Heatmap para recall
            sns.heatmap(report_df6[['recall']].sort_values(by='recall', ascending=False), annot=True, cmap='Blues', ax=axes6[1])
            axes6[1].set_title('Heatmap de Recall')

            # Heatmap para f1-score
            sns.heatmap(report_df6[['f1-score']].sort_values(by='f1-score', ascending=False), annot=True, cmap='Blues', ax=axes6[2])
            axes6[2].set_title('Heatmap de F1-Score')

            # Ajustes de diseño
            plt.tight_layout()

            # Guardar la figura para su uso en Streamlit
            st.pyplot(fig6)
        
        # age = st.number_input('Edad', min_value=18, max_value=90, step=1)
        # gender6 = st.number_input('Genero', min_value=0, max_value=1, step=1)
        air_pollution = st.number_input('Contaminación del aire', value=4, min_value=1, max_value=8, step=1, help='1-8')
        alcohol_use = st.number_input('Consumo de alcohol', value=4, min_value=1, max_value=8, step=1, help='1-8')
        dust_allergy = st.number_input('Alergia al polvo', value=4, min_value=1, max_value=8, step=1, help='1-8')
        occupational_hazards = st.number_input('Riesgos laborales', value=4, min_value=1, max_value=8, step=1, help='1-8')
        genetic_risk = st.number_input('Riesgo genético', value=3, min_value=1, max_value=7, step=1, help='1-7')
        chronic_lung_disease = st.number_input('Enfermedad pulmonar crónica', value=3, min_value=1, max_value=7, step=1, help='1-7')
        balanced_diet = st.number_input('Dieta equilibrada', value=3, min_value=1, max_value=7, step=1, help='1-7')
        obesity = st.number_input('Obesidad', value=3, min_value=1, max_value=7, step=1, help='1-7')
        smoking = st.number_input('Fumar', value=4, min_value=1, max_value=8, step=1, help='1-8')
        passive_smoker = st.number_input('Fumador pasivo', value=4, min_value=1, max_value=8, step=1, help='1-8')
        chest_paint = st.number_input('Dolor en el pecho', value=5, min_value=1, max_value=9, step=1, help='1-9')
        coughing_of_blood = st.number_input('Tos de sangre', value=5, min_value=1, max_value=9, step=1, help='1-9')
        fatigue = st.number_input('Fatiga', value=5, min_value=1, max_value=9, step=1, help='1-9')
        weight_loss = st.number_input('Pérdida de peso', value=4, min_value=1, max_value=8, step=1, help='1-8')
        shortness_of_breath = st.number_input('Falta de aliento', value=5, min_value=1, max_value=9, step=1, help='1-9')
        # wheezing = st.number_input('Sibilancias', min_value=1, max_value=3, step=1)
        # swallowing_difficulty = st.number_input('Dificultad para tragar', min_value=1, max_value=3, step=1)
        clubbing_of_finger_nails = st.number_input('Clubbing of finger nails', value=5, min_value=1, max_value=9, step=1, help='1-9')
        frequent_cold = st.number_input('Resfriado frecuente', value=3, min_value=1, max_value=7, step=1, help='1-7')
        dry_cough = st.number_input('Tos seca', value=3, min_value=1, max_value=7, step=1, help='1-7')
        # snoring = st.number_input('Ronquidos', min_value=1, max_value=3, step=1)
        
        col1, col2, col3 = st.columns(3)
        
        with col2:
            if st.button('Predecir',key='6'):
                data6 = { 
                            'Air Pollution': air_pollution,
                            'Alcohol use': alcohol_use,
                            'Dust Allergy': dust_allergy,
                            'OccuPational Hazards': occupational_hazards,
                            'Genetic Risk': genetic_risk,
                            'chronic Lung Disease': chronic_lung_disease,
                            'Balanced Diet': balanced_diet,
                            'Obesity': obesity,
                            'Smoking': smoking,
                            'Passive Smoker': passive_smoker,
                            'Chest Pain': chest_paint,
                            'Coughing of Blood': coughing_of_blood,
                            'Fatigue': fatigue,
                            'Weight Loss': weight_loss,
                            'Shortness of Breath': shortness_of_breath,
                            'Clubbing of Finger Nails': clubbing_of_finger_nails,
                            'Frequent Cold': frequent_cold,
                            'Dry Cough': dry_cough,
                            }
                df6 = pd.DataFrame(data6, index=[0])
                prediction6 = model6.predict(df6)
                prediction6 = le6.inverse_transform(prediction6.tolist())
                st.markdown(f"""
                            #### Su cancer es 
                            # **{prediction6[0]}** """)
                    
if __name__ == '__main__':
    main()