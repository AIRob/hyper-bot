from bot.knowledge_graph.bot_kg_infer import KGLableNamesInfer


def main():
    #KGLableNamesInfer
    print('****************Labels Name******************')
    kglni = KGLableNamesInfer()
    kglni.testLabelDiseaseName()
    kglni.testLabelDiseaseProfile()
    kglni.testLabelDiseaseBase()
    kglni.testLabelDiseaseId()
    kglni.testLabelDiseaseBaseList('disease_alias')
    kglni.testLabelDiseaseIsmedical()
    kglni.testLabelDiseaseBaseList('incidence_site')
    kglni.testLabelDiseaseContagious()
    kglni.testLabelDiseaseBaseMultiplePeople()
    #kglni.testLabelDiseaseBaseTypicalSymptoms()
    #kglni.testLabelDiseaseBaseComplication()
    kglni.testLabelDiseaseBaseList('typical_symptoms')
    kglni.testLabelDiseaseBaseList('complication')

    kglni.testLabelDiseaseDiagnosis()

    kglni.testLabelDiseaseDiagnosisList('best_time')
    kglni.testLabelDiseaseDiagnosisList('duration_visit')
    kglni.testLabelDiseaseDiagnosisList('followup_freq')
    kglni.testLabelDiseaseDiagnosisList('pre_treat')

    kglni.testLabelDiseaseCheck()
    kglni.testLabelDiseaseCheckURL()
    kglni.testLabelDiseaseCheckList('common_check')
    kglni.testLabelDiseaseCheckList('checks')
    kglni.testLabelDiseaseCollectCount()

    kglni.testLabelDiseaseSymptoms()
    kglni.testLabelDiseaseSymptomsURL()
    kglni.testLabelDiseaseSymptomsList('common_symptoms')
    kglni.testLabelDiseaseSymptomsList('links_symptoms')
    kglni.testLabelDiseaseSymptomsList('symptoms')
    kglni.testLabelDiseaseSymptomsCollectCount()

    kglni.testLabelDiseaseComplication()
    kglni.testLabelDiseaseComplicationURL()
    kglni.testLabelDiseaseComplicationList('common_complication')
    kglni.testLabelDiseaseComplicationList('links_symptoms')
    kglni.testLabelDiseaseComplicationList('complication')
    kglni.testLabelDiseaseComplicationList('complication_detail')
    kglni.testLabelDiseaseComplicationCollectCount()

    kglni.testLabelDiseaseTreat()
    kglni.testLabelDiseaseTreatMethod()
    kglni.testLabelDiseaseTreatList('treat_method') #null
    kglni.testLabelDiseaseTreatCosts()
    kglni.testLabelDiseaseTreatRate()
    kglni.testLabelDiseaseTreatCycle()
    kglni.testLabelDiseaseTreatList('common_drugs')
    kglni.testLabelDiseaseTreatList('visit_department')
    kglni.testLabelDiseaseDrug()
    kglni.testLabelDiseaseDrugList('盐酸阿罗洛尔片')
    kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','therapeutic_diseases')
    ####################################
    kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','goods_common_name') #''
    kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','goods_name') #''
    kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','approval_rum') #''
    ####################################
    kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','indication')
    ####################################
    kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','functions') #null
    kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','ingredients') #null
    ####################################
    kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','adverse_reactions')
    kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','precautions')
    kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','taboo')
    kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','medicine_interactions')
    kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','pharmacological_action')
    ####################################
    kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','special_population')#null
    ####################################
    kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','dosage') #''
    kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','storage')
    kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','validity_period')
    kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','drug_form')
    kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','drug_spec')
    kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','manual_revision_date')
    kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','manufacturer')
    kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','business_short_name')
    kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','validproduction_address')
    kglni.testLabelDiseaseDrugDepthXList('盐酸阿罗洛尔片','business_number')
    '''
    kglni.testLabelDiseaseCause()
    kglni.testLabelDiseaseHowPrevent()
    kglni.testLabelDiseaseQAcorpus()
    
    kglni.testLabelPath()
    kglni.testLabelTreat()
    kglni.testLabelCount()
    kglni.testPathExists()
    kglni.testPathDiseaseDrugsExists()
    
    #KGLableCountInfer
    print('****************Labels Count******************')
    kglci = KGLableCountInfer()
    kglci.testLabelCount()

    #KGIndividualNodesInfer
    print('****************Individual Nodes******************')
    kgini = KGIndividualNodesInfer()
    kgini.testIndividualNodes()

    #KGPathExistsInfer
    print('****************Path Exists******************')
    kgpei = KGPathExistsInfer()
    kgpei.testPathExists()

    #KGPathDiseaseDrugsExistsInfer
    print('****************Disease Drugs Exists******************')
    kgpddei = KGPathDiseaseDrugsExistsInfer()
    kgpddei.testPathDiseaseDrugsExists()
    '''

if __name__ == '__main__':
    main()
