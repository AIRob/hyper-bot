from bot.knowledge_graph.BotKgLabelDiseaseBase import BotKGInferLabelDiseaseBase
from bot.knowledge_graph.BotKgLabelDiseaseDiagnosis import BotKGInferLabelDiseaseDiagnosis
from bot.knowledge_graph.BotKgLabelDiseaseCheck import BotKGInferLabelDiseaseCheck
from bot.knowledge_graph.BotKgLabelDiseaseDetailComplication import BotKGinferLabelDiseaseComplication
from bot.knowledge_graph.BotKgLabelDiseaseDetailSymptoms import BotKGinferLabelDiseaseSymptoms
from bot.knowledge_graph.BotKgLabelDiseaseCause import BotKGInferLabelDiseaseCause
from bot.knowledge_graph.BotKgLabelDiseaseHowPrevent import BotKGInferLabelDiseaseHowPrevent
from bot.knowledge_graph.BotKgLabelDiseaseQAcorpus import BotKGInferLabelDiseaseQAcorpus
from bot.knowledge_graph.BotKgLabelDiseaseDrug import BotKGInferLabelDiseaseDrug
from bot.knowledge_graph.BotKgLabelDiseaseTreat import BotKGInferLabelDiseaseTreat

#from sim_sentence_api import vote_sim_reply


def botKGInferAllLabel(sentence):
    bkgildb = BotKGInferLabelDiseaseBase()
    bkgildd = BotKGInferLabelDiseaseDiagnosis()
    bkgildc = BotKGInferLabelDiseaseCheck()
    bkgilds = BotKGinferLabelDiseaseSymptoms()
    bkgildcom = BotKGinferLabelDiseaseComplication()
    bkgildcau = BotKGInferLabelDiseaseCause()
    bkgildt = BotKGInferLabelDiseaseTreat()
    bkgildhp = BotKGInferLabelDiseaseHowPrevent()
    bkgildqa = BotKGInferLabelDiseaseQAcorpus()
    bkgilddrg = BotKGInferLabelDiseaseDrug()

    if '高血压' in sentence and '疾病简介' in sentence:
        print(bkgildb.inferLabelDiseaseProfile(sentence))
        return (bkgildb.inferLabelDiseaseProfile(sentence))
    #None
    elif '高血压' in sentence and '基本知识' in sentence:
        return (bkgildb.inferLabelDiseaseBase(sentence))

    elif '高血压' in sentence and ('别名' in sentence or '其他叫法' in sentence):
        #return (bkgildb.inferLabelDiseaseBaseAlias(sentence))
        return (bkgildb.inferLabelDiseaseBaseAlias())

    elif '高血压' in sentence and '医保' in sentence:
        return (bkgildb.inferLabelDiseaseBaseIsmedical(sentence))

    elif '高血压' in sentence and ('发病部位' in sentence or '发病部位' in sentence):
        return (bkgildb.inferLabelDiseaseBaseIncidenceSite(sentence))

    elif '高血压' in sentence and ('传染性' in sentence or '传染' in sentence):
        return (bkgildb.inferLabelDiseaseBaseContagious(sentence))

    elif '高血压' in sentence and ('多发人群' in sentence or '多发性' in sentence):
        return (bkgildb.inferLabelDiseaseBaseMultiplePeople(sentence))

    elif '高血压' in sentence and '典型症状' in sentence:
        return (bkgildb.inferLabelDiseaseBaseTypicalSymptoms(sentence))

    elif '高血压' in sentence and '并发症简介' in sentence:
        return (bkgildb.inferLabelDiseaseBaseComplication(sentence))

    #2
    elif '高血压' in sentence and ('最佳时间' in sentence or '最佳就诊时间' in sentence):
        return (bkgildd.inferLabelDiseaseDiagnosisBesttime(sentence))
    elif '高血压' in sentence and ('就诊时长' in sentence or '就诊多长' in sentence):
        return (bkgildd.inferLabelDiseaseDiagnosisDurationVisit(sentence))
    elif '高血压' in sentence and ('复诊频率' in sentence or '复诊' in sentence):
        return (bkgildd.inferLabelDiseaseDiagnosisFollowupfreq(sentence))
    elif '高血压' in sentence and ('就诊前准备' in sentence or '就诊准备' in sentence):
        return (bkgildd.inferLabelDiseaseDiagnosisPretreat(sentence))

    #3
    #elif '高血压' in sentence and ('检查' in sentence or '疾病检查' in sentence):
    #    return (bkgildc.inferLabelDiseaseCheck(sentence))
    elif '高血压' in sentence and ('检查URL' in sentence or '检查链接' in sentence):
        return (bkgildc.inferLabelDiseaseCheckURL(sentence))
    elif '高血压' in sentence and '常见检查' in sentence:
        return (bkgildc.inferLabelDiseaseCheckCommoncheck(sentence))
    elif '高血压' in sentence and '详细检查' in sentence:
        return (bkgildc.inferLabelDiseaseCheckChecks(sentence))
    elif '高血压' in sentence and ('检查浏览量' in sentence or '检查收藏量' in sentence):
        return (bkgildc.inferLabelDiseaseCheckCollectCount(sentence))

    #4 
    elif '高血压' in sentence and ('症状URL' in sentence or '症状链接' in sentence):
        return (bkgilds.inferLabelDiseaseSymptomsURL(sentence))
    elif '高血压' in sentence and '主要症状' in sentence:
        return (bkgilds.inferLabelDiseaseSymptomsCommonsymptoms(sentence))
    elif '高血压' in sentence and '症状' in sentence:
        return (bkgilds.inferLabelDiseaseSymptomsLinkssymptoms(sentence))
    elif '高血压' in sentence and ('症状浏览量' in sentence or '症状收藏量' in sentence):
        return (bkgilds.inferLabelDiseaseSymptomsCollectCount(sentence))

    #5
    elif '高血压' in sentence and ('并发症URL' in sentence or '并发症链接' in sentence):
        return (bkgildcom.inferLabelDiseaseComplicationURL(sentence))
    elif '高血压' in sentence and ('常见并发症' in sentence or '关联症状' in sentence or '关联疾病' in sentence):
        return (bkgildcom.inferLabelDiseaseComplicationCommonComplication(sentence))
    elif '高血压' in sentence and '主要并发症' in sentence:
        return (bkgildcom.inferLabelDiseaseComplication(sentence))
    elif '高血压' in sentence and '详细并发症' in sentence :
        return (bkgildcom.inferLabelDiseaseComplicationDetail(sentence))
    elif '高血压' in sentence and ('浏览量' in sentence or '收藏量' in sentence):
        return (bkgildcom.inferLabelDiseaseComplicationCollectCount(sentence))

    #6
    elif '高血压' in sentence and ('病因' in sentence or '原因' in sentence):
        return (bkgildcau.inferLabelDiseaseCause(sentence))
     
    #7
    #elif '高血压' in sentence and ('治疗' in sentence or '治疗方案' in sentence):
    #    return (bkgildt.inferLabelDiseaseTreat(sentence))
    elif '高血压' in sentence and ('治疗方法' in sentence or '如何治疗' in sentence):
        return (bkgildt.inferLabelDiseaseTreatMethod(sentence))
    elif '高血压' in sentence and ('治疗费用' in sentence or '治疗费' in sentence):
        return (bkgildt.inferLabelDiseaseTreatCosts(sentence))
    elif '高血压' in sentence and ('治愈率' in sentence or '治愈' in sentence):
        return (bkgildt.inferLabelDiseaseTreatRate(sentence))
    elif '高血压' in sentence and ('治疗周期' in sentence or '疗程' in sentence):
        return (bkgildt.inferLabelDiseaseTreatCycle(sentence))
    elif '高血压' in sentence and ('常用药品' in sentence or '必备药' in sentence):
        return (bkgildt.inferLabelDiseaseTreatCommondrugs(sentence))
    elif '高血压' in sentence and ('就诊科室' in sentence or '科室' in sentence):
        return (bkgildt.inferLabelDiseaseTreatVisitdepartment(sentence))

    #8
    elif '高血压' in sentence and ('预防' in sentence or '如何预防' in sentence):
        return (bkgildhp.inferLabelDiseaseHowPrevent(sentence))
        
    #9
    elif '高血压' in sentence and ('问诊' in sentence or '咨询' in sentence):
        return (bkgildqa.inferLabelDiseaseQAcorpus(sentence))
    
    #10  
    elif '高血压' in sentence and '盐酸阿罗洛尔片' in sentence and '疾病' in sentence:
        return (bkgilddrg.inferLabelDiseaseDrugTherapeuticdiseases(sentence))
    elif '高血压' in sentence and '盐酸阿罗洛尔片' in sentence and ('功能主治' in sentence or '主治功能' in sentence):
        return (bkgilddrg.inferLabelDiseaseDrugFunctions(sentence))  
    elif '高血压' in sentence and '盐酸阿罗洛尔片' in sentence and ('成份' in sentence or '成分' in sentence or '组成' in sentence):
        return (bkgilddrg.inferLabelDiseaseDrugIngredients(sentence)) 
    elif '高血压' in sentence and '盐酸阿罗洛尔片' in sentence and '不良反应' in sentence:
        return (bkgilddrg.inferLabelDiseaseDrugAdversereactions(sentence)) 
    elif '高血压' in sentence and '盐酸阿罗洛尔片' in sentence and ('注意事项' in sentence or '注意' in sentence):
        return (bkgilddrg.inferLabelDiseaseDrugPrecautions(sentence))
    elif '高血压' in sentence and '盐酸阿罗洛尔片' in sentence and ('禁忌' in sentence or '药品禁忌' in sentence):
        return (bkgilddrg.inferLabelDiseaseDrugTaboo(sentence))
    elif '高血压' in sentence and '盐酸阿罗洛尔片' in sentence and ('药物相互作用' in sentence or '相互作用' in sentence):
        return (bkgilddrg.inferLabelDiseaseDrugMedicineInteractions(sentence))
    elif '高血压' in sentence and '盐酸阿罗洛尔片' in sentence and ('药理作用' in sentence or '药理' in sentence):
        return (bkgilddrg.inferLabelDiseaseDrugPharmacologicalAction(sentence))
    elif '高血压' in sentence and '盐酸阿罗洛尔片' in sentence and ('特殊人群' in sentence or '特殊人群用药' in sentence or '特殊用药' in sentence):
        return (bkgilddrg.inferLabelDiseaseDrugSpecialPopulation(sentence))
        #null
    else:
        return "none"
def main():
    sentence = '高血压的盐酸阿罗洛尔片特殊人群用药'
    print(botKGInferAllLabel(sentence))

if __name__ == '__main__':
    main()

