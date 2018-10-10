from bot.knowledge_graph.BotKGInfer import KGLableNamesInfer


class BotKGInferLabelDiseaseDiagnosis(object):
    """docstring for Py2neoConnect"""
    def __init__(self):
        self.kglni = KGLableNamesInfer()

    def inferLabelDiseaseDiagnosis(self,sentence):
        if '高血压' in sentence and ('诊疗知识' in sentence or '诊断知识' in sentence):
            return self.kglni.testLabelDiseaseDiagnosis()
    
    def inferLabelDiseaseDiagnosisBesttime(self,sentence):
        if '高血压' in sentence and ('最佳时间' in sentence or '最佳就诊时间' in sentence):
            bestTimeStr =  self.kglni.testLabelDiseaseDiagnosisList('best_time')
            if bestTimeStr == '无特殊，尽快就诊':return bestTimeStr
            else:return '无特殊，尽快就诊'

    def inferLabelDiseaseDiagnosisDurationVisit(self,sentence):
        if '高血压' in sentence and ('就诊时长' in sentence or '就诊时间' in sentence):
            return self.kglni.testLabelDiseaseDiagnosisList('duration_visit')
    
    def inferLabelDiseaseDiagnosisFollowupfreq(self,sentence):
        if '高血压' in sentence and ('复诊频率' in sentence or '复诊' in sentence):
            return self.kglni.testLabelDiseaseDiagnosisList('followup_freq')

    def inferLabelDiseaseDiagnosisPretreat(self,sentence):
        if '高血压' in sentence and ('就诊前准备' in sentence or '就诊准备' in sentence):
            return self.kglni.testLabelDiseaseDiagnosisList('pre_treat')

    
class InferLabelDiseaseDiagnosisTest(object):
    """docstring for InferLabelTest"""
    def __init__(self):
        self.test = BotKGInferLabelDiseaseDiagnosis()

    def inferLabelDiseaseDiagnosisTest(self):
        diseaseDiagnosis = "高血压疾病诊断知识是什么？"
        return self.test.inferLabelDiseaseDiagnosis(diseaseDiagnosis)

    def inferLabelDiseaseDiagnosisBesttimeTest(self):
        #diseaseDiagnosisBesttime =  '高血压最好最佳什么时候就诊时间？'
        diseaseDiagnosisBesttime =  '高血压#最佳时间？'
        return self.test.inferLabelDiseaseDiagnosisBesttime(diseaseDiagnosisBesttime)

    def inferLabelDiseaseDiagnosisDurationVisitTest(self):
        diseaseDiagnosisDurationVisit = '高血压就诊时长是多久？'
        return self.test.inferLabelDiseaseDiagnosisDurationVisit(diseaseDiagnosisDurationVisit)
    
    def inferLabelDiseaseDiagnosisFollowupfreqTest(self):
        diseaseDiagnosisFollowupfreq = '高血压复诊频率是多久？'
        return self.test.inferLabelDiseaseDiagnosisFollowupfreq(diseaseDiagnosisFollowupfreq)

    def inferLabelDiseaseDiagnosisPretreatTest(self):
        diseaseDiagnosisPretreat =  '高血压就诊前准备需要做哪些？'
        return self.test.inferLabelDiseaseDiagnosisPretreat(diseaseDiagnosisPretreat)


def main():
    #test1 = BotKGInferLabelDiseaseDiagnosis()
    #print(test1.inferLabelDiseaseDiagnosisBesttimeTest())
    
    test = InferLabelDiseaseDiagnosisTest()
    test.inferLabelDiseaseDiagnosisTest()
    print(test.inferLabelDiseaseDiagnosisBesttimeTest())
    print(test.inferLabelDiseaseDiagnosisDurationVisitTest())
    print(test.inferLabelDiseaseDiagnosisFollowupfreqTest())
    print(test.inferLabelDiseaseDiagnosisPretreatTest())
    
if __name__ == '__main__':
    main()
