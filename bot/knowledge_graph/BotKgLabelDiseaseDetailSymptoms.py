from bot.knowledge_graph.BotKGInfer import KGLableNamesInfer


class BotKGinferLabelDiseaseSymptoms(object):
    """docstring for Py2neoConnect"""
    def __init__(self):
        self.kglni = KGLableNamesInfer()

    def inferLabelDiseaseSymptoms(self,sentence):
        if '高血压' in sentence and ('详细症状' in sentence or '主要症状' in sentence):
            return self.kglni.testLabelDiseaseSymptoms()
    
    def inferLabelDiseaseSymptomsURL(self,sentence):
        if '高血压' in sentence and ('症状URL' in sentence or '症状链接' in sentence):
            return  self.kglni.testLabelDiseaseSymptomsURL()

    def inferLabelDiseaseSymptomsCommonsymptoms(self,sentence):
        if '高血压' in sentence and '主要症状' in sentence:
            return self.kglni.testLabelDiseaseSymptomsList('common_symptoms')
    
    def inferLabelDiseaseSymptomsLinkssymptoms(self,sentence):
        if '高血压' in sentence and '症状' in sentence:
            return self.kglni.testLabelDiseaseSymptomsList('links_symptoms')

    def inferLabelDiseaseSymptomsCollectCount(self,sentence):
        if '高血压' in sentence and ('浏览量' in sentence or '收藏量' in sentence):
            return self.kglni.testLabelDiseaseSymptomsCollectCount()

    
class inferLabelDiseaseSymptomsTest(object):
    """docstring for InferLabelTest"""
    def __init__(self):
        self.test = BotKGinferLabelDiseaseSymptoms()

    def inferLabelDiseaseSymptomsTest(self):
        diseaseSymptoms = "高血压疾病详细症状是什么？"
        return self.test.inferLabelDiseaseSymptoms(diseaseSymptoms)

    def inferLabelDiseaseSymptomsURLTest(self):
        diseaseSymptomsURL =  '高血压详细症状URL是哪个？'
        return self.test.inferLabelDiseaseSymptomsURL(diseaseSymptomsURL)

    def inferLabelDiseaseSymptomsCommonsymptomsTest(self):
        detailSymptomsCommonsymptoms = '高血压主要症状有哪些？'
        return self.test.inferLabelDiseaseSymptomsCommonsymptoms(detailSymptomsCommonsymptoms)
    
    def inferLabelDiseaseSymptomsLinkssymptomsTest(self):
        detailSymptomsLinkssymptoms = '高血压详细的症状有哪些？'
        return self.test.inferLabelDiseaseSymptomsLinkssymptoms(detailSymptomsLinkssymptoms)

    def inferLabelDiseaseSymptomsCollectCountTest(self):
        detailSymptomsCollectCount =  '高血压诊断症状收藏量有多大？'
        return self.test.inferLabelDiseaseSymptomsCollectCount(detailSymptomsCollectCount)


def main():

    test = inferLabelDiseaseSymptomsTest()
    test.inferLabelDiseaseSymptomsTest()
    print(test.inferLabelDiseaseSymptomsURLTest())
    print(test.inferLabelDiseaseSymptomsCommonsymptomsTest())
    print(test.inferLabelDiseaseSymptomsLinkssymptomsTest())
    print(test.inferLabelDiseaseSymptomsCollectCountTest())
    
if __name__ == '__main__':
    main()
    