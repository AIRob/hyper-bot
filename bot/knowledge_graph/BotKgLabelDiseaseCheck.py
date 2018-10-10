from bot.knowledge_graph.BotKGInfer import KGLableNamesInfer


class BotKGInferLabelDiseaseCheck(object):
    """docstring for Py2neoConnect"""
    def __init__(self):
        self.kglni = KGLableNamesInfer()

    def inferLabelDiseaseCheck(self,sentence):
        if '高血压' in sentence and ('检查' in sentence or '疾病检查' in sentence):
            return self.kglni.testLabelDiseaseCheck()
    
    def inferLabelDiseaseCheckURL(self,sentence):
        if '高血压' in sentence and ('检查URL' in sentence or '检查链接' in sentence):
            return  self.kglni.testLabelDiseaseCheckURL()

    def inferLabelDiseaseCheckCommoncheck(self,sentence):
        if '高血压' in sentence and '常见检查' in sentence:
            return self.kglni.testLabelDiseaseCheckList('common_check')
    
    def inferLabelDiseaseCheckChecks(self,sentence):
        if '高血压' in sentence and '详细检查' in sentence:
            return self.kglni.testLabelDiseaseCheckList('checks')

    def inferLabelDiseaseCheckCollectCount(self,sentence):
        if '高血压' in sentence and ('浏览量' in sentence or '收藏量' in sentence):
            return self.kglni.testLabelDiseaseCollectCount()

    
class InferLabelDiseaseCheckTest(object):
    """docstring for InferLabelTest"""
    def __init__(self):
        self.test = BotKGInferLabelDiseaseCheck()

    def inferLabelDiseaseCheckTest(self):
        diseaseCheck = "高血压疾病诊断知识是什么？"
        return self.test.inferLabelDiseaseCheck(diseaseCheck)

    def inferLabelDiseaseCheckURLTest(self):
        diseaseCheckURL =  '高血压检查URL是那个？'
        return self.test.inferLabelDiseaseCheckURL(diseaseCheckURL)

    def inferLabelDiseaseCheckCommoncheckTest(self):
        diseaseCheckCommoncheck = '高血压有哪些常见检查'
        return self.test.inferLabelDiseaseCheckCommoncheck(diseaseCheckCommoncheck)
    
    def inferLabelDiseaseCheckChecksTest(self):
        diseaseCheckChecks = '高血压详细检查有哪些'
        return self.test.inferLabelDiseaseCheckChecks(diseaseCheckChecks)

    def inferLabelDiseaseCheckCollectCountTest(self):
        diseaseCheckCollectCount =  '高血压检查收藏量有多大？'
        return self.test.inferLabelDiseaseCheckCollectCount(diseaseCheckCollectCount)


def main():
 
    test = InferLabelDiseaseCheckTest()
    test.inferLabelDiseaseCheckTest()
    print(test.inferLabelDiseaseCheckURLTest())
    print(test.inferLabelDiseaseCheckCommoncheckTest())
    print(test.inferLabelDiseaseCheckChecksTest())
    print(test.inferLabelDiseaseCheckCollectCountTest())
    
if __name__ == '__main__':
    main()
    