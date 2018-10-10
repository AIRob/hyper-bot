from bot.knowledge_graph.BotKGInfer import KGLableNamesInfer


class BotKGinferLabelDiseaseComplication(object):
    """docstring for Py2neoConnect"""
    def __init__(self):
        self.kglni = KGLableNamesInfer()
    
    def inferLabelDiseaseComplicationURL(self,sentence):
        if '高血压' in sentence and ('并发症URL' in sentence or '并发症链接' in sentence):
            return  self.kglni.testLabelDiseaseComplicationURL()

    def inferLabelDiseaseComplicationCommonComplication(self,sentence):
        if '高血压' in sentence and ('常见并发症' in sentence or '关联症状' in sentence or '关联疾病' in sentence):
            return self.kglni.testLabelDiseaseComplicationList('common_complication')
    
    def inferLabelDiseaseComplication(self,sentence):
        if '高血压' in sentence and '主要并发症' in sentence:
            return self.kglni.testLabelDiseaseComplicationList('complication')

    def inferLabelDiseaseComplicationDetail(self,sentence):
        if '高血压' in sentence and '详细并发症' in sentence :
            return self.kglni.testLabelDiseaseComplicationList('complication_detail')

    def inferLabelDiseaseComplicationCollectCount(self,sentence):
        if '高血压' in sentence and ('浏览量' in sentence or '收藏量' in sentence):
            return self.kglni.testLabelDiseaseComplicationCollectCount()

    
class inferLabelDiseaseComplicationTest(object):
    """docstring for InferLabelTest"""
    def __init__(self):
        self.test = BotKGinferLabelDiseaseComplication()

    def inferLabelDiseaseComplicationURLTest(self):
        diseaseComplicationURL =  '高血压并发症URL是哪个？'
        return self.test.inferLabelDiseaseComplicationURL(diseaseComplicationURL)

    def inferLabelDiseaseComplicationCommonComplicationTest(self):
        detailComplicationCommonComplication = '高血压常见并发症有哪些？'
        return self.test.inferLabelDiseaseComplicationCommonComplication(detailComplicationCommonComplication)
    
    def inferLabelDiseaseComplicationTest(self):
        diseaseComplication = "高血压疾病主要并发症是什么？"
        return self.test.inferLabelDiseaseComplication(diseaseComplication)

    def inferLabelDiseaseComplicationDetailTest(self):
        detailComplicationDetail = '高血压的详细并发症有哪些？'
        return self.test.inferLabelDiseaseComplicationDetail(detailComplicationDetail)

    def inferLabelDiseaseComplicationCollectCountTest(self):
        detailComplicationCollectCount =  '高血压并发症收藏量有多大？'
        return self.test.inferLabelDiseaseComplicationCollectCount(detailComplicationCollectCount)


def main():

    test = inferLabelDiseaseComplicationTest()
    
    print(test.inferLabelDiseaseComplicationURLTest())
    print(test.inferLabelDiseaseComplicationCommonComplicationTest())
    print(test.inferLabelDiseaseComplicationTest())
    print(test.inferLabelDiseaseComplicationDetailTest())
    print(test.inferLabelDiseaseComplicationCollectCountTest())
    
if __name__ == '__main__':
    main()
    