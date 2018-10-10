from bot.knowledge_graph.BotKGInfer import KGLableNamesInfer


class BotKGInferLabelDiseaseQAcorpus(object):
    """docstring for Py2neoConnect"""
    def __init__(self):
        self.kglni = KGLableNamesInfer()

    def inferLabelDiseaseQAcorpus(self,sentence):
        if '高血压' in sentence and ('问诊' in sentence or '咨询' in sentence):
            return self.kglni.testLabelDiseaseQAcorpus()
    
    
class InferLabelDiseaseQAcorpusTest(object):
    """docstring for InferLabelTest"""
    def __init__(self):
        self.test = BotKGInferLabelDiseaseQAcorpus()

    def inferLabelDiseaseQAcorpusTest(self):
        diseaseQAcorpus = "高血压疾病到哪儿咨询？"
        return self.test.inferLabelDiseaseQAcorpus(diseaseQAcorpus)


def main():
    test = InferLabelDiseaseQAcorpusTest()
    print(test.inferLabelDiseaseQAcorpusTest())
    
if __name__ == '__main__':
    main()
    