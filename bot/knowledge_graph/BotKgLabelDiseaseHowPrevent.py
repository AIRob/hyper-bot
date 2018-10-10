from bot.knowledge_graph.BotKGInfer import KGLableNamesInfer


class BotKGInferLabelDiseaseHowPrevent(object):
    """docstring for Py2neoConnect"""
    def __init__(self):
        self.kglni = KGLableNamesInfer()

    def inferLabelDiseaseHowPrevent(self,sentence):
        if '高血压' in sentence and ('预防' in sentence or '如何预防' in sentence):
            return self.kglni.testLabelDiseaseHowPrevent()
    
    
class InferLabelDiseaseHowPreventTest(object):
    """docstring for InferLabelTest"""
    def __init__(self):
        self.test = BotKGInferLabelDiseaseHowPrevent()

    def inferLabelDiseaseHowPreventTest(self):
        diseaseHowPrevent = "高血压疾病该怎么预防？"
        return self.test.inferLabelDiseaseHowPrevent(diseaseHowPrevent)


def main():
    test = InferLabelDiseaseHowPreventTest()
    print(test.inferLabelDiseaseHowPreventTest())
    
if __name__ == '__main__':
    main()
    