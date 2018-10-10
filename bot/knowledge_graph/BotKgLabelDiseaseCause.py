from bot.knowledge_graph.BotKGInfer import KGLableNamesInfer


class BotKGInferLabelDiseaseCause(object):
    """docstring for Py2neoConnect"""
    def __init__(self):
        self.kglni = KGLableNamesInfer()

    def inferLabelDiseaseCause(self,sentence):
        if '高血压' in sentence and ('病因' in sentence or '原因' in sentence):
            return self.kglni.testLabelDiseaseCause()
    
    
class InferLabelDiseaseCauseTest(object):
    """docstring for InferLabelTest"""
    def __init__(self):
        self.test = BotKGInferLabelDiseaseCause()

    def inferLabelDiseaseCauseTest(self):
        diseaseCause = "高血压疾病病因是什么？"
        return self.test.inferLabelDiseaseCause(diseaseCause)


def main():
    test = InferLabelDiseaseCauseTest()
    print(test.inferLabelDiseaseCauseTest())
    
if __name__ == '__main__':
    main()
    