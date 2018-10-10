from bot.knowledge_graph.BotKGInfer import KGLableNamesInfer


class BotKGInferLabelDiseaseTreat(object):
    """docstring for Py2neoConnect"""
    def __init__(self):
        self.kglni = KGLableNamesInfer()

    def inferLabelDiseaseTreat(self,sentence):
        if '高血压' in sentence and ('治疗' in sentence or '治疗方案' in sentence):
            return self.kglni.testLabelDiseaseTreat()
    
    def inferLabelDiseaseTreatMethod(self,sentence):
        if '高血压' in sentence and ('治疗方法' in sentence or '如何治疗' in sentence):
            return  self.kglni.testLabelDiseaseTreatMethod()

    def inferLabelDiseaseTreatCosts(self,sentence):
        if '高血压' in sentence and ('治疗费用' in sentence or '治疗费' in sentence):
            return self.kglni.testLabelDiseaseTreatCosts()
    
    def inferLabelDiseaseTreatRate(self,sentence):
        if '高血压' in sentence and ('治愈率' in sentence or '治愈' in sentence):
            return self.kglni.testLabelDiseaseTreatRate()

    def inferLabelDiseaseTreatCycle(self,sentence):
        if '高血压' in sentence and ('治疗周期' in sentence or '疗程' in sentence):
            return self.kglni.testLabelDiseaseTreatCycle()

    def inferLabelDiseaseTreatCommondrugs(self,sentence):
        if '高血压' in sentence and ('常用药品' in sentence or '必备药' in sentence):
            return self.kglni.testLabelDiseaseTreatList('common_drugs')

    def inferLabelDiseaseTreatVisitdepartment(self,sentence):
        if '高血压' in sentence and ('就诊科室' in sentence or '科室' in sentence):
            return self.kglni.testLabelDiseaseTreatList('visit_department')
    
class InferLabelDiseaseTreatTest(object):
    """docstring for InferLabelTest"""
    def __init__(self):
        self.test = BotKGInferLabelDiseaseTreat()

    def inferLabelDiseaseTreatTest(self):
        diseaseTreat = "高血压有哪些治疗方案？"
        return self.test.inferLabelDiseaseTreat(diseaseTreat)

    def inferLabelDiseaseTreatMethodTest(self):
        diseaseTreatMeth =  '高血压治疗方法有哪些？'
        return self.test.inferLabelDiseaseTreatMethod(diseaseTreatMeth)

    def inferLabelDiseaseTreatCostsTest(self):
        diseaseTreatCosts = '高血压治疗费用是多少？'
        return self.test.inferLabelDiseaseTreatCosts(diseaseTreatCosts)
    
    def inferLabelDiseaseTreatRateTest(self):
        diseaseTreatRate = '高血压治愈率有多高？'
        return self.test.inferLabelDiseaseTreatRate(diseaseTreatRate)

    def inferLabelDiseaseTreatCycleTest(self):
        diseaseTreatCycle =  '高血压治疗周期有多长？'
        return self.test.inferLabelDiseaseTreatCycle(diseaseTreatCycle)

    def inferLabelDiseaseTreatCommondrugsTest(self):
        diseaseTreatCommondrugs = '高血压常用药品必备药有哪些'
        return self.test.inferLabelDiseaseTreatCommondrugs(diseaseTreatCommondrugs)

    def inferLabelDiseaseTreatVisitdepartmentTest(self):
        diseaseTreatVisitdepartment =  '高血压去哪个科室'
        return self.test.inferLabelDiseaseTreatVisitdepartment(diseaseTreatVisitdepartment)

def main():
 
    test = InferLabelDiseaseTreatTest()
    print(test.inferLabelDiseaseTreatTest())
    print(test.inferLabelDiseaseTreatMethodTest())
    print(test.inferLabelDiseaseTreatCostsTest())
    print(test.inferLabelDiseaseTreatRateTest())
    print(test.inferLabelDiseaseTreatCycleTest())
    print(test.inferLabelDiseaseTreatCommondrugsTest())
    print(test.inferLabelDiseaseTreatVisitdepartmentTest())
    
if __name__ == '__main__':
    main()
    