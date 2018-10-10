from bot.knowledge_graph.BotKGInfer import KGLableNamesInfer


class BotKGInferLabelDiseaseDrug(object):
    """docstring for Py2neoConnect"""
    def __init__(self):
        self.kglni = KGLableNamesInfer()

    def inferLabelDiseaseDrug(self,sentence):
        if '高血压' in sentence and ('药品' in sentence or '治疗药品' in sentence):
            return self.kglni.testLabelDiseaseDrug()
    
    def inferLabelDiseaseDrugList(self,sentence):
        if '高血压' in sentence and '盐酸阿罗洛尔片' in sentence and '疾病' in sentence:
            return self.kglni.testLabelDiseaseDrugList('盐酸阿罗洛尔片')
    
    def inferLabelDiseaseDrugTherapeuticdiseases(self,sentence):
        if '高血压' in sentence and '盐酸阿罗洛尔片' in sentence and '疾病' in sentence:
            return self.kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','therapeutic_diseases')

    def inferLabelDiseaseDrugFunctions(self,sentence):
        if '高血压' in sentence and '盐酸阿罗洛尔片' in sentence and ('功能主治' in sentence or '主治功能' in sentence):
            return self.kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','functions') #null

    def inferLabelDiseaseDrugIngredients(self,sentence):
        if '高血压' in sentence and '盐酸阿罗洛尔片' in sentence and ('成份' in sentence or '成分' in sentence or '组成' in sentence):
            return self.kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','ingredients') #null

    def inferLabelDiseaseDrugAdversereactions(self,sentence):
        if '高血压' in sentence and '盐酸阿罗洛尔片' in sentence and '不良反应' in sentence:
            return self.kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','adverse_reactions')

    def inferLabelDiseaseDrugPrecautions(self,sentence):
        if '高血压' in sentence and '盐酸阿罗洛尔片' in sentence and ('注意事项' in sentence or '注意' in sentence):
            return self.kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','precautions')

    def inferLabelDiseaseDrugTaboo(self,sentence):
        if '高血压' in sentence and '盐酸阿罗洛尔片' in sentence and ('禁忌' in sentence or '药品禁忌' in sentence):
            return self.kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','taboo')

    def inferLabelDiseaseDrugMedicineInteractions(self,sentence):
        if '高血压' in sentence and '盐酸阿罗洛尔片' in sentence and ('药物相互作用' in sentence or '相互作用' in sentence):
            return self.kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','medicine_interactions')

    def inferLabelDiseaseDrugPharmacologicalAction(self,sentence):
        if '高血压' in sentence and '盐酸阿罗洛尔片' in sentence and ('药理作用' in sentence or '药理' in sentence):
            return self.kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','pharmacological_action')

    def inferLabelDiseaseDrugSpecialPopulation(self,sentence):
        if '高血压' in sentence and '盐酸阿罗洛尔片' in sentence and ('特殊人群' in sentence or '特殊人群用药' in sentence or '特殊用药' in sentence):
            return self.kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','special_population')#null

class InferLabelDiseaseDurgTest(object):
    """docstring for InferLabelTest"""
    def __init__(self):
        self.test = BotKGInferLabelDiseaseDrug()

    def inferLabelDiseaseDrugTest(self):
        diseaseDrug = "高血压治疗药品有哪些？"
        return self.test.inferLabelDiseaseDrug(diseaseDrug)

    def inferLabelDiseaseDrugListTest(self):
        diseaseDrugList = '治疗高血压的盐酸阿罗洛尔片还可以治疗哪些疾病？'
        return self.test.inferLabelDiseaseDrugList(diseaseDrugList)

    def inferLabelDiseaseDrugTherapeuticdiseasesTest(self):
        diseaseDrugTherapeuticdiseases = '治疗高血压的盐酸阿罗洛尔片还可以治疗哪些疾病？'
        return self.test.inferLabelDiseaseDrugTherapeuticdiseases(diseaseDrugTherapeuticdiseases)

    def inferLabelDiseaseDrugFunctionsTest(self):
        diseaseDrugFunctions = '治疗高血压的盐酸阿罗洛尔片的主治功能有哪些？'
        return self.test.inferLabelDiseaseDrugTherapeuticdiseases(diseaseDrugFunctions)

    def inferLabelDiseaseDrugIngredientsTest(self):
        diseaseDrugIngredients = '治疗高血压的盐酸阿罗洛尔片的成分有哪些？'
        return self.test.inferLabelDiseaseDrugIngredients(diseaseDrugIngredients)

    def inferLabelDiseaseDrugAdversereactionsTest(self):
        diseaseDrugAdversereactions = '治疗高血压的盐酸阿罗洛尔片有哪些不良反应？'
        return self.test.inferLabelDiseaseDrugAdversereactions(diseaseDrugAdversereactions)

    def inferLabelDiseaseDrugPrecautionsTest(self):
        diseaseDrugPrecautions = '治疗高血压的盐酸阿罗洛尔片有哪些注意事项？'
        return self.test.inferLabelDiseaseDrugPrecautions(diseaseDrugPrecautions)

    def inferLabelDiseaseDrugTabooTest(self):
        diseaseDrugTaboo = '治疗高血压的盐酸阿罗洛尔片有哪些禁忌？'
        return self.test.inferLabelDiseaseDrugTaboo(diseaseDrugTaboo)

    def inferLabelDiseaseDrugMedicineInteractionsTest(self):
        diseaseDrugMedicineInteractions = '治疗高血压的盐酸阿罗洛尔片的药物相互作用是什么？'
        return self.test.inferLabelDiseaseDrugMedicineInteractions(diseaseDrugMedicineInteractions)

    def inferLabelDiseaseDrugPharmacologicalActionTest(self):
        diseaseDrugPharmacologicalAction = '治疗高血压的盐酸阿罗洛尔片的药理作用是什么？'
        return self.test.inferLabelDiseaseDrugPharmacologicalAction(diseaseDrugPharmacologicalAction)

    def inferLabelDiseaseDrugSpecialPopulationTest(self):
        diseaseDrugSpecialPopulation = '治疗高血压的盐酸阿罗洛尔片特殊人群用药有哪些？'
        return self.test.inferLabelDiseaseDrugSpecialPopulation(diseaseDrugSpecialPopulation)


def main():
    test = InferLabelDiseaseDurgTest()
    print(test.inferLabelDiseaseDrugTest())
    print(test.inferLabelDiseaseDrugListTest())
    print(test.inferLabelDiseaseDrugTherapeuticdiseasesTest())
    print(test.inferLabelDiseaseDrugFunctionsTest())
    print(test.inferLabelDiseaseDrugIngredientsTest())
    print(test.inferLabelDiseaseDrugAdversereactionsTest())
    print(test.inferLabelDiseaseDrugPrecautionsTest())
    print(test.inferLabelDiseaseDrugTabooTest())
    print(test.inferLabelDiseaseDrugMedicineInteractionsTest())
    print(test.inferLabelDiseaseDrugPharmacologicalActionTest())
    print(test.inferLabelDiseaseDrugSpecialPopulationTest())
    
if __name__ == '__main__':
    main()
    