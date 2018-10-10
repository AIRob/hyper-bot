from bot.knowledge_graph.BotKGInfer import KGLableNamesInfer


class BotKGInferLabelDiseaseBase(object):
    """docstring for Py2neoConnect"""
    def __init__(self):
        self.kglni = KGLableNamesInfer()

    def inferLabelDiseaseProfile(self,sentence):
        if '高血压' in sentence and '简介' in sentence:
            return self.kglni.testLabelDiseaseProfile()

    def inferLabelDiseaseBase(self,sentence):
        if '高血压' in sentence and '基本知识' in sentence:
            return self.kglni.testLabelDiseaseBase()
    
    def inferLabelDiseaseId(self,sentence):
        if '高血压' in sentence and '编号id' in sentence:
            return self.kglni.testLabelDiseaseId()

    def inferLabelDiseaseBaseAliasX(self,sentence):
        if '高血压' in sentence and ('别名' in sentence or '其他叫法' in sentence):
            return self.kglni.testLabelDiseaseBaseList('disease_alias')
    
    def inferLabelDiseaseBaseAlias(self):
        return self.kglni.testLabelDiseaseBaseList('disease_alias')

    def inferLabelDiseaseBaseIsmedical(self,sentence):
        if '高血压' in sentence and '医保' in sentence:
            return self.kglni.testLabelDiseaseIsmedical()
    
    def inferLabelDiseaseBaseIncidenceSite(self,sentence):
        if '高血压' in sentence and ('发病部位' in sentence or '发病部位' in sentence):
            return self.kglni.testLabelDiseaseBaseList('incidence_site')

    def inferLabelDiseaseBaseContagious(self,sentence):
        if '高血压' in sentence and ('传染性' in sentence or '传染' in sentence):
            return self.kglni.testLabelDiseaseContagious()

    def inferLabelDiseaseBaseMultiplePeople(self,sentence):
        if '高血压' in sentence and ('多发人群' in sentence or '多发性' in sentence):
            mulPeopleStr =  self.kglni.testLabelDiseaseBaseList('multiple_people')
            metadata = '中老年人，平时钠盐的摄入量过多的人，父母患有高血压者，摄入动物脂肪较多者，长期饮'
            if mulPeopleStr == metadata:return mulPeopleStr
            else:return metadata

    def inferLabelDiseaseBaseTypicalSymptoms(self,sentence):
        if '高血压' in sentence and '典型症状' in sentence:
            return self.kglni.testLabelDiseaseBaseList('typical_symptoms')

    def inferLabelDiseaseBaseComplication(self,sentence):
        if '高血压' in sentence and '并发症' in sentence:
            return self.kglni.testLabelDiseaseBaseList('complication')



class InferLabelDiseaseBaseTest(object):
    """docstring for InferLabelTest"""
    def __init__(self):
        self.test = BotKGInferLabelDiseaseBase()
        
    def inferLabelDiseaseProfileTest(self):
        diseaseProfile = "高血压疾病简介是什么？"
        return self.test.inferLabelDiseaseProfile(diseaseProfile)

    def inferLabelDiseaseBaseTest(self):
        diseaseBase = "高血压疾病基本知识是什么？"
        return self.test.inferLabelDiseaseBase(diseaseBase)

    def inferLabelDiseaseIdTest(self):
        diseaseId = '高血压疾病编号是多少？'
        return self.test.inferLabelDiseaseId(diseaseId)

    def inferLabelDiseaseBaseAliasTest(self):
        diseaseAlias =  '高血压别名或其他叫法有哪些'
        return self.test.inferLabelDiseaseBaseAlias(diseaseAlias)

    def inferLabelDiseaseBaseIsmedicalTest(self):
        diseaseIsmedical = '高血压是医保疾病？'
        return self.test.inferLabelDiseaseBaseIsmedical(diseaseIsmedical)
    
    def inferLabelDiseaseBaseIncidenceSiteTest(self):
        diseaseIncidenceSite = '高血压一般发病部位在哪儿'
        return self.test.inferLabelDiseaseBaseIncidenceSite(diseaseIncidenceSite)

    def inferLabelDiseaseBaseContagiousTest(self):
        diseaseContagious =  '高血压有无传染性'
        return self.test.inferLabelDiseaseBaseContagious(diseaseContagious)

    def inferLabelDiseaseBaseMultiplePeopleTest(self):
        diseaseMultiplePeople = '高血压多发人群有哪些人？'
        return self.test.inferLabelDiseaseBaseMultiplePeople(diseaseMultiplePeople)

    def inferLabelDiseaseBaseTypicalSymptomsTest(self):
        diseaseTypicalSymptoms = '高血压典型症状有哪些'
        return self.test.inferLabelDiseaseBaseTypicalSymptoms(diseaseTypicalSymptoms)

    def inferLabelDiseaseBaseComplicationTest(self):
        diseaseComplication = '高血压有哪些并发症？'
        return self.test.inferLabelDiseaseBaseComplication(diseaseComplication)

def main():
    test = InferLabelDiseaseBaseTest()
    print(test.inferLabelDiseaseIdTest())
    print(test.inferLabelDiseaseProfileTest())
    print(test.inferLabelDiseaseBaseTest())
    print(test.inferLabelDiseaseBaseAliasTest())
    print(test.inferLabelDiseaseBaseIsmedicalTest())
    print(test.inferLabelDiseaseBaseIncidenceSiteTest())
    print(test.inferLabelDiseaseBaseContagiousTest())
    print(test.inferLabelDiseaseBaseMultiplePeopleTest())
    print(test.inferLabelDiseaseBaseTypicalSymptomsTest())
    print(test.inferLabelDiseaseBaseComplicationTest())

if __name__ == '__main__':

    main()