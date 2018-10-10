import py2neo
from py2neo import Graph

class Py2neoConnect(object):
    """docstring for Py2neoConnect"""
    def __init__(self):
        py2neo.authenticate("localhost:7474", "neo4j", "airob")
        # Connect to Graph and get the instance of Graph
        graph = Graph("http://localhost:7474/db/data/")
        self.graph = graph
        

class KGLableNamesInfer(Py2neoConnect):    
    """docstring for KGLableNamesInfer"""
    def testLabelDiseaseName(self):
        #results = self.graph.cypher.execute("MATCH (n{name:'疾病基本知识'}) RETURN n as labelName")
        #results = self.graph.cypher.execute("MATCH (n:disease_base) RETURN n.name as labelName")
        #results = self.graph.cypher.execute("MATCH (n:disease_profile) RETURN n.name as labelName")
        #results = self.graph.cypher.execute(\
        #"MATCH (n:disease_base)-[*1..3]->(n2:complication) return n2.name as labelComplicationName")
        #results = self.graph.cypher.execute(\
        #"MATCH (n:disease_base)-[re]->(n2:disease_alias) return n2.name as labelName")

        results = self.graph.cypher.execute("MATCH (n:disease_name{name:'高血压'}) RETURN n.name as labelName")
        #print(results)
        #print(results[0].labelName)
        return results[0].labelName

    def testLabelDiseaseId(self):
        results = self.graph.cypher.execute("MATCH (n:disease_name{name:'高血压'})-[re]->(n2:disease_id) RETURN n2.name as labelName")
        #print(results)
        #print(results[0].labelName)
        return results[0].labelName

    #1 merge
    def testLabelDiseaseProfile(self):
        results = self.graph.cypher.execute("MATCH (n:disease_name{name:'高血压'})-[*1..2]->(n2:disease_profile) return n2.name as labelName")
        #print(results)
        #print(results[0].labelName)
        labelNameList = []
        for i in range(len(results)):
            labelNameList.append(results[i].labelName)
        #print('、'.join(labelNameList))
        return ('、'.join(labelNameList)) 

    ##0 merge
    def testLabelDiseaseBase(self):
        #results = self.graph.cypher.execute("MATCH (n{name:'疾病基本知识'}) RETURN n as labelName")
        results = self.graph.cypher.execute("MATCH (n:disease_base) RETURN n.name as labelName")
        #results = self.graph.cypher.execute("MATCH (n:disease_profile) RETURN n.name as labelName")
        #results = self.graph.cypher.execute(\
        #    "MATCH (n:disease_base)-[*1..3]->(n2:complication) return n2.name as labelComplicationName")
        #results = self.graph.cypher.execute(\
        #    "MATCH (n:disease_base)-[re]->(n2:disease_alias) return n2.name as labelName")
        #print(results)
        #print(results[0].labelName)
        labelNameList = []
        for i in range(len(results)-1):
            labelNameList.append(results[i].labelName)
        #print('、'.join(labelNameList)+'、典型症状')
        return ('、'.join(labelNameList)+'、典型症状')

    #3 merge
    def testLabelDiseaseIsmedical(self):
        results = self.graph.cypher.execute("MATCH (n:disease_base) RETURN n.name as labelName")        
        #print(results[2])
        #print(results[2].labelName)
        return (results[2].labelName)
    
    def testLabelDiseaseBaseMultiplePeople(self):
        results = self.graph.cypher.execute("MATCH (n:disease_base) RETURN n.name as labelName")        
        #print(results[2])
        #print(results[3].labelName)
        return (results[3].labelName) 
    
    def testLabelDiseaseContagious(self):
        results = self.graph.cypher.execute("MATCH (n:disease_base) RETURN n.name as labelName")        
        #print(results[5])
        #print(results[5].labelName)
        return (results[5].labelName)

    def testLabelDiseaseBaseTypicalSymptoms(self):
        #results = self.graph.cypher.execute("MATCH (n{name:'疾病基本知识'}) RETURN n as labelName")
        #results = self.graph.cypher.execute("MATCH (n:disease_base) RETURN n.name as labelName")
        #results = self.graph.cypher.execute("MATCH (n:disease_profile) RETURN n.name as labelName")
        #results = self.graph.cypher.execute(\
        #    "MATCH (n:disease_base)-[*1..3]->(n2:complication) return n2.name as labelComplicationName")
        results = self.graph.cypher.execute(\
            "MATCH (n:disease_base)-[re]->(n2:typical_symptoms) return n2.name as labelName")
        #print(results)
        #print(results[0].labelName)
        labelNameList = []
        for i in range(len(results)):
            labelNameList.append(results[i].labelName.strip())
        #print('、'.join(labelNameList))
        return ('、'.join(labelNameList))

    def testLabelDiseaseBaseComplication(self):
        #results = self.graph.cypher.execute("MATCH (n{name:'疾病基本知识'}) RETURN n as labelName")
        #results = self.graph.cypher.execute("MATCH (n:disease_base) RETURN n.name as labelName")
        #results = self.graph.cypher.execute("MATCH (n:disease_profile) RETURN n.name as labelName")
        #results = self.graph.cypher.execute(\
        #    "MATCH (n:disease_base)-[*1..3]->(n2:complication) return n2.name as labelComplicationName")
        results = self.graph.cypher.execute(\
            "MATCH (n:disease_base)-[re]->(n2:complication) return n2.name as labelName")
        #print(results)
        #print(results[0].labelName)
        labelNameList = []
        for i in range(len(results)):
            labelNameList.append(results[i].labelName)
        #print('、'.join(labelNameList))
        return ('、'.join(labelNameList))

    #2 merge
    def testLabelDiseaseBaseList(self,listname):
        #results = self.graph.cypher.execute("MATCH (n{name:'疾病基本知识'}) RETURN n as labelName")
        #results = self.graph.cypher.execute("MATCH (n:disease_base) RETURN n.name as labelName")
        #results = self.graph.cypher.execute("MATCH (n:disease_profile) RETURN n.name as labelName")
        #results = self.graph.cypher.execute(\
        #    "MATCH (n:disease_base)-[*1..3]->(n2:complication) return n2.name as labelComplicationName")
        results = self.graph.cypher.execute(\
            f"MATCH (n:disease_base)-[re]->(n2:{listname}) return n2.name as labelName")
        #print(results)
        #print(results[0].labelName)
        labelNameList = []
        for i in range(len(results)):
            labelNameList.append(results[i].labelName.strip())
        #print('、'.join(labelNameList))
        return ('、'.join(labelNameList))

    #0 merge 
    def testLabelDiseaseDiagnosis(self):
        results = self.graph.cypher.execute("MATCH (n:disease_diagnosis) RETURN n.name as labelName")
        #print(results)
        labelNameList = []
        for i in range(len(results)):
            labelNameList.append(results[i].labelName)
        #print('、'.join(labelNameList))
        return ('、'.join(labelNameList))

    #2 merge
    def testLabelDiseaseDiagnosisList(self,listname):
        results = self.graph.cypher.execute(\
            f"MATCH (n:disease_diagnosis)-[re]->(n2:{listname}) return n2.name as labelName")
        #print(results)
        #print(results[0].labelName)
        labelNameList = []
        for i in range(len(results)):
            labelNameList.append(results[i].labelName.strip())
        #print('、'.join(labelNameList))
        #return ('、'.join(labelNameList))
        return ('、'.join(labelNameList))

    #0 merge 
    def testLabelDiseaseCheck(self):
        results = self.graph.cypher.execute("MATCH (n:disease_check) RETURN n.name as labelName")
        print(results)
        labelNameList = []
        for i in range(len(results)):
            labelNameList.append(results[i].labelName)
        #print('、'.join(labelNameList))
        return ('、'.join(labelNameList))

    #2 merge
    def testLabelDiseaseCheckList(self,listname):
        results = self.graph.cypher.execute(\
            f"MATCH (n:disease_check)-[re]->(n2:{listname}) return n2.name as labelName")
        #print(results)
        #print(results[0].labelName)
        labelNameList = []
        for i in range(len(results)):
            labelNameList.append(results[i].labelName)
        #print('、'.join(labelNameList))
        return ('、'.join(labelNameList))

    #3 merge
    def testLabelDiseaseCheckURL(self):
        results = self.graph.cypher.execute("MATCH (n:disease_check) RETURN n.name as labelName")        
        #print(results[2])
        #print(results[0].labelName)
        return (results[0].labelName)

    #3 merge
    def testLabelDiseaseCollectCount(self):
        results = self.graph.cypher.execute("MATCH (n:disease_check) RETURN n.name as labelName")        
        #print(results[2])
        #print(results[5].labelName)
        return (results[5].labelName)
    
    #高血压症状
    #0 merge 
    def testLabelDiseaseSymptoms(self):
        results = self.graph.cypher.execute("MATCH (n:disease_symptoms) RETURN n.name as labelName")
        print(results)
        labelNameList = []
        for i in range(len(results)):
            labelNameList.append(results[i].labelName)
        #print('、'.join(labelNameList))
        return ('、'.join(labelNameList))

    #4 merge
    def testLabelDiseaseSymptomsList(self,listname):
        results = self.graph.cypher.execute(\
            f"MATCH (n:disease_symptoms)-[*1..3]->(n2:{listname}) return n2.name as labelName")
        #print(results)
        #print(results[0].labelName)
        labelNameList = []
        for i in range(len(results)):
            labelNameList.append(results[i].labelName)
        #print('、'.join(labelNameList))
        return ('、'.join(labelNameList))

    #3 merge
    def testLabelDiseaseSymptomsURL(self):
        results = self.graph.cypher.execute("MATCH (n:disease_check) RETURN n.name as labelName")        
        #print(results[2])
        #print(results[0].labelName)
        return (results[0].labelName)

    #3 merge
    def testLabelDiseaseSymptomsCollectCount(self):
        results = self.graph.cypher.execute("MATCH (n:disease_check) RETURN n.name as labelName")        
        #print(results[2])
        #print(results[5].labelName)
        return (results[5].labelName)

    #高血压并发症症状
    #testLabelDiseaseComplication()
    #0 merge 
    def testLabelDiseaseComplication(self):
        results = self.graph.cypher.execute("MATCH (n:complication) RETURN n.name as labelName")
        #print(results)
        labelNameList = []
        for i in range(len(results)):
            labelNameList.append(results[i].labelName.strip())
        #print('、'.join(labelNameList))
        return ('、'.join(labelNameList))

    #4 merge
    def testLabelDiseaseComplicationList(self,listname):
        results = self.graph.cypher.execute(\
            f"MATCH (n:complication)-[*1..3]->(n2:{listname}) return n2.name as labelName")
        #print(results)
        #print(results[0].labelName)
        labelNameList = []
        for i in range(len(results)):
            labelNameList.append(results[i].labelName.strip())
        #print('、'.join(labelNameList))
        return ('、'.join(labelNameList))
    
    def testLabelDiseaseComplicationLinksSymptoms(self):
        results = self.graph.cypher.execute(\
            "MATCH (n:complication)-[*1..3]->(n2:common_complication) return n2.name as labelName")
        #print(results)
        #print(results[0].labelName)
        labelNameList = []
        for i in range(len(results)):
            labelNameList.append(results[i].labelName.strip())
        #print('、'.join(labelNameList))
        return ('、'.join(labelNameList[:7]))

    #3 merge
    def testLabelDiseaseComplicationURL(self):
        results = self.graph.cypher.execute("MATCH (n:complication) RETURN n.name as labelName")        
        #print(results[2])
        #print(results[7].labelName)
        return (results[7].labelName)

    #3 merge
    def testLabelDiseaseComplicationCollectCount(self):
        results = self.graph.cypher.execute("MATCH (n:complication) RETURN n.name as labelName")        
        #print(results[2])
        #print(results[11].labelName)
        return (results[11].labelName)

    #0 merge 
    def testLabelDiseaseTreat(self):
        results = self.graph.cypher.execute("MATCH (n:disease_treat) RETURN n.name as labelName")
        print(results)
        labelNameList = []
        for i in range(len(results)):
            labelNameList.append(results[i].labelName)
        #print('、'.join(labelNameList))
        return ('、'.join(labelNameList))

    #2 merge
    def testLabelDiseaseTreatList(self,listname):
        results = self.graph.cypher.execute(\
            f"MATCH (n:disease_treat)-[re]->(n2:{listname}) return n2.name as labelName")
        #print(results)
        #print(results[0].labelName)
        labelNameList = []
        for i in range(len(results)):
            labelNameList.append(results[i].labelName)
        #print('、'.join(labelNameList))
        return ('、'.join(labelNameList))
    
    #other 
    def testLabelDiseaseTreatMethod(self):
        url = 'http://jbk.39.net/gxy/yyzl/'
        #print(url)
        return url

    #3 merge
    def testLabelDiseaseTreatCosts(self):
        results = self.graph.cypher.execute("MATCH (n:disease_treat) RETURN n.name as labelName")        
        #print(results[2])
        #print(results[2].labelName.strip())
        return (results[2].labelName.strip())

    def testLabelDiseaseTreatRate(self):
        results = self.graph.cypher.execute("MATCH (n:disease_treat) RETURN n.name as labelName")        
        #print(results[2])
        ##print(results[1].labelName.strip())
        return (results[1].labelName.strip())

    def testLabelDiseaseTreatCycle(self):
        results = self.graph.cypher.execute("MATCH (n:disease_treat) RETURN n.name as labelName")        
        #print(results[2])
        #print(results[3].labelName)
        return (results[3].labelName)

    #1 merge
    def testLabelDiseaseCause(self):
        results = self.graph.cypher.execute("MATCH (n:disease_name{name:'高血压'})-[*1..2]->(n2:disease_cause) return n2.name as labelName")
        #print(results)
        #print(results[0].labelName)
        labelNameList = []
        for i in range(len(results)):
            labelNameList.append(results[i].labelName)
        #print('、'.join(labelNameList))
        return ('、'.join(labelNameList)) 

    #1 merge
    def testLabelDiseaseHowPrevent(self):
        results = self.graph.cypher.execute("MATCH (n:disease_name{name:'高血压'})-[*1..2]->(n2:how_prevent) return n2.name as labelName")
        #print(results)
        #print(results[0].labelName)
        labelNameList = []
        for i in range(len(results)):
            labelNameList.append(results[i].labelName)
        #print('、'.join(labelNameList))
        #return (set(labelNameList))
        return ('、'.join(labelNameList))

    #qa_corpus
    def testLabelDiseaseQAcorpus(self):
        results = self.graph.cypher.execute(\
            "MATCH (n:qa_corpus)-[re]->(n2:sub_qa_data) return n2.name as labelName")
        labelNameList = []
        for i in range(len(results)):
            labelNameList.append(results[i].labelName)
        #print(' '.join(set(labelNameList)))
        return ('、'.join(set(labelNameList)))

    def testLabelDiseaseDrug(self):
        results = self.graph.cypher.execute(\
            "match data=(n0:disease_name{name:'高血压'})-[*1..5]->(n1:drugsinfo{name:'盐酸阿罗洛尔片'})\
            -[*1..3]->(n2:therapeutic_diseases) return data as labelPath")
        #print(len(results))
        #print(results[0].labelPath)
        #print(str(results[0].labelPath).split('\"')[-2])

        pathList = []
        for i in range(len(results)):
            tmp = str(results[i].labelPath).split('\"')[-2]
            pathList.append(tmp)
        #print(set(pathList))
        return (' '.join(set(pathList)))

    def testLabelDiseaseDrugList(self,drugname):
        results = self.graph.cypher.execute(\
            f"match data=(n0:disease_name)-[*1..5]->(n1:drugsinfo)\
            -[*1..3]->(n2:therapeutic_diseases) \
            where n0.name='高血压' and n1.name=\'{drugname}\'\
            return data as labelPath")
        '''
        cql = "match data=(n0:disease_name)-[*1..5]->(n1:drugsinfo)\
            -[*1..3]->(n2:therapeutic_diseases) \
            where n0.name='高血压' and n1.name=\'{}\'\
            return data as labelPath".format(drugname)
        results = self.graph.cypher.execute(cql)
        '''  
        #print(len(results))
        #print(results[0].labelPath)
        #print(str(results[0].labelPath).split('\"')[-2])

        pathList = []
        for i in range(len(results)):
            tmp = str(results[i].labelPath).split('\"')[-2]
            pathList.append(tmp)
        #print(set(pathList))
        return (' '.join(set(pathList)))

    def testLabelDiseaseDrugDepthList(self,drugname,outname):
        results = self.graph.cypher.execute(\
            f"match data=(n0:disease_name)-[*1..5]->(n1:drugsinfo)\
            -[*1..3]->(n2:{outname}) \
            where n0.name='高血压' and n1.name=\'{drugname}\'\
            return data as labelPath")
        '''
        cql = "match data=(n0:disease_name)-[*1..5]->(n1:drugsinfo)\
            -[*1..3]->(n2:therapeutic_diseases) \
            where n0.name='高血压' and n1.name=\'{}\'\
            return data as labelPath".format(drugname)
        results = self.graph.cypher.execute(cql)
        '''  
        pathList = []
        for i in range(len(results)):
            tmp = str(results[i].labelPath).split('\"')[-2]
            pathList.append(tmp)
        #print(set(pathList))
        return ('、'.join(set(pathList)))

    def testLabelDiseaseDrugDepthXList(self,drugname,outname):
        #results = self.graph.cypher.execute(\
        #    f"MATCH (n1:disease_name)-[*1..2]->(n2:{outname}) \
        #    where n1.name=\'{drugname}\'  return n2.name as labelName")
        
        cql = "MATCH (n1:drugsinfo)-[*1..5]->(n2:{1}) \
            where n1.name=\'{0}\' return n2.name as labelName".format(drugname,outname)
        print(cql)
        results = self.graph.cypher.execute(cql)
         
        pathList = []
        for i in range(len(results)):
            tmp = str(results[i].labelPath).split('\"')[-2]
            pathList.append(tmp)
        print(set(pathList))

    def testLabelTreat(self):
        results = self.graph.cypher.execute(\
            "match data=(n1:disease_name{name:'高血压'})-[re]->(n2:disease_treat) return data as treat")
        print(len(results))
        print(results[0].treat)
        print(str(results[0].treat).split('\"')[-2])

    def testLabelCount(self):
        results = self.graph.cypher.execute("MATCH (n) return count(DISTINCT labels(n)) as countLabel")
        print(results[0].countLabel)

    def testIndividualNodes(self):
        #Define a Node which we need to check
        return 'Individual Nodes'

    def testPathExists(self):
        #Query whether there are Nodes linked with Relationship
        results = self.graph.cypher.execute("MATCH data = (n0:disease_name{name:'高血压'})-[*1..5]->(n1:drugsinfo{name:'盐酸阿罗洛尔片'}) return count(data) as countPath")
        #Ensure that count=1. Not more, neither less
        #self.assertEqual(4,results[0].countPath)
        print(results[0].countPath)

    def testPathDiseaseDrugsExists(self):
        #Query whether there are Nodes linked with Relationship - drugsinfo
        #match data=(n0:disease_name{name:"高血压"})-[*1..5]->(n1:drugsinfo{name:"盐酸阿罗洛尔片"})-[*1..3]->(n2:therapeutic_diseases) return data
        #
        #(n1:drugsinfo{name:'盐酸阿罗洛尔片'})-[*1..3]->(n2:therapeutic_diseases) return count(r) as countPath
        #MATCH (n0:disease_name{name:'高血压'})-[*1..5]->(n1:drugsinfo{name:'盐酸阿罗洛尔片'})-[*1..3]->(n2:therapeutic_diseases) return count(r) as countPath
        #MATCH (n1:drugsinfo{name:'盐酸阿罗洛尔片'})-[*1..3]->(n2:therapeutic_diseases)
        results = self.graph.cypher.execute("MATCH data = (n0:disease_name{name:'高血压'})-[*1..5]->(n1:drugsinfo{name:'盐酸阿罗洛尔片'})-[*1..3]->(n2:therapeutic_diseases) return data as countPath")
        #Ensure that count=1. Not more, neither less
        print(results[0].countPath)
        print(len(results))

class KGLableCountInfer(Py2neoConnect):
    """docstring for KGLableCountInfer"""
    def testLabelCount(self):
        results = self.graph.cypher.execute("MATCH (n) return count(DISTINCT labels(n)) as countLabel")
        #We should have only 5 Labels "FEMALE(2) MALE(4) MOVIE(3) STUDENT(1) TEACHER(1)"
        print(results[0].countLabel)


class KGIndividualNodesInfer(Py2neoConnect):
    """docstring for KGIndividualNodesInfer"""
    def testIndividualNodes(self):
        #Define a Node which we need to check
        '''
        bradley = Node("disease_id",'TEACHER',name='Bradley', surname='Green', age=24, country='US')
        #Now get the Node from server
        results = self.graph.cypher.execute("MATCH (n) where n.name='Bradley' return n as bradley")
        #Both Nodes should be equal
        self.assertEqual(results[0].bradley,bradley)
        '''
        return 'Individual Nodes'


class KGPathExistsInfer(Py2neoConnect):
    """docstring for KGPathExistsInfer"""
    def testPathExists(self):
        #Query whether there are Nodes linked with Relationship - FRIEND
        results = self.graph.cypher.execute("MATCH data = (n0:disease_name{name:'高血压'})-[*1..5]->(n1:drugsinfo{name:'盐酸阿罗洛尔片'}) return count(data) as countPath")
        #Ensure that count=1. Not more, neither less
        #self.assertEqual(4,results[0].countPath)
        print(results[0].countPath)


class KGPathDiseaseDrugsExistsInfer(Py2neoConnect):
    """docstring for KGPathDiseaseDrugsExistsInfer"""      
    def testPathDiseaseDrugsExists(self):
        #Query whether there are Nodes linked with Relationship - TEACHES
        #match data=(n0:disease_name{name:"高血压"})-[*1..5]->(n1:drugsinfo{name:"盐酸阿罗洛尔片"})-[*1..3]->(n2:therapeutic_diseases) return data
        #
        #(n1:drugsinfo{name:'盐酸阿罗洛尔片'})-[*1..3]->(n2:therapeutic_diseases) return count(r) as countPath
        #MATCH (n0:disease_name{name:'高血压'})-[*1..5]->(n1:drugsinfo{name:'盐酸阿罗洛尔片'})-[*1..3]->(n2:therapeutic_diseases) return count(r) as countPath
        #MATCH (n1:drugsinfo{name:'盐酸阿罗洛尔片'})-[*1..3]->(n2:therapeutic_diseases)
        results = self.graph.cypher.execute("MATCH data = (n0:disease_name{name:'高血压'})-[*1..5]->(n1:drugsinfo{name:'盐酸阿罗洛尔片'})-[*1..3]->(n2:therapeutic_diseases) return data as countPath")
        #Ensure that count=1. Not more, neither less
        print(results[0].countPath)

        print(len(results))

def main():
    #KGLableNamesInfer
    print('****************Labels Name******************')
    kglni = KGLableNamesInfer()
    '''
    kglni.testLabelDiseaseName()
    kglni.testLabelDiseaseProfile()
    kglni.testLabelDiseaseBase()
    kglni.testLabelDiseaseId()
    kglni.testLabelDiseaseBaseList('disease_alias')
    kglni.testLabelDiseaseIsmedical()
    kglni.testLabelDiseaseBaseList('incidence_site')
    kglni.testLabelDiseaseContagious()
    kglni.testLabelDiseaseBaseMultiplePeople()
    #kglni.testLabelDiseaseBaseTypicalSymptoms()
    #kglni.testLabelDiseaseBaseComplication()
    kglni.testLabelDiseaseBaseList('typical_symptoms')
    kglni.testLabelDiseaseBaseList('complication')
    
    kglni.testLabelDiseaseDiagnosis()

    print(kglni.testLabelDiseaseDiagnosisList('best_time'))
    print(kglni.testLabelDiseaseDiagnosisList('duration_visit'))
    print(kglni.testLabelDiseaseDiagnosisList('followup_freq'))
    print(kglni.testLabelDiseaseDiagnosisList('pre_treat'))
    
    kglni.testLabelDiseaseCheck()
    kglni.testLabelDiseaseCheckURL()
    kglni.testLabelDiseaseCheckList('common_check')
    kglni.testLabelDiseaseCheckList('checks')
    kglni.testLabelDiseaseCollectCount()
    
    kglni.testLabelDiseaseSymptoms()
    kglni.testLabelDiseaseSymptomsURL()
    kglni.testLabelDiseaseSymptomsList('common_symptoms')
    kglni.testLabelDiseaseSymptomsList('links_symptoms')
    kglni.testLabelDiseaseSymptomsList('symptoms')
    kglni.testLabelDiseaseSymptomsCollectCount()

    
    kglni.testLabelDiseaseComplication()
    print(kglni.testLabelDiseaseComplicationURL())
    print('****')
    print(kglni.testLabelDiseaseComplicationList('common_complication'))
    print('****')
    print(kglni.testLabelDiseaseComplicationList('links_symptoms'))
    print(kglni.testLabelDiseaseComplicationLinksSymptoms())
    print('****')
    print(kglni.testLabelDiseaseComplicationList('complication'))
    print('****')
    print(kglni.testLabelDiseaseComplicationList('complication_detail'))
    print('****')
    print(kglni.testLabelDiseaseComplicationCollectCount())
    
    kglni.testLabelDiseaseTreat()
    kglni.testLabelDiseaseTreatMethod()
    kglni.testLabelDiseaseTreatList('treat_method') #null
    kglni.testLabelDiseaseTreatCosts()
    kglni.testLabelDiseaseTreatRate()
    kglni.testLabelDiseaseTreatCycle()
    kglni.testLabelDiseaseTreatList('common_drugs')
    kglni.testLabelDiseaseTreatList('visit_department')
    '''
    kglni.testLabelDiseaseDrug()
    kglni.testLabelDiseaseDrugList('盐酸阿罗洛尔片')
    print(kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','therapeutic_diseases'))
    ####################################
    kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','goods_common_name') #''
    kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','goods_name') #''
    kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','approval_rum') #''
    ####################################
    kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','indication')
    ####################################
    kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','functions') #null
    kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','ingredients') #null
    ####################################
    kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','adverse_reactions')
    kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','precautions')
    kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','taboo')
    kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','medicine_interactions')
    kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','pharmacological_action')
    ####################################
    kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','special_population')#null
    ####################################
    kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','dosage') #''
    kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','storage')
    kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','validity_period')
    kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','drug_form')
    kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','drug_spec')
    kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','manual_revision_date')
    kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','manufacturer')
    kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','business_short_name')
    kglni.testLabelDiseaseDrugDepthList('盐酸阿罗洛尔片','validproduction_address')
    #kglni.testLabelDiseaseDrugDepthXList('盐酸阿罗洛尔片','business_number')
    '''
    kglni.testLabelDiseaseCause()
    kglni.testLabelDiseaseHowPrevent()
    kglni.testLabelDiseaseQAcorpus()
    '''

if __name__ == '__main__':
    main()
