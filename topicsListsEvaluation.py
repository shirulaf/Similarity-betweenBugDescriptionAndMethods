import json
import csv

import matplotlib.pyplot as plt
import statistics
import numpy as np


data_directory = './dataFiles/'
modelOutput_directory = './topicModel_outputFiles/'

numberOfTopics = 0
TOPIC_BY_MODEL_THRESHOLD = 0
WITHIN_TOPIC_METHOD_COHERENCE = 0
BUGS_METHODS_NUMBER_THRESHOLD = 100
bugs_data_dic  = {}
methods_data_dic = {}
topicsMethodsMapping_dic = {}


print()

def mapHelper(s):
    data = s.strip().split(' ')
    if float(data[1]) < TOPIC_BY_MODEL_THRESHOLD:
        return
    return  (data[0], float(data[1]))

def mapBugsAndMethods():
    global bugs_data_dic , methods_data_dic

    bugs_data_dic.clear()
    methods_data_dic.clear()


    with open('dataFiles/bugs_lang.json', 'r') as bugs_reader:
        bugs_data_dic = json.load(bugs_reader)

    bugsIds_list = list(bugs_data_dic.keys())

    with open('dataFiles/methods_lang_clean.json', 'r') as methods_reader:
        methods_data_dic = json.load(methods_reader)

    methodsIds_list = list(methods_data_dic.keys())



    # make bugs_data_dic as:  bug  -> [model_bug_numbering, model_bug_topics , bug_changed_methods]
    # make methods_data_dic as:  method  -> [model_method_numbering, model_method_topics ]
    bugsNumOfMethods = []
    with open  ('topicModel_outputFiles/lang_methods_topics.csv', 'r', encoding='utf-8') as model_data_reader:
        data_model = csv.DictReader(model_data_reader, delimiter=',')
        for row in data_model:
            num = (int)(row['Document_No'])
            topics = row['Topics']

            # filter method's and bug's topics by coherence THRESHOLD
            topics = list(map(mapHelper, topics.split(',') ))
            topics = list(filter(None, topics))
            topics = sorted(topics, key=getKey, reverse=True)

            if num < len(methodsIds_list)  :
                methods_data_dic[methodsIds_list[num]] = [num, topics]

            else:
                method_list = bugs_data_dic[bugsIds_list[num - len(methodsIds_list)]][1]

                # filter bugs with very high number of methods
                if len(method_list)> BUGS_METHODS_NUMBER_THRESHOLD:
                    del bugs_data_dic[bugsIds_list[num - len(methodsIds_list)]]
                    # print(len(method_list))
                    continue


                bugs_data_dic[bugsIds_list[num - len(methodsIds_list)]][0] = num
                bugs_data_dic[bugsIds_list[num - len(methodsIds_list)]][1] = topics
                bugsNumOfMethods.append(len(method_list))

                method_list_numbered = []
                for m in method_list:
                    method_list_numbered.append(methods_data_dic[m][0])

                bugs_data_dic[bugsIds_list[num - len(methodsIds_list)]].append(method_list_numbered)

    # plotNumOfMethodsPerBugHist(bugsNumOfMethods)



def plotNumOfMethodsPerBugHist(bugsNumOfMethods):
    plt.hist(bugsNumOfMethods, bins=[1, 60, 100, 200, 600])
    plt.show()
    print("len", len(bugsNumOfMethods))
    print("sum", sum(bugsNumOfMethods))


#ehlper function for list sorting
def getKey(item):
    return item[1]





# evaluateMethod1_ByFullListOutput :
# no coherence threshold, list order by topics coherence, then by methods coherence
# evaluateMethod2_ByUsingThreshold :
# coherence threshold of th, list order by topics coherence, then by methods coherence
def evaluateModel():


    percentagesEvaluation = [10,20,30,40,50]
    evaluateMethod1_ByFullListOutput(percentagesEvaluation )



def mapTopicsToMethods():
    global topicsMethodsMapping_dic

    topicsMethodsMapping_dic.clear()

    # get number of topics in the model
    with open(modelOutput_directory + 'lang_topics_list.csv') as reader:
        numberOfTopics = len(reader.readlines())

    for i in range(0,numberOfTopics):
        topicsMethodsMapping_dic[str(i)] = []

    for method, m_data in methods_data_dic.items():
        m_num = m_data[0]
        topics = m_data[1]


        for t in topics:
            top_num = t[0]
            top_coher = t[1]

            # #filter methods in each topic according to its topic coherence
            # if top_coher > WITHIN_TOPIC_METHOD_COHERENCE:
            #     topicsMethodsMapping_dic[top_num].append((m_num, top_coher ))
            topicsMethodsMapping_dic[top_num].append((m_num, top_coher))


    num_of_methods_summary = []
    # sort method list of topics by coherence
    for top, m_list in topicsMethodsMapping_dic.items():
        m_list = sorted(m_list, key=getKey, reverse=True)
        coherence_list = list(map(lambda tp: tp[1] , m_list))
        print("topic:{} , max:{} , min:{} , avg:{}, median:{}".format(top, max(coherence_list),              min(coherence_list), statistics.mean(coherence_list), statistics.median(coherence_list)))

        count_before_filter = len( topicsMethodsMapping_dic[top])
        list_coher_filter = statistics.mean(coherence_list)
        filered_list = list(filter(lambda tp: tp[1] > list_coher_filter , m_list))

        #----- IF YOU WANT VALUES AND NUBERING FILTER ON RESULTS
        # topicsMethodsMapping_dic[top] = filered_list[:int(len(filered_list)*WITHIN_TOPIC_METHOD_COHERENCE)]
        #----- IF YOU WANT FILTER ON RESULTS
        topicsMethodsMapping_dic[top] = filered_list
        #----- IF YOU WANT NO FILTER ON RESULTS
        # topicsMethodsMapping_dic[top] = m_list


        count_after_filter = len( topicsMethodsMapping_dic[top])
        num_of_methods_summary.append((top,count_after_filter))
        # print("count_before_filter", count_before_filter,  "count_after_filter" ,count_after_filter )

    print(num_of_methods_summary)











def evaluateMethod1_ByFullListOutput( percentagesEvaluation):

    bugsHitCounter = {}
    numIterations = 0

    for per in percentagesEvaluation:
        bugsHitCounter[per] = 0

    TotalBugsMethodsNum = 0
    for bug, data in bugs_data_dic.items():

        topics = data[1]

        #get tagged data of bugs methods
        bugMethodsId_Tagged =  data[2]
        TotalBugsMethodsNum += len(data[2])

        #  get the bug methods from the model according to topic
        topicMethodList = []

        for topic in topics:
            topicMethodList.extend(map (lambda tp: tp[0], topicsMethodsMapping_dic[topic[0]]))

        bugsMethodsId_Model = []

        #clean duplicates and keep the order of the list
        for id in topicMethodList:
            if id not in bugsMethodsId_Model:
                bugsMethodsId_Model.append(id)

        methodsNumber = len(bugsMethodsId_Model)





        startList = 0
        for per in percentagesEvaluation:


            if len(bugMethodsId_Tagged) == 0:
                break

            numIterations +=1

            counter = int(methodsNumber*per/100)

            partList = bugsMethodsId_Model[startList:counter]

            hitCounter = len(set(partList).intersection(set(bugMethodsId_Tagged)))

            bugsHitCounter[per] += hitCounter
            #     print("bug:{}, Per:{} , num:{}/{}".format(data[0],per, hitCounter, len(bugMethodsId_Tagged)))

            startList = counter

            if hitCounter>0:
                inters = (set(partList).intersection(set(bugMethodsId_Tagged)))
                s1 = set(bugMethodsId_Tagged)
                s2 = set(partList)
                s1.difference_update(s2)
                bugMethodsId_Tagged = list(s1)

                # print()



    #add nu,ber of methods from lower list percentages to higher
    for i in range(0,len(percentagesEvaluation)-1):
        bugsHitCounter[percentagesEvaluation[i+1]] += bugsHitCounter[percentagesEvaluation[i]]


    print(bugsHitCounter)
    print("Total methods to find: " + str(TotalBugsMethodsNum))
    numOfTestedMethods =  []
    # for per in percentagesEvaluation:
    #     numOfTestedMethods.append(int(methodsNumber * per / 100))
    #
    # print("numOfTestedMethods: " + str(numOfTestedMethods))
    print("numIterations", numIterations)












trsh = [0, 0.01, 0.05, 0.07, 0.1  ]

within_trsh = np.arange(0.5,1.05, 0.05)


for t in trsh:
    TOPIC_BY_MODEL_THRESHOLD = t

    for wt in within_trsh:
        WITHIN_TOPIC_METHOD_COHERENCE = wt

        print("TOPIC_BY_MODEL_THRESHOLD: " + str(TOPIC_BY_MODEL_THRESHOLD))
        print("WITHIN_TOPIC_METHOD_COHERENCE: " + str(WITHIN_TOPIC_METHOD_COHERENCE))

        mapBugsAndMethods()
        mapTopicsToMethods()
        evaluateModel()


