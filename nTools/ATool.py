import json

with open("tM.json", "r", encoding = "utf-8") as infile:
    data = json.load(infile)

def rmEx(dic):
    del dic["extraction"]

def getT(dic):
    return dic["input_text"]

def getBasicInfo(dic):
    temp = dic["results"][0]["claim"]
    fin = dict()
    fin["灾害类型"] = temp["hazard_type"]
    fin["地点"] = temp["location"]
    fin["经纬度"] = str(temp["lon"]) + ", " + str(temp["lat"])
    fin["时间段"] = temp["time_interval"]
    return fin

def getN(dic):
    result = dic["results"]
    length = len(result)
    rList = list()
    for i in range(length):
        dictionary = result[i]
        type_dict = dictionary["claim"]
        f_type = type_dict["fact_type"]
        stat = dictionary["status"]
        match stat:
            case "supported":
                stat = "正确"
            case "contradicted":
                stat = "错误"
            case _:
                stat = "未查询到结果"
        match f_type:
            case "occurrence":
                f_type = "发生"
                rList.append({"检测项" : f_type, "检测结果" : stat})
            case "count":
                f_type = "发生次数"
                tDict = {"检测项" : f_type, "检测结果" : stat}
                if "actual_value" in dictionary:
                    count = dictionary["actual_value"]
                    tDict.update({"实际发生次数" : count})
                rList.append(tDict)
            case "magnitude":
                f_type = "震级"
                rList.append({"检测项" : f_type, "检测结果" : stat})
    return rList

def getNumberOfText(dict):
    return dict["total"]

def getSingleText(dict):
    return dict["items"]

def getIndex(dict):
    return dict["index"]


def processTo(data):
    rmEx(data)
    finalD = {"模型输入文本" : getT(data)}
    finalD.update(getBasicInfo(data))
    sampList = getN(data)
    finalD.update({"幻觉判定结果" : sampList})
    return finalD

def processMultiple(raw_data):
    numb = getNumberOfText(raw_data)
    mid_data_list = getSingleText(raw_data)
    fin_dic = {"本次处理文本条数" : numb}
    for i in range(numb):
        single_text = mid_data_list[i]
        index = getIndex(single_text)
        del single_text["index"]
        temp = processTo(single_text)
        fin_dic.update({"当前处理第"+str(index)+"条文本" : temp})     
    return fin_dic





