from disaster_core import create_detector_from_env
from server import verify_disaster_text, verify_disaster_batch
import json
from ATool import processTo, processMultiple

detector = create_detector_from_env()
print("\n输入0可结束该程序\n")
while (True):
   
    opt = input("请选择要处理单条或者多条文本(1代表单,2代表多): ")
    if (opt=='1'):

        text = input("请输入文本:")
        temp = verify_disaster_text(text)
        result = processTo(temp)    
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif (opt=='2'):
    
        mark = input("请输入分割符号:")
        text = input("请输入文本:")
        tlist = text.split(mark)
        temp = verify_disaster_batch(tlist)
        results = processMultiple(temp)
        print(json.dumps(results, ensure_ascii=False, indent=2))
    
    elif (opt=='0'):
        break
    else:
        print("输入错误")

