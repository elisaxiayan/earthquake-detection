使用方式：

  # 单数据集
  python3 analyze_earthquake_results.py <question.json> <result.json> -n 真实地震

  # 合并两个数据集
  python3 analyze_earthquake_results.py \
      earthquakes_true.json earthquakes_true_result.json -n 真实地震 \
      --second false_domestic_earthquake_batch_50.json false_domestic_result.json --second-name 伪造地震 \
      -o combined_analysis.csv

  # 指定输出路径
  python3 analyze_earthquake_results.py q.json r.json -n 真实地震 -o my_output.csv

  脚本逻辑：

  - 从 input_text 正则提取地点、震级、北京时间、经纬度
  - 从 result 的 results[] 提取 occurrence/magnitude 判定、USGS 事件数、匹配震级、时间窗口
  - 从 extraction.claims[] 判断提取完整性、结果地点、是否被翻译为英文
  - 自动计算震级差值，按规则生成综合判定、错误分类、根因分析、改进建议
  - 运行结束后打印汇总统计

  每次新跑出 result 后，直接传对应的 question JSON + result JSON 就能生成标准格式的 CSV。