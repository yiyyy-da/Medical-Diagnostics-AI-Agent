# 导入所需的模块
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from Utils.Agents import Cardiologist, Psychologist, Pulmonologist, MultidisciplinaryTeam
from dotenv import load_dotenv
import json, os
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 从 .env 文件加载 API 密钥
load_dotenv(dotenv_path='apikey.env')

# 读取医疗报告文件
# 这里读取的是一个关于恐慌症发作（Panic Attack Disorder）的病历
with open(r"Medical Reports\Medical Rerort - Michael Johnson - Panic Attack Disorder.txt", "r") as file:
    medical_report = file.read()

# 初始化各个专科医生 Agent
# 传入病历报告，每个 Agent 会根据自己的专业背景进行分析
agents = {
    "Cardiologist": Cardiologist(medical_report),  # 心脏病专家
    "Psychologist": Psychologist(medical_report),  # 心理学家
    "Pulmonologist": Pulmonologist(medical_report) # 呼吸内科专家
}

# 执行 Agent 并获取其分析结果的函数
def get_response(agent_name, agent):
    response = agent.run()
    return agent_name, response

# 使用线程池并发运行各个 Agent，提高处理效率
responses = {}
with ThreadPoolExecutor() as executor:
    # 提交任务到线程池
    futures = {executor.submit(get_response, name, agent): name for name, agent in agents.items()}
    
    # 收集已完成的任务结果
    for future in as_completed(futures):
        agent_name, response = future.result()
        responses[agent_name] = response

# 初始化多学科专家团队 (MDT) Agent
# 将心脏病专家、心理学家和呼吸内科专家的分析报告汇总给团队
team_agent = MultidisciplinaryTeam(
    cardiologist_report=responses["Cardiologist"],
    psychologist_report=responses["Psychologist"],
    pulmonologist_report=responses["Pulmonologist"]
)

# 运行 MDT Agent 生成最终诊断建议
final_diagnosis = team_agent.run()
final_diagnosis_text = "### Final Diagnosis:\n\n" + final_diagnosis
txt_output_path = "results/final_diagnosis.txt"

# 确保结果保存目录存在
os.makedirs(os.path.dirname(txt_output_path), exist_ok=True)

# 将最终诊断结果写入文本文件
with open(txt_output_path, "w") as txt_file:
    txt_file.write(final_diagnosis_text)

print(f"最终诊断结果已保存至: {txt_output_path}")


