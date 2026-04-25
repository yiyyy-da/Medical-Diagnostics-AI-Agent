import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# 基础 Agent 类，定义了所有医学专家 Agent 的共同行为
class Agent:
    def __init__(self, medical_report=None, role=None, extra_info=None):
        self.medical_report = medical_report  # 医疗报告内容
        self.role = role                      # Agent 的角色（如：心脏病专家）
        self.extra_info = extra_info          # 额外信息（主要用于多学科团队汇总）
        # 根据角色和信息初始化提示词模板
        self.prompt_template = self.create_prompt_template()
        # 初始化 LLM 模型
        # 按照用户要求改为 qwen-max (通义千问)
        # 注意：目前通义千问官方最强模型为 qwen-max，若您使用的是特定版本的 qwen3-max 请自行调整
        self.model = ChatOpenAI(
            temperature=0, 
            model="qwen3-max", 
            openai_api_base=os.getenv("OPENAI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        )

    # 根据不同的角色创建相应的 Prompt 模板
    def create_prompt_template(self):
        if self.role == "MultidisciplinaryTeam":
            # 多学科专家团队的提示词：汇总各专家的分析并给出最终诊断
            templates = f"""
                扮演一个由医疗专业人员组成的多学科团队。
                你将收到由心脏病专家、心理学家和呼吸内科专家审阅后的病历报告。
                任务：审查这些专家的分析报告，进行综合分析，并列出该患者可能存在的 3 个健康问题。
                仅返回 3 个可能健康问题的列表（点号分隔），并为每个问题提供原因。
                
                心脏病专家报告: {self.extra_info.get('cardiologist_report', '')}
                心理学家报告: {self.extra_info.get('psychologist_report', '')}
                呼吸内科专家报告: {self.extra_info.get('pulmonologist_report', '')}
            """
        else:
            # 各个专科医生的提示词模板
            templates = {
                "Cardiologist": """
                    扮演一名心脏病专家。你将收到一份患者的医疗报告。
                    任务：审查患者的心脏检查结果，包括心电图、血液检查、动态心电图监测和超声心动图。
                    重点：确定是否存在任何可能解释患者症状的细微心脏问题征兆。排除可能在常规测试中被忽略的潜在心脏疾病，如心律失常或结构异常。
                    建议：提供关于是否需要进一步进行心脏测试或监测的指导，以确保没有隐藏的心脏相关担忧。如果发现心脏问题，请提出潜在的管理策略。
                    请仅返回患者症状的可能原因以及建议的后续步骤。
                    医疗报告: {medical_report}
                """,
                "Psychologist": """
                    扮演一名心理学家。你将收到一份患者报告。
                    任务：审查患者报告并提供心理评估。
                    重点：识别可能影响患者健康的潜在心理健康问题，如焦虑、抑郁或创伤。
                    建议：提供关于如何应对这些心理健康问题的指导，包括治疗、咨询或其他干预措施。
                    请仅返回可能的心理健康问题以及建议的后续步骤。
                    患者报告: {medical_report}
                """,
                "Pulmonologist": """
                    扮演一名呼吸内科专家。你将收到一份患者报告。
                    任务：审查患者报告并提供肺部评估。
                    重点：识别可能影响患者呼吸的潜在呼吸系统问题，如哮喘、COPD（慢阻肺）或肺部感染。
                    建议：提供关于如何应对这些呼吸问题的指导，包括肺功能测试、影像学研究或其他干预措施。
                    请仅返回可能的呼吸系统问题以及建议的后续步骤。
                    患者报告: {medical_report}
                """
            }
            templates = templates[self.role]
        return PromptTemplate.from_template(templates)
    
    # 运行 Agent，将 Prompt 发送给模型并获取回复
    def run(self):
        print(f"{self.role} 正在运行分析...")
        prompt = self.prompt_template.format(medical_report=self.medical_report)
        try:
            response = self.model.invoke(prompt)
            return response.content
        except Exception as e:
            print("执行过程中出现错误:", e)
            return None

# 定义具体的专科 Agent 类，继承自基础 Agent 类
# 心脏病专家 Agent
class Cardiologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Cardiologist")

# 心理学家 Agent
class Psychologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Psychologist")

# 呼吸内科专家 Agent
class Pulmonologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Pulmonologist")

# 多学科专家团队 (MDT) Agent
# 负责汇总三个专家的分析结果并给出最终结论
class MultidisciplinaryTeam(Agent):
    def __init__(self, cardiologist_report, psychologist_report, pulmonologist_report):
        extra_info = {
            "cardiologist_report": cardiologist_report,
            "psychologist_report": psychologist_report,
            "pulmonologist_report": pulmonologist_report
        }
        super().__init__(role="MultidisciplinaryTeam", extra_info=extra_info)
