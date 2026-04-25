import streamlit as st
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from Utils.Agents import Cardiologist, Psychologist, Pulmonologist, MultidisciplinaryTeam

# 页面配置
st.set_page_config(page_title="AI 医疗诊断辅助系统", page_icon="🏥", layout="wide")

# 加载环境变量
load_dotenv(dotenv_path='apikey.env')

# 侧边栏：配置和选择
st.sidebar.title("⚙️ 设置")
report_dir = "Medical Reports"
available_reports = [f for f in os.listdir(report_dir) if f.endswith('.txt')]

selected_report_file = st.sidebar.selectbox("选择医疗报告", available_reports)

st.title("🏥 AI 医疗多学科会诊演示系统")
st.markdown("""
本系统通过多个 AI 智能体（心脏病专家、心理学家、呼吸内科专家）协作，对复杂的医疗病例进行并行分析，并由多学科团队给出最终诊断建议。
""")

# 主界面布局
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📄 原始医疗报告")
    if selected_report_file:
        file_path = os.path.join(report_dir, selected_report_file)
        with open(file_path, "r", encoding="utf-8") as f:
            medical_report = f.read()
        st.text_area("报告内容", medical_report, height=400)
    else:
        st.warning("请在侧边栏选择一个医疗报告。")

with col2:
    st.subheader("🧠 AI 诊断分析")
    
    if st.button("🚀 开始多学科会诊分析", type="primary"):
        if not medical_report:
            st.error("未找到报告内容！")
        else:
            with st.status("正在进行专家会诊...", expanded=True) as status:
                # 初始化 Agent
                st.write("正在分发任务给各科专家...")
                agents = {
                    "心脏病专家": Cardiologist(medical_report),
                    "心理学家": Psychologist(medical_report),
                    "呼吸内科专家": Pulmonologist(medical_report)
                }

                # 执行 Agent 的函数
                def get_response(agent_name, agent):
                    return agent_name, agent.run()

                # 并发执行
                responses = {}
                st.write("专家正在并行分析中...")
                with ThreadPoolExecutor() as executor:
                    futures = {executor.submit(get_response, name, agent): name for name, agent in agents.items()}
                    
                    for future in as_completed(futures):
                        agent_name, response = future.result()
                        responses[agent_name] = response
                        st.write(f"✅ {agent_name} 分析完成")

                # MDT 汇总
                st.write("正在汇总多学科专家意见...")
                team_agent = MultidisciplinaryTeam(
                    cardiologist_report=responses["心脏病专家"],
                    psychologist_report=responses["心理学家"],
                    pulmonologist_report=responses["呼吸内科专家"]
                )
                final_diagnosis = team_agent.run()
                status.update(label="分析完成！", state="complete", expanded=False)

            # 显示结果
            st.divider()
            st.success("### 🏁 最终诊断建议")
            st.markdown(final_diagnosis)

            # 各专家详细意见折叠栏
            with st.expander("查看各科专家详细意见"):
                for name, resp in responses.items():
                    st.write(f"**{name} 的意见：**")
                    st.info(resp)

# 页脚
st.divider()
st.caption("注：本系统仅供研究和教育参考，不作为临床医疗诊断依据。")
