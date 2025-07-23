import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import re

# 设置中文字体防止乱码
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 城市中心坐标（修复原代码语法错误）
city_coords = {
    '北京': (39.9, 116.4), '上海': (31.2, 121.5), '武汉': (30.6, 114.3),
    '深圳': (22.5, 114.0), '广州': (23.1, 113.3), '杭州': (30.3, 120.2),
    '南京': (32.1, 118.8), '天津': (39.1, 117.2), '成都': (30.6, 104.1),
    '青岛': (36.1, 120.3)
}


def extract_city(address):
    match = re.match(r'(.{2,3})市', address)
    if match:
        return match.group(1)
    return '其他'


# 投诉类型标注
def label_complaint_type(text):
    if any(word in text for word in ['态度', '语气', '表情', '不耐烦']):
        return '服务态度'
    elif any(word in text for word in ['排队', '等待', '时间长', '慢']):
        return '业务效率'
    elif any(word in text for word in ['坏了', '故障', '死机']):
        return '系统故障'
    elif any(word in text for word in ['手续费', '年费', '扣款', '收益']):
        return '费用争议'
    else:
        return '其他'


# 区域类型标注
def label_area_type(branch_name):
    if '中心' in branch_name or '市' in branch_name:
        return '商业中心'
    elif '大学' in branch_name or '校区' in branch_name:
        return '教育区'
    elif '机场' in branch_name or '港口' in branch_name:
        return '交通枢纽'
    elif '开发' in branch_name or '高新' in branch_name:
        return '工业区'
    else:
        return '居民区'


# 数据处理函数
def process_data(df):
    df['complaint_type'] = df['complaint_text'].apply(label_complaint_type)
    df['area_type'] = df['branch_name'].apply(label_area_type)
    df['city'] = df['address'].apply(extract_city)
    df['latitude'] = df['city'].apply(lambda x: city_coords.get(x, (35.0, 110.0))[0] + np.random.uniform(-0.05, 0.05))
    df['longitude'] = df['city'].apply(lambda x: city_coords.get(x, (35.0, 110.0))[1] + np.random.uniform(-0.05, 0.05))
    df['daily_complaints'] = np.random.poisson(3, len(df))
    df['complaint_rate'] = df['daily_complaints'] / df['daily_customers']
    df['new_staff_ratio'] = df['new_staff_count'] / df['staff_count']
    df['employee_level'] = pd.cut(df['employee_experience'], bins=[0, 2, 5, 10], labels=['新人', '中级', '资深'])
    return df


# 核心新增：为每个支行生成与员工数量匹配的员工编号池，并为投诉记录分配员工
def generate_employee_ids(df):
    # 1. 按支行分组，获取每个支行的员工总数（staff_count）
    branch_staff = df[['branch_id', 'staff_count']].drop_duplicates()  # 去重，保留每个支行的员工数

    # 2. 为每个支行生成员工编号（格式：branch_id + 序号，如101支行有17名员工→10101~10117）
    branch_employee_map = {}  # 存储：branch_id → [员工编号列表]
    for _, row in branch_staff.iterrows():
        bid = row['branch_id']
        staff_num = int(row['staff_count'])  # 该支行的员工总数
        # 生成员工编号（如101支行17名员工→10101到10117）
        employees = [f"{bid}{str(i).zfill(2)}" for i in range(1, staff_num + 1)]
        branch_employee_map[bid] = employees

    # 3. 为每条投诉记录随机分配所属支行的一个员工编号
    def assign_employee(branch_id):
        employees = branch_employee_map[branch_id]
        return np.random.choice(employees)  # 从该支行的员工中随机选一个

    df['employee_id'] = df['branch_id'].apply(assign_employee)
    return df


# 主程序
st.set_page_config(layout="wide")
st.title("📊 银行网点投诉分析平台")

uploaded_file = st.file_uploader("📥 上传CSV文件（包含投诉数据）", type=["csv"])

if uploaded_file:
    # 读取数据集
    df = pd.read_csv(uploaded_file)

    # 核心步骤：基于支行员工数量生成员工编号并分配给投诉记录
    df = generate_employee_ids(df)

    # 原有数据处理
    df = process_data(df)

    # 数据预览（展示员工编号与支行的关联）
    st.subheader("📌 数据预览（含员工编号）")
    st.dataframe(df[['branch_id', 'branch_name', 'staff_count', 'employee_id', 'complaint_text']].head(20))

    # 原有投诉类型分布（保持不变）
    st.subheader("📈 投诉类型分布")
    fig1, ax1 = plt.subplots()
    labels = df['complaint_type'].value_counts().index
    sizes = df['complaint_type'].value_counts().values
    colors = sns.color_palette('pastel')[0:len(labels)]
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1)

    # 原有区域类型与投诉类型（保持不变）
    st.subheader("📊 区域类型与投诉类型")
    cross_tab = pd.crosstab(df['area_type'], df['complaint_type'])
    st.bar_chart(cross_tab)

    # 原有员工经验与投诉类型（保持不变）
    st.subheader("📌 员工经验与投诉类型关系热力图")
    pivot = pd.crosstab(df['employee_level'], df['complaint_type'])
    fig3, ax3 = plt.subplots()
    sns.heatmap(pivot, annot=True, fmt='d', cmap='YlGnBu', ax=ax3)
    st.pyplot(fig3)

    # 员工编号与投诉关联分析（基于合理分配的员工编号）
    st.subheader("👤 员工编号与投诉关联分析")

    # 1. 员工投诉数量排名（Top10）
    st.subheader("1. 员工投诉数量排名（Top10）")
    emp_complaint_count = df['employee_id'].value_counts().head(10).reset_index()
    emp_complaint_count.columns = ['员工编号', '投诉次数']
    fig_emp_count, ax_emp_count = plt.subplots(figsize=(10, 6))
    sns.barplot(x='员工编号', y='投诉次数', data=emp_complaint_count, ax=ax_emp_count, palette='light:#5A9')
    ax_emp_count.set_title('员工投诉数量排名（Top10）', fontsize=12)
    ax_emp_count.set_xlabel('员工编号', fontsize=10)
    ax_emp_count.set_ylabel('投诉次数', fontsize=10)
    plt.xticks(rotation=45)
    st.pyplot(fig_emp_count)

    # 2. 员工与投诉类型分布（热力图）
    st.subheader("2. 员工投诉类型分布（热力图）")
    top_10_emp = df['employee_id'].value_counts().head(10).index  # 取投诉最多的10名员工
    df_top_emp = df[df['employee_id'].isin(top_10_emp)]
    emp_type_cross = pd.crosstab(df_top_emp['employee_id'], df_top_emp['complaint_type'])
    fig_emp_type, ax_emp_type = plt.subplots(figsize=(10, 6))
    sns.heatmap(emp_type_cross, annot=True, fmt='d', cmap='Blues', ax=ax_emp_type)
    ax_emp_type.set_title('Top10高投诉员工的投诉类型分布', fontsize=12)
    st.pyplot(fig_emp_type)

    # 3. 支行-员工投诉关联（补充：展示支行下员工的投诉分布）
    st.subheader("3. 支行-员工投诉分布（示例支行）")
    # 选一个投诉较多的支行展示
    top_branch = df['branch_id'].value_counts().index[0]
    df_top_branch = df[df['branch_id'] == top_branch]
    branch_emp_count = df_top_branch['employee_id'].value_counts().reset_index()
    branch_emp_count.columns = ['员工编号', '投诉次数']
    fig_branch_emp, ax_branch_emp = plt.subplots(figsize=(10, 6))
    sns.barplot(x='员工编号', y='投诉次数', data=branch_emp_count, ax=ax_branch_emp, palette='light:#87CEFA')
    ax_branch_emp.set_title(f'支行 {top_branch} 下各员工的投诉次数', fontsize=12)
    ax_branch_emp.set_xlabel('员工编号', fontsize=10)
    ax_branch_emp.set_ylabel('投诉次数', fontsize=10)
    plt.xticks(rotation=45)
    st.pyplot(fig_branch_emp)

    # 原有预测模型（保持不变）
    st.subheader("🧠 投诉类型预测（模型演示）")
    X = df[['employee_experience', 'daily_customers', 'staff_count', 'new_staff_ratio']]
    X = pd.concat([X, pd.get_dummies(df['area_type'], drop_first=True)], axis=1)
    y = df['complaint_type'].astype('category').cat.codes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf = RandomForestClassifier().fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    st.success(f"模型预测准确率：{score:.2%}")

else:
    st.info(
        "请上传包含投诉信息的CSV文件（如simulated_complaints_500.csv），需包含 branch_id、staff_count（员工数量）、complaint_text 等字段。")