import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 读取CSV文件
data = pd.read_csv('trajectories-0750am-0805am.csv')

# 将 'v_Vel' 转换为 km/h
data['v_Vel_kmh'] = data['v_Vel'] * 0.3048 * 3.6

# 将 'v_Acc' 转换为 m/s^2
data['v_Acc_m_s2'] = data['v_Acc'] * 0.3048

# 选择用于速度聚类的特征
speed_features = data[['v_Vel_kmh']]
# 选择用于加速度聚类的特征
acceleration_features = data[['v_Acc_m_s2']]

# 标准化数据
scaler = StandardScaler()
scaled_speed_features = scaler.fit_transform(speed_features)
scaled_acceleration_features = scaler.fit_transform(acceleration_features)

# 使用KMeans聚类，假设分为3类
kmeans_speed = KMeans(n_clusters=3, random_state=42)
data['Speed_Cluster'] = kmeans_speed.fit_predict(scaled_speed_features)

kmeans_acceleration = KMeans(n_clusters=3, random_state=42)
data['Acceleration_Cluster'] = kmeans_acceleration.fit_predict(scaled_acceleration_features)

# 获取簇的数量
speed_cluster_counts = data['Speed_Cluster'].value_counts()
acceleration_cluster_counts = data['Acceleration_Cluster'].value_counts()

# 计算簇的占比
speed_cluster_proportions = speed_cluster_counts / len(data)
acceleration_cluster_proportions = acceleration_cluster_counts / len(data)

# 绘制速度和加速度的散点图
plt.figure(figsize=(12, 6))

# 绘制速度聚类的散点图
plt.subplot(1, 2, 1)
sns.scatterplot(x='v_Vel_kmh', y='v_Acc_m_s2', hue='Speed_Cluster', data=data,
                palette=['lightgreen', 'yellow', 'lightblue'], legend='full', alpha=1)
plt.title('Scatter Plot of Speed with Clusters')
plt.xlabel('Speed (km/h)')
plt.ylabel('Acceleration (m/s^2)')

# 添加速度簇占比文本标签
for cluster_label, proportion in zip(speed_cluster_proportions.index, speed_cluster_proportions):
    cluster_data = data[data['Speed_Cluster'] == cluster_label]
    x_center = cluster_data['v_Vel_kmh'].mean()
    y_center = cluster_data['v_Acc_m_s2'].mean()
    plt.text(x_center, y_center, f'{proportion:.2%}', ha='center', va='center', fontweight='bold', color='purple')

# 绘制加速度聚类的散点图
plt.subplot(1, 2, 2)
sns.scatterplot(x='v_Vel_kmh', y='v_Acc_m_s2', hue='Acceleration_Cluster', data=data,
                palette=['lightgreen', 'yellow', 'lightblue'], legend='full', alpha=1)
plt.title('Scatter Plot of Acceleration with Clusters')
plt.xlabel('Speed (km/h)')
plt.ylabel('Acceleration (m/s^2)')

# 添加加速度簇占比文本标签
for cluster_label, proportion in zip(acceleration_cluster_proportions.index, acceleration_cluster_proportions):
    cluster_data = data[data['Acceleration_Cluster'] == cluster_label]
    x_center = cluster_data['v_Vel_kmh'].mean()
    y_center = cluster_data['v_Acc_m_s2'].mean()
    plt.text(x_center, y_center, f'{proportion:.2%}', ha='center', va='center', fontweight='bold', color='purple')

plt.tight_layout()
plt.show()
