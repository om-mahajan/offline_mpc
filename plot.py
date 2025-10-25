import pandas  as pd
import numpy as np
import matplotlib.pyplot as plt

db1 = pd.read_csv('D:/IITM/RL_safe_diff/logs/safedice_energy/OfflinePointGoal1Gymnasium-v0/safedice_flow_matching/seed-000-2025-10-25-15-01-58/phase2_flow/progress.csv')
db2 = pd.read_csv('D:/IITM/RL_safe_diff/logs/safedice_energy/OfflinePointGoal1Gymnasium-v0/safedice_flow_matching/seed-000-2025-10-25-14-21-59/phase2_flow/progress.csv')
db2_re = pd.read_csv('D:/IITM/RL_safe_diff/share_diffusion/safedice/Point_Goal1/ep_reward/ep_reward_50_0.csv')
db2_cost = pd.read_csv('D:/IITM/RL_safe_diff/share_diffusion/safedice/Point_Goal1/ep_cost/ep_cost_50_0.csv')

plt.figure(figsize=(12,5))

db1.dropna(inplace=True)  
print(db1.head())
X1 = db1['Train/Step']
y1_re = db1['Eval/Reward']
y1_cost = db1['Eval/Cost']

db2.dropna(inplace=True)  
print(db2.head())
X2 = db2['Train/Step']
y2_re = db2['Eval/Reward']
y2_cost = db2['Eval/Cost']

""" 

X2 = db2_re['Step']  
y2_re = db2_re['Value']
y2_cost = db2_cost['Value'] """

# Calculate moving average for flow reward (window size = 5)
window_size = 1
y1_re_smooth = y1_re.rolling(window=window_size, min_periods=1).mean()
plt.subplot(1, 2, 1)
plt.plot(X1, y1_re_smooth, label='d1', color='blue', linewidth=1)  # Moving average (bold)
plt.plot(X2, y2_re, label='d2', color='cyan', linestyle='--')
plt.xlabel('Training Step')
plt.ylabel('Reward')
plt.title('Train')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot for cost
window_size = 10
y1_cost_smooth = y1_cost.rolling(window=window_size, min_periods=1).mean()
plt.subplot(1, 2, 2)
plt.plot(X1, y1_cost_smooth, label='d1', color='blue', linewidth=1)
plt.plot(X2, y2_cost, label='d2', color='cyan', linestyle='--')
plt.xlabel('Training Step')
plt.ylabel('Cost')
plt.title('Train')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()