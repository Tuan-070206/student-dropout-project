import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix, 
    ConfusionMatrixDisplay
)

# 1. Khởi tạo thư mục xuất kết quả
os.makedirs('outputs', exist_ok=True)

# 2. Nạp dữ liệu & Làm sạch tên cột
data_path = 'data/raw/dataset.csv'
df = pd.read_csv(data_path, sep=None, engine='python')

# Xóa dấu ngoặc kép và khoảng trắng thừa ở tên cột
df.columns = df.columns.str.replace('"', '').str.strip()

print("=== 1. KHÁM PHÁ DỮ LIỆU BAN ĐẦU ===")
print(f"Số mẫu và thuộc tính: {df.shape}")
print(f"Số giá trị thiếu (NA): {df.isnull().sum().sum()}")
print("\nPhân bố nhãn sinh viên (Target):")
print(df['Target'].value_counts())

# 3. Phân tách biến độc lập (X) và biến phụ thuộc (y)
X = df.drop(columns=['Target'])
y = df['Target']

# Mã hóa Target thành số (0, 1, 2)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Lọc danh sách cột phân loại phù hợp với dữ liệu thực tế
raw_categorical_cols = [
    'Marital status', 'Application mode', 'Course', 'Daytime/evening attendance',
    'Previous qualification', 'Nacionality', "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation", 'Displaced', 'Educational special needs',
    'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International'
]
categorical_cols = [col for col in raw_categorical_cols if col in X.columns]
numerical_cols = [col for col in X.columns if col not in categorical_cols]

# Xuất Heatmap tương quan học tập
academic_cols = [col for col in numerical_cols if 'Curricular units' in col]
if academic_cols:
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[academic_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Ma trận tương quan giữa các môn học')
    plt.tight_layout()
    plt.savefig('outputs/correlation_heatmap.png')
    plt.close()

# 4. Phân tách tập Train / Test giữ nguyên tỉ lệ lớp
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 5. Xây dựng Pipeline chống rò rỉ dữ liệu
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

models = {
    'Logistic Regression (Baseline)': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    'Decision Tree': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(max_depth=7, random_state=42))
    ]),
    'Random Forest': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
}

# 6. Huấn luyện và Đánh giá Cross-Validation
print("\n=== 2. KẾT QUẢ SO SÁNH MÔ HÌNH ===")
results = {}

for name, pipeline in models.items():
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    
    results[name] = {
        'Pipeline': pipeline,
        'CV Mean Accuracy': cv_scores.mean(),
        'Test Accuracy': test_acc,
        'Predictions': y_pred
    }
    
    print(f"[{name}]")
    print(f"  + CV Accuracy (5-fold): {cv_scores.mean()*100:.2f}%")
    print(f"  + Test Accuracy:        {test_acc*100:.2f}%\n")

# 7. Phân tích mô hình tốt nhất
best_model_name = max(results, key=lambda k: results[k]['Test Accuracy'])
best_info = results[best_model_name]
best_pipeline = best_info['Pipeline']
y_pred_best = best_info['Predictions']

print(f"=== 3. BÁO CÁO CHI TIẾT MÔ HÌNH TỐT NHẤT ({best_model_name}) ===")
print(classification_report(y_test, y_pred_best, target_names=label_encoder.classes_))

# Trực quan Confusion Matrix
cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.title(f'Confusion Matrix - {best_model_name}')
plt.tight_layout()
plt.savefig('outputs/confusion_matrix.png')
plt.close()

# Trực quan Feature Importance / Feature Coefficients
classifier = best_pipeline.named_steps['classifier']
encoded_cat_cols = best_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_cols)
all_feature_names = numerical_cols + list(encoded_cat_cols)

# Trích xuất tầm quan trọng đặc trưng dựa vào loại mô hình
if hasattr(classifier, 'feature_importances_'):
    importances = classifier.feature_importances_
elif hasattr(classifier, 'coef_'):
    # Với Logistic Regression đa lớp, lấy trung bình độ lớn tuyệt đối của hệ số trọng số
    importances = np.abs(classifier.coef_).mean(axis=0)
else:
    importances = None

if importances is not None:
    feat_imp = pd.Series(importances, index=all_feature_names).sort_values(ascending=False).head(15)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='mako')
    plt.title(f'Top 15 Đặc trưng tác động lớn nhất ({best_model_name})')
    plt.xlabel('Importance Score / Coefficient Value')
    plt.tight_layout()
    plt.savefig('outputs/feature_importance.png')
    plt.close()

# 8. Mở rộng Chương 6: Phân cụm sinh viên K-Means
print("\n=== 4. PHÂN CỤM SINH VIÊN (K-MEANS) ===")
cluster_features = ['Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)']
cluster_features = [col for col in cluster_features if col in df.columns]

if len(cluster_features) == 2:
    X_cluster = df[cluster_features].dropna()
    kmeans = KMeans(n_clusters=3, random_state=42).fit(X_cluster)
    df_clustered = X_cluster.copy()
    df_clustered['Cluster'] = kmeans.labels_

    plt.figure(figsize=(8, 6))
    plt.scatter(df_clustered[cluster_features[0]], df_clustered[cluster_features[1]], c=df_clustered['Cluster'], cmap='viridis', alpha=0.6)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='red', marker='X', s=200, label='Tâm cụm')
    plt.title('Phân cụm sinh viên theo điểm HK1 và HK2')
    plt.xlabel('Điểm trung bình HK1')
    plt.ylabel('Điểm trung bình HK2')
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/kmeans_student_clusters.png')
    plt.close()

print("\nHoàn tất! Các biểu đồ đã được lưu tự động vào thư mục 'outputs/'.")