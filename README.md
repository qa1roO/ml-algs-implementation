# ml-algs-implementation
Этот репозиторий содержит мою реализацию ключевых алгоритмов машинного обучения в рамках обучения в Школе 21  на направлении Data Sceince.

> Все классы имеют префикс `My` во избежание конфликтов имён с scikit-learn.

## Реализованные алгоритмы

### Supervised learning
| Алгоритм | Тип задачи |
|----------|------------|
| Линейная регрессия (градиентный спуск и аналитическое решение) | Регрессия |
| Ridge (аналитическое решение) | Регрессия |
| Lasso (градиентный спуск) | Регрессия |
| Логистическая регрессия (SGD) | Классификация |
| Наивный байесовский классификатор | Классификация |
| Метод k ближайших соседей | Классификация |
| Дерево решений (классификатор и регрессор) | Оба |
| Случайный лес | Классификация |
| Градиентный бустинг | Классификация |
| Extremely Randomized Trees | Классификация |

### Unsupervised
| Алгоритм |
|----------|
| K-Means |
| DBSCAN |  
| GMM |  

### Neural networks
| Модель | Фреймворк |
|--------|-----------|
| MLP | NumPy |
| MLP | PyTorch |
| LeNet-5 | PyTorch |

### Validation
`KFold`  `StratifiedKFold`  `GroupKFold`  `TimeSeriesSplit`

---

## Результаты
Все эксперименты воспроизводятся в [`demo.ipynb`](demo.ipynb).

**Регрессия**: датасет - [Diabetes](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html) (442 объекта, 10 признаков)

| Модель | RMSE | R² |
|--------|------|----|
| MyLinearRegression (GD) | 61.32 | 0.321 |
| sklearn LinearRegression | 56.27 | 0.428 |
| MyRidge (аналитическое) | 60.36 | 0.342 |
| sklearn Ridge | 60.36 | 0.342 |
| MyLasso  | 68.39 | 0.155 |
| sklearn Lasso | 57.08 | 0.411 |
| MyDecisionTreeRegressor | 66.08 | 0.211 |
| sklearn DecisionTreeRegressor | 66.08 | 0.211 |


**Классификация**: датасет - [Breast Cancer](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset)

| Модель | Accuracy | F1 |
|--------|----------|----|
| MyLogisticRegression | 0.9737 | 0.9787 |
| MyRandomForestClassifier | 0.9649 | 0.9722 |
| MyGBDTClassifier | 0.9561 | 0.9655 |

**Кластеризация** — датасет: [make_blobs](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html) (3 кластера)

| Модель | Silhouette |
|--------|:----------:|
| MyKMeans (k=3) | 0.7502 |
| sklearn KMeans (k=3) | 0.7502 |
| MyDBSCAN (eps=0.3) | 0.7539 |
| sklearn DBSCAN (eps=0.3) | 0.7539 |
| MyGMM (k=3) | 0.7502 |
| sklearn GaussianMixture (k=3) | 0.7502 |


**Нейронные сети**: - датасет [Breast Cancer](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset)

| Модель | Accuracy | F1 |
|--------|----------|----|
| My_MLP (NumPy) | 0.9825 | 0.9861 |
| sklearn MLPClassifier | 0.9912 | 0.9931 |



---

## Запуск

```bash
git clone https://github.com/qa1roO/ml-algs-implementation
cd ml-algs-implementation
pip install -r requirements.txt
jupyter notebook demo.ipynb
```

---

## Стек

`Python` · `NumPy` · `Pandas` · `Matplotlib` · `PyTorch` · `torchvision`
