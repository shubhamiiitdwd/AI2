from enum import Enum


class MLTask(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"


class ModelType(str, Enum):
    DRF = "DRF"
    GLM = "GLM"
    XGBOOST = "XGBoost"
    GBM = "GBM"
    DEEP_LEARNING = "DeepLearning"
    STACKED_ENSEMBLE = "StackedEnsemble"


class ClusteringAlgorithm(str, Enum):
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    GMM = "gmm"


class TrainingStatus(str, Enum):
    QUEUED = "queued"
    DATA_CHECK = "data_check"
    FEATURES = "features"
    TRAINING = "training"
    EVALUATION = "evaluation"
    COMPLETE = "complete"
    FAILED = "failed"
    STOPPED = "stopped"
    CLUSTERING = "clustering"
    CLUSTERING_EVAL = "clustering_eval"


class StorageMode(str, Enum):
    LOCAL = "local"
    AZURE = "azure"


class AIMode(str, Enum):
    HUGGINGFACE = "huggingface"
    AZURE = "azure"


MODEL_INFO = {
    ModelType.DRF: {"name": "Distributed Random Forest", "speed": "Medium", "description": "Ensemble of random trees"},
    ModelType.GLM: {"name": "Generalized Linear Model", "speed": "Fast", "description": "Logistic regression variant"},
    ModelType.XGBOOST: {"name": "XGBoost", "speed": "Medium", "description": "Gradient boosting framework"},
    ModelType.GBM: {"name": "Gradient Boosting Machine", "speed": "Medium", "description": "H2O's GBM implementation"},
    ModelType.DEEP_LEARNING: {"name": "Deep Learning", "speed": "Slow", "description": "Neural network models"},
    ModelType.STACKED_ENSEMBLE: {"name": "Stacked Ensemble", "speed": "Slow", "description": "Meta-learner combining models"},
}
