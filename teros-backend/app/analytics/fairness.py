"""
공정성 강화 모듈 (Fairness Enhancement Module)

적대적 학습 및 인과 추론 기반 편향 완화
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from scipy import stats


class PredictorModel(nn.Module):
    """예측 모델 (Predictor) - 학생 재능 예측"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 10):
        """
        Args:
            input_dim: 입력 차원
            hidden_dim: 은닉층 차원
            output_dim: 출력 차원 (재능 카테고리 수)
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """순전파"""
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class AdversaryModel(nn.Module):
    """적대 모델 (Adversary) - 민감 정보 역추론"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 2):
        """
        Args:
            input_dim: 입력 차원 (예측 모델의 representation)
            hidden_dim: 은닉층 차원
            output_dim: 출력 차원 (민감 정보 카테고리 수, 예: 성별 2개)
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """순전파"""
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class AdversarialDebiasingNetwork:
    """적대적 편향 제거 네트워크"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        talent_categories: int = 10,
        sensitive_categories: int = 2,
        learning_rate: float = 0.001,
        lambda_adv: float = 1.0
    ):
        """
        Args:
            input_dim: 입력 차원
            hidden_dim: 은닉층 차원
            talent_categories: 재능 카테고리 수
            sensitive_categories: 민감 정보 카테고리 수
            learning_rate: 학습률
            lambda_adv: 적대적 손실 가중치
        """
        self.predictor = PredictorModel(input_dim, hidden_dim, talent_categories)
        self.adversary = AdversaryModel(hidden_dim, 64, sensitive_categories)
        
        self.predictor_optimizer = torch.optim.Adam(
            self.predictor.parameters(),
            lr=learning_rate
        )
        self.adversary_optimizer = torch.optim.Adam(
            self.adversary.parameters(),
            lr=learning_rate
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.lambda_adv = lambda_adv
    
    def train_step(
        self,
        features: torch.Tensor,
        talent_labels: torch.Tensor,
        sensitive_labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        학습 스텝
        
        Args:
            features: 입력 특징
            talent_labels: 재능 레이블
            sensitive_labels: 민감 정보 레이블
            
        Returns:
            손실 값 딕셔너리
        """
        # 예측 모델 학습
        self.predictor_optimizer.zero_grad()
        
        # 예측 모델의 representation 추출
        representation = self.predictor.fc2(
            self.predictor.relu(
                self.predictor.dropout(
                    self.predictor.relu(self.predictor.fc1(features))
                )
            )
        )
        
        # 예측 모델 출력
        talent_pred = self.predictor(features)
        predictor_loss = self.criterion(talent_pred, talent_labels)
        
        # 적대 모델 출력
        sensitive_pred = self.adversary(representation.detach())
        adversary_loss = self.criterion(sensitive_pred, sensitive_labels)
        
        # 예측 모델 손실: 예측 정확도 ↑ + 적대 모델 손실 ↑
        total_loss = predictor_loss - self.lambda_adv * adversary_loss
        total_loss.backward()
        self.predictor_optimizer.step()
        
        # 적대 모델 학습
        self.adversary_optimizer.zero_grad()
        sensitive_pred = self.adversary(representation)
        adversary_loss = self.criterion(sensitive_pred, sensitive_labels)
        adversary_loss.backward()
        self.adversary_optimizer.step()
        
        return {
            "predictor_loss": predictor_loss.item(),
            "adversary_loss": adversary_loss.item(),
            "total_loss": total_loss.item()
        }
    
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """재능 예측"""
        self.predictor.eval()
        with torch.no_grad():
            return self.predictor(features)


class CausalInferenceModule:
    """인과 추론 기반 잠재력 추정 모듈"""
    
    def __init__(self):
        """인과 추론 모듈 초기화"""
        self.scaler = StandardScaler()
    
    def estimate_potential(
        self,
        observed_data: np.ndarray,
        instrumental_variables: np.ndarray,
        control_variables: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        도구 변수 분석을 통한 순수한 학습 잠재력 추정
        
        Args:
            observed_data: 관찰된 데이터 (예: 학업 성취도)
            instrumental_variables: 도구 변수 (예: 자기주도성)
            control_variables: 통제 변수 (예: 사교육 시간)
            
        Returns:
            잠재력 추정 결과
        """
        # 데이터 정규화
        observed_data_scaled = self.scaler.fit_transform(observed_data.reshape(-1, 1))
        iv_scaled = self.scaler.fit_transform(instrumental_variables.reshape(-1, 1))
        
        # 환경적 요인 통제
        if control_variables is not None:
            control_scaled = self.scaler.fit_transform(control_variables.reshape(-1, 1))
            # 통제 변수의 영향 제거
            from sklearn.linear_model import LinearRegression
            reg = LinearRegression()
            reg.fit(control_scaled, observed_data_scaled)
            residual = observed_data_scaled - reg.predict(control_scaled)
        else:
            residual = observed_data_scaled
        
        # 도구 변수를 통한 순수한 잠재력 추정
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression()
        reg.fit(iv_scaled, residual)
        potential = reg.predict(iv_scaled)
        
        return {
            "potential": potential.flatten(),
            "coefficient": reg.coef_[0][0],
            "r_squared": reg.score(iv_scaled, residual)
        }


class FairnessEnhancementModule:
    """공정성 강화 모듈"""
    
    def __init__(
        self,
        input_dim: int,
        talent_categories: int = 10,
        sensitive_categories: int = 2
    ):
        """
        Args:
            input_dim: 입력 특징 차원
            talent_categories: 재능 카테고리 수
            sensitive_categories: 민감 정보 카테고리 수
        """
        self.debiasing_network = AdversarialDebiasingNetwork(
            input_dim=input_dim,
            talent_categories=talent_categories,
            sensitive_categories=sensitive_categories
        )
        self.causal_inference = CausalInferenceModule()
    
    def train(
        self,
        features: np.ndarray,
        talent_labels: np.ndarray,
        sensitive_labels: np.ndarray,
        epochs: int = 100
    ) -> Dict[str, Any]:
        """
        공정성 강화 모델 학습
        
        Args:
            features: 입력 특징
            talent_labels: 재능 레이블
            sensitive_labels: 민감 정보 레이블
            epochs: 학습 에포크 수
            
        Returns:
            학습 결과
        """
        features_tensor = torch.FloatTensor(features)
        talent_labels_tensor = torch.LongTensor(talent_labels)
        sensitive_labels_tensor = torch.LongTensor(sensitive_labels)
        
        losses = []
        for epoch in range(epochs):
            loss = self.debiasing_network.train_step(
                features_tensor,
                talent_labels_tensor,
                sensitive_labels_tensor
            )
            losses.append(loss)
        
        return {
            "final_loss": losses[-1],
            "all_losses": losses
        }
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """재능 예측 (편향 제거)"""
        features_tensor = torch.FloatTensor(features)
        predictions = self.debiasing_network.predict(features_tensor)
        return predictions.numpy()
    
    def estimate_pure_potential(
        self,
        achievement: np.ndarray,
        self_directedness: np.ndarray,
        control_variables: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        순수한 학습 잠재력 추정
        
        Args:
            achievement: 학업 성취도
            self_directedness: 자기주도성 (도구 변수)
            control_variables: 통제 변수 (예: 사교육 시간)
            
        Returns:
            잠재력 추정 결과
        """
        return self.causal_inference.estimate_potential(
            achievement,
            self_directedness,
            control_variables
        )
    
    def evaluate_fairness(
        self,
        predictions: np.ndarray,
        sensitive_attributes: np.ndarray,
        actual_labels: np.ndarray
    ) -> Dict[str, float]:
        """
        공정성 평가 (χ² 검정)
        
        Args:
            predictions: 예측 결과
            sensitive_attributes: 민감 정보 (예: 성별)
            actual_labels: 실제 레이블
            
        Returns:
            공정성 평가 결과
        """
        # 예측 카테고리 추출
        pred_categories = np.argmax(predictions, axis=1) if predictions.ndim > 1 else predictions
        
        # 성별별 예측 분포 계산
        contingency_table = []
        for sensitive_val in np.unique(sensitive_attributes):
            mask = sensitive_attributes == sensitive_val
            pred_dist = np.bincount(pred_categories[mask], minlength=len(np.unique(pred_categories)))
            contingency_table.append(pred_dist)
        
        # χ² 검정
        chi2, p_value = stats.chi2_contingency(contingency_table)[:2]
        
        # 편향 완화 여부 확인 (p > 0.05이면 편향 없음)
        is_fair = p_value > 0.05
        
        return {
            "chi_squared": chi2,
            "p_value": p_value,
            "is_fair": is_fair,
            "fairness_status": "Fair" if is_fair else "Biased"
        }

