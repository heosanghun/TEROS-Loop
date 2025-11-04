"""
설명가능성 모듈 (Explainability Module)

멀티모달 근거 시각화 및 대화형 반사실적 설명
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from captum.attr import IntegratedGradients, GradientShap
from captum.attr import visualization as viz
import cv2
from scipy import ndimage
import random


class IntegratedGradientsExplainer:
    """Integrated Gradients 해석기 (텍스트용)"""
    
    def __init__(self, model: nn.Module):
        """
        Args:
            model: 해석할 모델
        """
        self.model = model
        self.ig = IntegratedGradients(model)
    
    def explain(
        self,
        input_tensor: torch.Tensor,
        target: int,
        baseline: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        텍스트에 대한 중요도 계산
        
        Args:
            input_tensor: 입력 텐서
            target: 타겟 클래스
            baseline: 기준점 (None이면 zero baseline)
            
        Returns:
            중요도 점수 배열
        """
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
        
        attributions = self.ig.attribute(
            input_tensor,
            baseline=baseline,
            target=target,
            n_steps=50
        )
        
        return attributions.detach().numpy()


class GradCAMExplainer:
    """Grad-CAM 해석기 (이미지용)"""
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model: 해석할 모델
            target_layer: 활성화 맵을 추출할 레이어
        """
        self.model = model
        self.target_layer = target_layer
    
    def explain(
        self,
        input_tensor: torch.Tensor,
        target: int
    ) -> np.ndarray:
        """
        이미지에 대한 Grad-CAM 활성화 맵 생성
        
        Args:
            input_tensor: 입력 이미지 텐서
            target: 타겟 클래스
            
        Returns:
            활성화 맵 (2D 배열)
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        # Backward pass
        self.model.zero_grad()
        output[0, target].backward()
        
        # Gradient 추출
        gradients = self.target_layer.weight.grad
        
        # 평균 풀링
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # 활성화 맵 계산
        activations = self.target_layer.weight
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = torch.relu(cam)
        
        # 정규화
        cam = cam.squeeze().cpu().detach().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam


class AudioAttentionExplainer:
    """오디오 어텐션 맵 해석기"""
    
    def __init__(self, model: nn.Module):
        """
        Args:
            model: 해석할 모델
        """
        self.model = model
    
    def explain(
        self,
        audio_features: np.ndarray,
        target: int
    ) -> np.ndarray:
        """
        오디오에 대한 어텐션 맵 생성
        
        Args:
            audio_features: 오디오 특징
            target: 타겟 클래스
            
        Returns:
            어텐션 점수 배열
        """
        # TODO: 실제 어텐션 메커니즘에서 추출
        # 현재는 더미 어텐션 맵 반환
        attention_scores = np.random.rand(len(audio_features))
        attention_scores = attention_scores / attention_scores.sum()
        
        return attention_scores


class MultimodalSaliencyMap:
    """멀티모달 근거 맵 생성기"""
    
    def __init__(
        self,
        text_model: Optional[nn.Module] = None,
        image_model: Optional[nn.Module] = None,
        audio_model: Optional[nn.Module] = None
    ):
        """
        Args:
            text_model: 텍스트 모델
            image_model: 이미지 모델
            audio_model: 오디오 모델
        """
        self.text_explainer = (
            IntegratedGradientsExplainer(text_model) if text_model else None
        )
        self.image_explainer = None  # TODO: 이미지 모델이 있을 때 초기화
        self.audio_explainer = (
            AudioAttentionExplainer(audio_model) if audio_model else None
        )
    
    def generate_saliency_map(
        self,
        text_data: Optional[Dict[str, Any]] = None,
        image_data: Optional[Dict[str, Any]] = None,
        audio_data: Optional[Dict[str, Any]] = None,
        target: int = 0
    ) -> Dict[str, Any]:
        """
        멀티모달 근거 맵 생성
        
        Args:
            text_data: 텍스트 데이터 (tokens, input_tensor 등)
            image_data: 이미지 데이터 (image_tensor 등)
            audio_data: 오디오 데이터 (audio_features 등)
            target: 타겟 클래스
            
        Returns:
            멀티모달 근거 맵
        """
        saliency_map = {
            "text": None,
            "image": None,
            "audio": None
        }
        
        # 텍스트 근거 맵
        if text_data and self.text_explainer:
            text_tensor = text_data.get("input_tensor")
            if text_tensor is not None:
                text_attributions = self.text_explainer.explain(
                    text_tensor,
                    target
                )
                saliency_map["text"] = {
                    "tokens": text_data.get("tokens", []),
                    "attributions": text_attributions.tolist(),
                    "highlighted_words": self._extract_highlighted_words(
                        text_data.get("tokens", []),
                        text_attributions
                    )
                }
        
        # 이미지 근거 맵
        if image_data and self.image_explainer:
            image_tensor = image_data.get("image_tensor")
            if image_tensor is not None:
                image_cam = self.image_explainer.explain(image_tensor, target)
                saliency_map["image"] = {
                    "saliency_map": image_cam.tolist(),
                    "highlighted_regions": self._extract_highlighted_regions(image_cam)
                }
        
        # 오디오 근거 맵
        if audio_data and self.audio_explainer:
            audio_features = audio_data.get("audio_features")
            if audio_features is not None:
                audio_attention = self.audio_explainer.explain(audio_features, target)
                saliency_map["audio"] = {
                    "attention_scores": audio_attention.tolist(),
                    "highlighted_segments": self._extract_highlighted_segments(
                        audio_attention
                    )
                }
        
        return saliency_map
    
    def _extract_highlighted_words(
        self,
        tokens: List[str],
        attributions: np.ndarray
    ) -> List[Dict[str, Any]]:
        """중요 단어 추출"""
        if len(attributions.shape) > 1:
            attributions = attributions.mean(axis=1)
        
        # 상위 20% 중요 단어 선택
        threshold = np.percentile(attributions, 80)
        highlighted = []
        
        for i, (token, attr) in enumerate(zip(tokens, attributions)):
            if attr > threshold:
                highlighted.append({
                    "token": token,
                    "index": i,
                    "importance": float(attr),
                    "rank": int(np.sum(attributions > attr))
                })
        
        return sorted(highlighted, key=lambda x: x["importance"], reverse=True)
    
    def _extract_highlighted_regions(self, cam: np.ndarray) -> List[Dict[str, Any]]:
        """중요 영역 추출"""
        # CAM을 0-255 범위로 변환
        cam_normalized = (cam * 255).astype(np.uint8)
        
        # 임계값 적용
        threshold = np.percentile(cam_normalized, 80)
        _, binary = cv2.threshold(cam_normalized, threshold, 255, cv2.THRESH_BINARY)
        
        # 연결 컴포넌트 추출
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
        
        regions = []
        for i in range(1, num_labels):  # 0은 배경
            x, y, w, h, area = stats[i]
            if area > 100:  # 최소 영역 크기
                regions.append({
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h),
                    "area": int(area),
                    "importance": float(np.mean(cam[y:y+h, x:x+w]))
                })
        
        return sorted(regions, key=lambda x: x["importance"], reverse=True)
    
    def _extract_highlighted_segments(
        self,
        attention_scores: np.ndarray,
        segment_length: float = 1.0
    ) -> List[Dict[str, Any]]:
        """중요 구간 추출"""
        threshold = np.percentile(attention_scores, 80)
        highlighted_segments = []
        
        current_segment_start = None
        for i, score in enumerate(attention_scores):
            if score > threshold:
                if current_segment_start is None:
                    current_segment_start = i * segment_length
            else:
                if current_segment_start is not None:
                    highlighted_segments.append({
                        "start_time": current_segment_start,
                        "end_time": (i - 1) * segment_length,
                        "importance": float(np.mean(attention_scores[int(current_segment_start / segment_length):i]))
                    })
                    current_segment_start = None
        
        return sorted(highlighted_segments, key=lambda x: x["importance"], reverse=True)


class CounterfactualExplainer:
    """반사실적 설명 생성기"""
    
    def __init__(self, model: nn.Module, mutation_rate: float = 0.1):
        """
        Args:
            model: 예측 모델
            mutation_rate: 변이율 (Genetic Algorithm)
        """
        self.model = model
        self.mutation_rate = mutation_rate
    
    def generate_counterfactual(
        self,
        original_input: np.ndarray,
        target_class: int,
        feature_names: List[str],
        max_iterations: int = 100
    ) -> Dict[str, Any]:
        """
        반사실적 설명 생성 (Genetic Algorithm 기반)
        
        Args:
            original_input: 원본 입력
            target_class: 목표 클래스
            feature_names: 특징 이름 리스트
            max_iterations: 최대 반복 횟수
            
        Returns:
            반사실적 설명
        """
        # 초기 개체군 생성
        population_size = 20
        population = self._initialize_population(original_input, population_size)
        
        best_solution = None
        best_fitness = float('inf')
        
        for iteration in range(max_iterations):
            # 적합도 평가
            fitness_scores = []
            for individual in population:
                fitness = self._evaluate_fitness(
                    individual,
                    original_input,
                    target_class
                )
                fitness_scores.append(fitness)
                
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_solution = individual.copy()
            
            # 선택, 교차, 변이
            population = self._evolve_population(
                population,
                fitness_scores,
                original_input
            )
        
        # 최적 해석 생성
        if best_solution is not None:
            changes = self._identify_changes(original_input, best_solution, feature_names)
            return {
                "original_prediction": self._predict(original_input),
                "counterfactual_prediction": self._predict(best_solution),
                "target_class": target_class,
                "changes": changes,
                "actionable_feedback": self._generate_actionable_feedback(changes)
            }
        
        return {}
    
    def _initialize_population(
        self,
        original: np.ndarray,
        size: int
    ) -> List[np.ndarray]:
        """초기 개체군 생성"""
        population = []
        for _ in range(size):
            individual = original.copy()
            # 랜덤 변이
            mutation_mask = np.random.random(len(individual)) < self.mutation_rate
            individual[mutation_mask] += np.random.normal(0, 0.1, np.sum(mutation_mask))
            population.append(individual)
        return population
    
    def _evaluate_fitness(
        self,
        individual: np.ndarray,
        original: np.ndarray,
        target_class: int
    ) -> float:
        """적합도 평가"""
        # 예측 확률
        prediction = self._predict(individual)
        target_prob = prediction[target_class]
        
        # 원본과의 거리 (변경 최소화)
        distance = np.linalg.norm(individual - original)
        
        # 적합도: 목표 확률 최대화 + 거리 최소화
        fitness = -target_prob + 0.1 * distance
        
        return fitness
    
    def _evolve_population(
        self,
        population: List[np.ndarray],
        fitness_scores: List[float],
        original: np.ndarray
    ) -> List[np.ndarray]:
        """개체군 진화 (선택, 교차, 변이)"""
        # 선택 (상위 50%)
        sorted_indices = np.argsort(fitness_scores)
        elite_size = len(population) // 2
        elite = [population[i] for i in sorted_indices[:elite_size]]
        
        # 교차 및 변이
        new_population = elite.copy()
        while len(new_population) < len(population):
            parent1, parent2 = random.sample(elite, 2)
            child = self._crossover(parent1, parent2)
            child = self._mutate(child, original)
            new_population.append(child)
        
        return new_population
    
    def _crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray
    ) -> np.ndarray:
        """교차"""
        crossover_point = random.randint(0, len(parent1))
        child = np.concatenate([
            parent1[:crossover_point],
            parent2[crossover_point:]
        ])
        return child
    
    def _mutate(
        self,
        individual: np.ndarray,
        original: np.ndarray
    ) -> np.ndarray:
        """변이"""
        mutation_mask = np.random.random(len(individual)) < self.mutation_rate
        individual[mutation_mask] += np.random.normal(0, 0.05, np.sum(mutation_mask))
        return individual
    
    def _identify_changes(
        self,
        original: np.ndarray,
        counterfactual: np.ndarray,
        feature_names: List[str]
    ) -> List[Dict[str, Any]]:
        """변경 사항 식별"""
        changes = []
        for i, (orig_val, cf_val, name) in enumerate(
            zip(original, counterfactual, feature_names)
        ):
            diff = cf_val - orig_val
            if abs(diff) > 0.01:  # 유의미한 변경
                changes.append({
                    "feature": name,
                    "original_value": float(orig_val),
                    "counterfactual_value": float(cf_val),
                    "change": float(diff),
                    "change_percentage": float(diff / (orig_val + 1e-8) * 100)
                })
        return sorted(changes, key=lambda x: abs(x["change"]), reverse=True)
    
    def _generate_actionable_feedback(
        self,
        changes: List[Dict[str, Any]]
    ) -> List[str]:
        """실행 가능한 피드백 생성"""
        feedback = []
        for change in changes[:5]:  # 상위 5개 변경사항
            if change["change"] > 0:
                feedback.append(
                    f"'{change['feature']}'을 {change['change_percentage']:.1f}% 증가시키면 "
                    f"결과가 개선될 수 있습니다."
                )
            else:
                feedback.append(
                    f"'{change['feature']}'을 {abs(change['change_percentage']):.1f}% 감소시키면 "
                    f"결과가 개선될 수 있습니다."
                )
        return feedback
    
    def _predict(self, input_data: np.ndarray) -> np.ndarray:
        """예측"""
        self.model.eval()
        with torch.no_grad():
            input_tensor = torch.FloatTensor(input_data).unsqueeze(0)
            output = self.model(input_tensor)
            return torch.softmax(output, dim=1).squeeze().numpy()


class ExplainabilityModule:
    """설명가능성 모듈"""
    
    def __init__(
        self,
        text_model: Optional[nn.Module] = None,
        image_model: Optional[nn.Module] = None,
        audio_model: Optional[nn.Module] = None,
        prediction_model: Optional[nn.Module] = None
    ):
        """
        Args:
            text_model: 텍스트 모델
            image_model: 이미지 모델
            audio_model: 오디오 모델
            prediction_model: 예측 모델 (반사실적 설명용)
        """
        self.saliency_map_generator = MultimodalSaliencyMap(
            text_model,
            image_model,
            audio_model
        )
        self.counterfactual_explainer = (
            CounterfactualExplainer(prediction_model) if prediction_model else None
        )
    
    def generate_explanation(
        self,
        text_data: Optional[Dict[str, Any]] = None,
        image_data: Optional[Dict[str, Any]] = None,
        audio_data: Optional[Dict[str, Any]] = None,
        target: int = 0
    ) -> Dict[str, Any]:
        """
        설명 생성
        
        Args:
            text_data: 텍스트 데이터
            image_data: 이미지 데이터
            audio_data: 오디오 데이터
            target: 타겟 클래스
            
        Returns:
            설명 결과
        """
        saliency_map = self.saliency_map_generator.generate_saliency_map(
            text_data,
            image_data,
            audio_data,
            target
        )
        
        return {
            "saliency_map": saliency_map,
            "explanation_type": "multimodal_evidence"
        }
    
    def generate_counterfactual_explanation(
        self,
        original_input: np.ndarray,
        target_class: int,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """
        반사실적 설명 생성
        
        Args:
            original_input: 원본 입력
            target_class: 목표 클래스
            feature_names: 특징 이름 리스트
            
        Returns:
            반사실적 설명
        """
        if self.counterfactual_explainer is None:
            return {"error": "Counterfactual explainer not initialized"}
        
        return self.counterfactual_explainer.generate_counterfactual(
            original_input,
            target_class,
            feature_names
        )

