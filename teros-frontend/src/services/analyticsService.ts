/**
 * 분석 API 서비스
 */
import { apiService } from './api';

export interface TalentProfile {
  student_id: string;
  talents: TalentCategory[];
  overall_score: number;
  top_talents: string[];
  career_recommendations: string[];
  learning_path: string[];
  created_at: string;
}

export interface TalentCategory {
  category_id: string;
  category_name: string;
  score: number;
  confidence: number;
  evidence: string[];
}

export interface Explanation {
  saliency_map: {
    text?: any;
    image?: any;
    audio?: any;
  };
  explanation_type: string;
}

export interface Counterfactual {
  original_prediction: number[];
  counterfactual_prediction: number[];
  target_class: number;
  changes: Change[];
  actionable_feedback: string[];
}

export interface Change {
  feature: string;
  original_value: number;
  counterfactual_value: number;
  change: number;
  change_percentage: number;
}

export const analyticsService = {
  async diagnoseTalent(
    studentId: string,
    textFeatures?: number[][],
    imageFeatures?: number[][],
    audioFeatures?: number[][],
    videoFeatures?: number[][],
    baseFeatures?: number[]
  ) {
    return apiService.post<TalentProfile>('/api/v1/analytics/talent-diagnosis', {
      student_id: studentId,
      text_features: textFeatures,
      image_features: imageFeatures,
      audio_features: audioFeatures,
      video_features: videoFeatures,
      base_features: baseFeatures,
    });
  },

  async generateExplanation(
    textData?: any,
    imageData?: any,
    audioData?: any,
    target: number = 0
  ) {
    return apiService.post<Explanation>('/api/v1/analytics/explain', {
      text_data: textData,
      image_data: imageData,
      audio_data: audioData,
      target,
    });
  },

  async generateCounterfactual(
    originalInput: number[],
    targetClass: number,
    featureNames: string[]
  ) {
    return apiService.post<Counterfactual>('/api/v1/analytics/counterfactual', {
      original_input: originalInput,
      target_class: targetClass,
      feature_names: featureNames,
    });
  },

  async addTeacherAnnotation(annotation: any) {
    return apiService.post('/api/v1/analytics/context/annotation', annotation);
  },

  async addMetacognitionReport(report: any) {
    return apiService.post('/api/v1/analytics/context/metacognition', report);
  },

  async getContext(studentId: string, documentId?: string) {
    const endpoint = `/api/v1/analytics/context/${studentId}${documentId ? `?document_id=${documentId}` : ''}`;
    return apiService.get(endpoint);
  },

  async getTags(category?: string) {
    const endpoint = `/api/v1/analytics/context/tags${category ? `?category=${category}` : ''}`;
    return apiService.get(endpoint);
  },

  async evaluateFairness(
    predictions: number[][],
    sensitiveAttributes: number[],
    actualLabels: number[]
  ) {
    return apiService.post('/api/v1/analytics/fairness/evaluate', {
      predictions,
      sensitive_attributes: sensitiveAttributes,
      actual_labels: actualLabels,
    });
  },
};

