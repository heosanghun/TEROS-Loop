/**
 * TEROS-Loop API 서비스
 */
import { apiService } from './api';

export interface DiscrepancyCase {
  case_id: string;
  student_id: string;
  prediction: any;
  actual_result: any;
  discrepancy_type: string;
  kl_divergence: number;
  detected_at: string;
  status: string;
}

export interface DeliberationItem {
  item_id: string;
  rule_id: string;
  rule: any;
  accumulated_evidence: any[];
  confidence_score: number;
  teacher_response?: string;
  teacher_feedback?: string;
  created_at: string;
  reviewed_at?: string;
}

export interface TEROSLoopStatistics {
  total_cases: number;
  pending_cases: number;
  analyzing_cases: number;
  resolved_cases: number;
  total_rules: number;
  pending_rules: number;
  approved_rules: number;
  rejected_rules: number;
  pending_deliberation: number;
  ontology_rules: number;
}

export const terosLoopService = {
  async processDiscrepancy(
    studentId: string,
    prediction: any,
    actualResult: any,
    teacherAnnotation?: any
  ) {
    return apiService.post('/api/v1/teros-loop/process-discrepancy', {
      student_id: studentId,
      prediction,
      actual_result: actualResult,
      teacher_annotation: teacherAnnotation,
    });
  },

  async updateConfidence(ruleId: string, observedResult: boolean) {
    return apiService.post('/api/v1/teros-loop/update-confidence', {
      rule_id: ruleId,
      observed_result: observedResult,
    });
  },

  async getPendingDeliberation() {
    return apiService.get<{ items: DeliberationItem[]; count: number }>(
      '/api/v1/teros-loop/deliberation/pending'
    );
  },

  async submitTeacherResponse(
    itemId: string,
    response: 'approve' | 'reject' | 'modify',
    feedback?: string
  ) {
    return apiService.post('/api/v1/teros-loop/deliberation/respond', {
      item_id: itemId,
      response,
      feedback,
    });
  },

  async getStatistics() {
    return apiService.get<{ statistics: TEROSLoopStatistics }>(
      '/api/v1/teros-loop/statistics'
    );
  },

  async getPendingCases() {
    return apiService.get<{ cases: DiscrepancyCase[]; count: number }>(
      '/api/v1/teros-loop/cases/pending'
    );
  },
};

