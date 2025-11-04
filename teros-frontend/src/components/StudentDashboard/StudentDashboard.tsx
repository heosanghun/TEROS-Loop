import React, { useState, useEffect } from 'react';
import {
  Box,
  Container,
  Typography,
  Paper,
  Card,
  CardContent,
  Grid,
  LinearProgress,
  Button,
  TextField,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Chip,
} from '@mui/material';
import { Radar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
} from 'chart.js';
import { analyticsService } from '../../services/analyticsService';
import type { TalentProfile } from '../../services/analyticsService';

ChartJS.register(
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend
);

const StudentDashboard: React.FC<{ studentId: string }> = ({ studentId }) => {
  const [talentProfile, setTalentProfile] = useState<TalentProfile | null>(null);
  const [metacognitionText, setMetacognitionText] = useState('');
  const [metacognitionDialogOpen, setMetacognitionDialogOpen] = useState(false);
  const [counterfactualQuestion, setCounterfactualQuestion] = useState('');
  const [counterfactualDialogOpen, setCounterfactualDialogOpen] = useState(false);
  const [counterfactualResult, setCounterfactualResult] = useState<any>(null);

  useEffect(() => {
    loadTalentProfile();
  }, [studentId]);

  const loadTalentProfile = async () => {
    const response = await analyticsService.diagnoseTalent(studentId);
    if (response.success && response.data) {
      setTalentProfile(response.data);
    }
  };

  const handleSubmitMetacognition = async () => {
    const report = {
      report_id: `report_${Date.now()}`,
      student_id: studentId,
      assignment_id: 'assignment_001',
      content: metacognitionText,
      difficulty_points: [],
      learned_points: [],
      reflection: metacognitionText,
    };

    const response = await analyticsService.addMetacognitionReport(report);
    if (response.success) {
      setMetacognitionDialogOpen(false);
      setMetacognitionText('');
    }
  };

  const handleAskCounterfactual = async () => {
    // TODO: 실제 반사실적 설명 생성
    // 현재는 더미 응답
    setCounterfactualResult({
      actionable_feedback: [
        '보고서에 구체적인 데이터 예시를 2개 이상 추가하면 논리성 점수가 향상될 수 있습니다.',
        '발표 자료의 시각적 구성을 개선하면 공간·시각 재능 점수가 상승할 수 있습니다.',
      ],
    });
    setCounterfactualDialogOpen(true);
  };

  const radarData = talentProfile
    ? {
        labels: talentProfile.talents.map((t) => t.category_name),
        datasets: [
          {
            label: '재능 점수',
            data: talentProfile.talents.map((t) => t.score),
            backgroundColor: 'rgba(102, 126, 234, 0.2)',
            borderColor: 'rgba(102, 126, 234, 1)',
            borderWidth: 2,
          },
        ],
      }
    : null;

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        내 재능 프로파일
      </Typography>

      {talentProfile ? (
        <Grid container spacing={3}>
          {/* 재능 레이더 차트 */}
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                재능 분포 (게임 캐릭터 스타일)
              </Typography>
              {radarData && <Radar data={radarData} />}
            </Paper>
          </Grid>

          {/* 재능 카테고리 상세 */}
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                재능 상세
              </Typography>
              <Box sx={{ mt: 2 }}>
                {talentProfile.talents.map((talent) => (
                  <Box key={talent.category_id} sx={{ mb: 2 }}>
                    <Box
                      sx={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        mb: 1,
                      }}
                    >
                      <Typography variant="body1">{talent.category_name}</Typography>
                      <Typography variant="body1" fontWeight="bold">
                        {talent.score.toFixed(1)}점
                      </Typography>
                    </Box>
                    <LinearProgress
                      variant="determinate"
                      value={talent.score}
                      sx={{ height: 8, borderRadius: 4 }}
                    />
                  </Box>
                ))}
              </Box>
            </Paper>
          </Grid>

          {/* 진로 추천 */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  진로 추천
                </Typography>
                <Box sx={{ mt: 2 }}>
                  {talentProfile.career_recommendations.map((career) => (
                    <Chip
                      key={career}
                      label={career}
                      color="primary"
                      sx={{ mr: 1, mb: 1 }}
                    />
                  ))}
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* 학습 경로 */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  추천 학습 경로
                </Typography>
                <Box sx={{ mt: 2 }}>
                  {talentProfile.learning_path.map((path, index) => (
                    <Typography key={index} variant="body2" sx={{ mb: 1 }}>
                      {index + 1}. {path}
                    </Typography>
                  ))}
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* 액션 버튼 */}
          <Grid item xs={12}>
            <Box sx={{ display: 'flex', gap: 2 }}>
              <Button
                variant="contained"
                onClick={() => setMetacognitionDialogOpen(true)}
              >
                메타인지 노트 작성
              </Button>
              <Button
                variant="outlined"
                onClick={() => setCounterfactualDialogOpen(true)}
              >
                반사실적 설명 조회
              </Button>
            </Box>
          </Grid>
        </Grid>
      ) : (
        <Typography>재능 프로파일을 불러오는 중...</Typography>
      )}

      {/* 메타인지 노트 다이얼로그 */}
      <Dialog
        open={metacognitionDialogOpen}
        onClose={() => setMetacognitionDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>메타인지 노트 작성</DialogTitle>
        <DialogContent>
          <TextField
            fullWidth
            label="어려웠던 점, 새로 배운 점, 성찰 내용을 작성하세요"
            multiline
            rows={6}
            value={metacognitionText}
            onChange={(e) => setMetacognitionText(e.target.value)}
            sx={{ mt: 2 }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setMetacognitionDialogOpen(false)}>취소</Button>
          <Button onClick={handleSubmitMetacognition} variant="contained">
            제출
          </Button>
        </DialogActions>
      </Dialog>

      {/* 반사실적 설명 다이얼로그 */}
      <Dialog
        open={counterfactualDialogOpen}
        onClose={() => setCounterfactualDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>반사실적 설명</DialogTitle>
        <DialogContent>
          <TextField
            fullWidth
            label="만약 ~했다면 결과가 어떻게 바뀌었을까?"
            multiline
            rows={3}
            value={counterfactualQuestion}
            onChange={(e) => setCounterfactualQuestion(e.target.value)}
            sx={{ mt: 2 }}
          />
          {counterfactualResult && (
            <Box sx={{ mt: 3 }}>
              <Typography variant="h6" gutterBottom>
                개선 제안:
              </Typography>
              {counterfactualResult.actionable_feedback?.map(
                (feedback: string, index: number) => (
                  <Typography key={index} variant="body2" sx={{ mb: 1 }}>
                    • {feedback}
                  </Typography>
                )
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCounterfactualDialogOpen(false)}>닫기</Button>
          <Button onClick={handleAskCounterfactual} variant="contained">
            질문하기
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default StudentDashboard;

