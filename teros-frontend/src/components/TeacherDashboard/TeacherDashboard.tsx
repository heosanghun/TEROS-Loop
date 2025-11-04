import React, { useState, useEffect } from 'react';
import {
  Box,
  Container,
  Typography,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Button,
  Card,
  CardContent,
  Grid,
  TextField,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import { analyticsService } from '../../services/analyticsService';
import { terosLoopService } from '../../services/terosLoopService';
import type { TalentProfile, DeliberationItem } from '../../services/analyticsService';

interface Student {
  id: string;
  name: string;
  talentProfile?: TalentProfile;
}

const TeacherDashboard: React.FC = () => {
  const [students, setStudents] = useState<Student[]>([]);
  const [selectedStudent, setSelectedStudent] = useState<Student | null>(null);
  const [deliberationItems, setDeliberationItems] = useState<DeliberationItem[]>([]);
  const [selectedItem, setSelectedItem] = useState<DeliberationItem | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [annotationDialogOpen, setAnnotationDialogOpen] = useState(false);
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [freeText, setFreeText] = useState('');

  useEffect(() => {
    loadStudents();
    loadDeliberationItems();
  }, []);

  const loadStudents = async () => {
    // TODO: 실제 학생 목록 API 호출
    const mockStudents: Student[] = [
      { id: 'student_001', name: '김철수' },
      { id: 'student_002', name: '이영희' },
      { id: 'student_003', name: '박민수' },
    ];
    setStudents(mockStudents);
  };

  const loadDeliberationItems = async () => {
    const response = await terosLoopService.getPendingDeliberation();
    if (response.success && response.data) {
      setDeliberationItems(response.data.items);
    }
  };

  const handleStudentSelect = async (student: Student) => {
    setSelectedStudent(student);
    // 재능 프로파일 로드
    const response = await analyticsService.diagnoseTalent(student.id);
    if (response.success && response.data) {
      setStudents(prev =>
        prev.map(s =>
          s.id === student.id ? { ...s, talentProfile: response.data } : s
        )
      );
      setSelectedStudent({ ...student, talentProfile: response.data });
    }
  };

  const handleDeliberationItemClick = (item: DeliberationItem) => {
    setSelectedItem(item);
    setDialogOpen(true);
  };

  const handleSubmitResponse = async (response: 'approve' | 'reject' | 'modify') => {
    if (!selectedItem) return;

    const feedback = response === 'modify' ? freeText : undefined;
    const result = await terosLoopService.submitTeacherResponse(
      selectedItem.item_id,
      response,
      feedback
    );

    if (result.success) {
      setDialogOpen(false);
      setSelectedItem(null);
      loadDeliberationItems();
    }
  };

  const handleAddAnnotation = async () => {
    if (!selectedStudent || selectedTags.length === 0) return;

    const annotation = {
      annotation_id: `annotation_${Date.now()}`,
      student_id: selectedStudent.id,
      teacher_id: 'teacher_001',
      document_id: 'document_001',
      document_type: 'text',
      tags: selectedTags,
      free_text: freeText,
    };

    const response = await analyticsService.addTeacherAnnotation(annotation);
    if (response.success) {
      setAnnotationDialogOpen(false);
      setSelectedTags([]);
      setFreeText('');
    }
  };

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        교사 대시보드
      </Typography>

      <Grid container spacing={3}>
        {/* 학생 목록 */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              학생 목록
            </Typography>
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>이름</TableCell>
                    <TableCell>재능 점수</TableCell>
                    <TableCell>상세</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {students.map((student) => (
                    <TableRow
                      key={student.id}
                      onClick={() => handleStudentSelect(student)}
                      sx={{ cursor: 'pointer' }}
                    >
                      <TableCell>{student.name}</TableCell>
                      <TableCell>
                        {student.talentProfile
                          ? `${student.talentProfile.overall_score.toFixed(1)}점`
                          : 'N/A'}
                      </TableCell>
                      <TableCell>
                        <Button size="small">보기</Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        </Grid>

        {/* 재능 프로파일 */}
        {selectedStudent?.talentProfile && (
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                {selectedStudent.name}의 재능 프로파일
              </Typography>
              <Box sx={{ mt: 2 }}>
                <Typography variant="body2" gutterBottom>
                  전체 점수: {selectedStudent.talentProfile.overall_score.toFixed(1)}점
                </Typography>
                <Typography variant="body2" gutterBottom>
                  상위 재능:
                </Typography>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 1 }}>
                  {selectedStudent.talentProfile.top_talents.map((talent) => (
                    <Chip key={talent} label={talent} color="primary" />
                  ))}
                </Box>
                <Typography variant="body2" gutterBottom sx={{ mt: 2 }}>
                  진로 추천:
                </Typography>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 1 }}>
                  {selectedStudent.talentProfile.career_recommendations.map(
                    (career) => (
                      <Chip key={career} label={career} color="secondary" />
                    )
                  )}
                </Box>
                <Button
                  variant="outlined"
                  sx={{ mt: 2 }}
                  onClick={() => setAnnotationDialogOpen(true)}
                >
                  맥락 정보 추가
                </Button>
              </Box>
            </Paper>
          </Grid>
        )}

        {/* TEROS-Loop 안건 심의 */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              TEROS-Loop 안건 심의 ({deliberationItems.length}건)
            </Typography>
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>규칙 ID</TableCell>
                    <TableCell>신뢰도</TableCell>
                    <TableCell>검증 횟수</TableCell>
                    <TableCell>상태</TableCell>
                    <TableCell>액션</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {deliberationItems.map((item) => (
                    <TableRow key={item.item_id}>
                      <TableCell>{item.rule_id}</TableCell>
                      <TableCell>
                        {(item.confidence_score * 100).toFixed(1)}%
                      </TableCell>
                      <TableCell>{item.rule.validation_count}</TableCell>
                      <TableCell>
                        <Chip
                          label={item.rule.status}
                          color={
                            item.rule.status === 'ready_for_review'
                              ? 'warning'
                              : 'default'
                          }
                        />
                      </TableCell>
                      <TableCell>
                        <Button
                          size="small"
                          onClick={() => handleDeliberationItemClick(item)}
                        >
                          심의하기
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        </Grid>
      </Grid>

      {/* 심의 다이얼로그 */}
      <Dialog open={dialogOpen} onClose={() => setDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>지식 규칙 심의</DialogTitle>
        <DialogContent>
          {selectedItem && (
            <Box>
              <Typography variant="body1" gutterBottom>
                규칙 ID: {selectedItem.rule_id}
              </Typography>
              <Typography variant="body1" gutterBottom>
                신뢰도: {(selectedItem.confidence_score * 100).toFixed(1)}%
              </Typography>
              <Typography variant="body1" gutterBottom>
                조건 (IF):
              </Typography>
              <Paper sx={{ p: 2, mb: 2, bgcolor: 'grey.100' }}>
                <pre>{JSON.stringify(selectedItem.rule.condition, null, 2)}</pre>
              </Paper>
              <Typography variant="body1" gutterBottom>
                결과 (THEN):
              </Typography>
              <Paper sx={{ p: 2, mb: 2, bgcolor: 'grey.100' }}>
                <pre>{JSON.stringify(selectedItem.rule.result, null, 2)}</pre>
              </Paper>
              <Typography variant="body1" gutterBottom>
                누적 사례: {selectedItem.accumulated_evidence.length}건
              </Typography>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDialogOpen(false)}>취소</Button>
          <Button
            onClick={() => handleSubmitResponse('reject')}
            color="error"
          >
            거부
          </Button>
          <Button
            onClick={() => handleSubmitResponse('modify')}
            color="warning"
          >
            수정
          </Button>
          <Button
            onClick={() => handleSubmitResponse('approve')}
            color="primary"
            variant="contained"
          >
            승인
          </Button>
        </DialogActions>
      </Dialog>

      {/* 맥락 정보 추가 다이얼로그 */}
      <Dialog
        open={annotationDialogOpen}
        onClose={() => setAnnotationDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>맥락 정보 추가</DialogTitle>
        <DialogContent>
          <TextField
            fullWidth
            label="자유 텍스트 주석"
            multiline
            rows={4}
            value={freeText}
            onChange={(e) => setFreeText(e.target.value)}
            sx={{ mt: 2 }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAnnotationDialogOpen(false)}>취소</Button>
          <Button onClick={handleAddAnnotation} variant="contained">
            추가
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default TeacherDashboard;

