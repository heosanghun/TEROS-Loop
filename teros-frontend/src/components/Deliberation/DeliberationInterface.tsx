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
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  LinearProgress,
} from '@mui/material';
import { terosLoopService } from '../../services/terosLoopService';
import type { DeliberationItem } from '../../services/terosLoopService';

const DeliberationInterface: React.FC = () => {
  const [items, setItems] = useState<DeliberationItem[]>([]);
  const [selectedItem, setSelectedItem] = useState<DeliberationItem | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [feedback, setFeedback] = useState('');
  const [statistics, setStatistics] = useState<any>(null);

  useEffect(() => {
    loadDeliberationItems();
    loadStatistics();
  }, []);

  const loadDeliberationItems = async () => {
    const response = await terosLoopService.getPendingDeliberation();
    if (response.success && response.data) {
      setItems(response.data.items);
    }
  };

  const loadStatistics = async () => {
    const response = await terosLoopService.getStatistics();
    if (response.success && response.data) {
      setStatistics(response.data.statistics);
    }
  };

  const handleItemClick = (item: DeliberationItem) => {
    setSelectedItem(item);
    setDialogOpen(true);
  };

  const handleSubmitResponse = async (response: 'approve' | 'reject' | 'modify') => {
    if (!selectedItem) return;

    const result = await terosLoopService.submitTeacherResponse(
      selectedItem.item_id,
      response,
      feedback
    );

    if (result.success) {
      setDialogOpen(false);
      setSelectedItem(null);
      setFeedback('');
      loadDeliberationItems();
      loadStatistics();
    }
  };

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        TEROS-Loop 공동 심의 인터페이스
      </Typography>

      {/* 통계 */}
      {statistics && (
        <Paper sx={{ p: 2, mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            TEROS-Loop 통계
          </Typography>
          <Box sx={{ display: 'flex', gap: 3, flexWrap: 'wrap' }}>
            <Box>
              <Typography variant="body2" color="text.secondary">
                전체 불일치 사례
              </Typography>
              <Typography variant="h5">{statistics.total_cases}</Typography>
            </Box>
            <Box>
              <Typography variant="body2" color="text.secondary">
                대기 중인 안건
              </Typography>
              <Typography variant="h5">{statistics.pending_deliberation}</Typography>
            </Box>
            <Box>
              <Typography variant="body2" color="text.secondary">
                승인된 규칙
              </Typography>
              <Typography variant="h5">{statistics.approved_rules}</Typography>
            </Box>
            <Box>
              <Typography variant="body2" color="text.secondary">
                온톨로지 규칙
              </Typography>
              <Typography variant="h5">{statistics.ontology_rules}</Typography>
            </Box>
          </Box>
        </Paper>
      )}

      {/* 안건 목록 */}
      <Paper sx={{ p: 2 }}>
        <Typography variant="h6" gutterBottom>
          검증 요청 안건 ({items.length}건)
        </Typography>
        <TableContainer>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>규칙 ID</TableCell>
                <TableCell>신뢰도</TableCell>
                <TableCell>검증 횟수</TableCell>
                <TableCell>누적 사례</TableCell>
                <TableCell>생성일</TableCell>
                <TableCell>액션</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {items.map((item) => (
                <TableRow key={item.item_id}>
                  <TableCell>{item.rule_id}</TableCell>
                  <TableCell>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <LinearProgress
                        variant="determinate"
                        value={item.confidence_score * 100}
                        sx={{ width: 100, height: 8, borderRadius: 4 }}
                      />
                      <Typography variant="body2">
                        {(item.confidence_score * 100).toFixed(1)}%
                      </Typography>
                    </Box>
                  </TableCell>
                  <TableCell>{item.rule.validation_count}</TableCell>
                  <TableCell>{item.accumulated_evidence.length}</TableCell>
                  <TableCell>
                    {new Date(item.created_at).toLocaleDateString()}
                  </TableCell>
                  <TableCell>
                    <Button
                      size="small"
                      variant="contained"
                      onClick={() => handleItemClick(item)}
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

      {/* 안건 상세 다이얼로그 */}
      <Dialog
        open={dialogOpen}
        onClose={() => setDialogOpen(false)}
        maxWidth="lg"
        fullWidth
      >
        <DialogTitle>지식 규칙 상세 심의</DialogTitle>
        <DialogContent>
          {selectedItem && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="h6" gutterBottom>
                규칙 정보
              </Typography>
              <Paper sx={{ p: 2, mb: 2, bgcolor: 'grey.50' }}>
                <Typography variant="body2" color="text.secondary">
                  규칙 ID: {selectedItem.rule_id}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  신뢰도: {(selectedItem.confidence_score * 100).toFixed(1)}%
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  검증 횟수: {selectedItem.rule.validation_count}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  긍정 사례: {selectedItem.rule.positive_count}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  부정 사례: {selectedItem.rule.negative_count}
                </Typography>
              </Paper>

              <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>
                조건 (IF)
              </Typography>
              <Paper sx={{ p: 2, mb: 2, bgcolor: 'blue.50' }}>
                <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace' }}>
                  {JSON.stringify(selectedItem.rule.condition, null, 2)}
                </pre>
              </Paper>

              <Typography variant="h6" gutterBottom>
                결과 (THEN)
              </Typography>
              <Paper sx={{ p: 2, mb: 2, bgcolor: 'green.50' }}>
                <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace' }}>
                  {JSON.stringify(selectedItem.rule.result, null, 2)}
                </pre>
              </Paper>

              <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>
                누적 사례 데이터
              </Typography>
              <Paper sx={{ p: 2, mb: 2, maxHeight: 200, overflow: 'auto' }}>
                {selectedItem.accumulated_evidence.map((evidence, index) => (
                  <Box key={index} sx={{ mb: 1 }}>
                    <Typography variant="body2">
                      사례 {index + 1}: {evidence.case_id || `Case ${index + 1}`}
                    </Typography>
                  </Box>
                ))}
              </Paper>

              <TextField
                fullWidth
                label="수정 의견 또는 피드백"
                multiline
                rows={4}
                value={feedback}
                onChange={(e) => setFeedback(e.target.value)}
                sx={{ mt: 2 }}
              />
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
    </Container>
  );
};

export default DeliberationInterface;

