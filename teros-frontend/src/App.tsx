import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Container,
  Box,
  Tabs,
  Tab,
} from '@mui/material';
import TeacherDashboard from './components/TeacherDashboard/TeacherDashboard';
import StudentDashboard from './components/StudentDashboard/StudentDashboard';
import DeliberationInterface from './components/Deliberation/DeliberationInterface';

function App() {
  const [currentTab, setCurrentTab] = useState(0);

  return (
    <Router>
      <Box sx={{ flexGrow: 1 }}>
        <AppBar position="static">
          <Toolbar>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              TEROS - 멀티모달 Agentic AI System
            </Typography>
            <Button color="inherit" component={Link} to="/teacher">
              교사 대시보드
            </Button>
            <Button color="inherit" component={Link} to="/student">
              학생 대시보드
            </Button>
            <Button color="inherit" component={Link} to="/deliberation">
              공동 심의
            </Button>
          </Toolbar>
        </AppBar>

        <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
          <Routes>
            <Route path="/" element={<TeacherDashboard />} />
            <Route path="/teacher" element={<TeacherDashboard />} />
            <Route
              path="/student"
              element={<StudentDashboard studentId="student_001" />}
            />
            <Route path="/deliberation" element={<DeliberationInterface />} />
          </Routes>
        </Container>
      </Box>
    </Router>
  );
}

export default App;
