# GitHub 폴더 업로더 사용 가이드

## 📋 개요

이 도구는 폴더를 드래그 앤 드롭하여 GitHub 저장소에 직접 업로드할 수 있는 웹 애플리케이션입니다.

## 🚀 사용 방법

### 1. GitHub Personal Access Token 생성

1. GitHub에 로그인
2. [Settings → Developer settings → Personal access tokens → Tokens (classic)](https://github.com/settings/tokens) 이동
3. **Generate new token (classic)** 클릭
4. 토큰 이름 입력 (예: "TEROS Uploader")
5. **다음 권한 체크:**
   - ✅ `repo` (전체 저장소 권한)
   - ✅ `public_repo` (공개 저장소 권한, 필요시)
6. **Generate token** 클릭
7. 생성된 토큰을 복사하여 안전한 곳에 보관 (⚠️ 다시 볼 수 없습니다!)

### 2. 웹 애플리케이션 열기

1. `github-uploader.html` 파일을 더블클릭하여 브라우저에서 열기
2. 또는 브라우저에서 파일을 드래그 앤 드롭

### 3. GitHub 설정 입력

- **GitHub Personal Access Token**: 위에서 생성한 토큰 입력
- **GitHub 사용자명**: GitHub 사용자명 (예: `your-username`)
- **저장소 이름**: 업로드할 저장소 이름 (예: `teros-project`)
  - 저장소가 없으면 자동으로 생성됩니다
- **브랜치 이름**: 기본값은 `main` (변경 가능)
- **업로드 경로**: 저장소 내 특정 경로에 업로드하려면 경로 입력 (예: `docs/`)
  - 루트에 업로드하려면 비워두기

### 4. 폴더 업로드

#### 방법 1: 드래그 앤 드롭
1. 업로드할 폴더를 찾기
2. 폴더를 드래그하여 업로더 영역에 드롭

#### 방법 2: 클릭하여 선택
1. 업로더 영역 클릭
2. 파일 탐색기에서 폴더 선택

### 5. 업로드 시작

1. 파일 목록과 폴더 구조 확인
2. **업로드 시작** 버튼 클릭
3. 진행 상황 확인
4. 업로드 완료 후 GitHub 링크에서 확인

## ⚠️ 주의사항

### 파일 크기 제한
- GitHub API는 파일당 최대 100MB 제한
- 큰 파일이 있는 경우 Git LFS 사용 고려

### 폴더 구조
- 폴더 구조가 그대로 유지됩니다
- 하위 폴더도 모두 포함됩니다

### 인코딩
- 텍스트 파일은 UTF-8 인코딩으로 업로드됩니다
- 바이너리 파일도 지원됩니다

### 기존 파일
- 같은 경로에 파일이 이미 있으면 자동으로 업데이트됩니다

## 🔧 문제 해결

### "401 Unauthorized" 오류
- 토큰이 올바른지 확인
- 토큰에 `repo` 권한이 있는지 확인

### "404 Not Found" 오류
- 사용자명과 저장소 이름이 올바른지 확인
- 저장소가 비공개인 경우 토큰에 적절한 권한이 있는지 확인

### "422 Unprocessable Entity" 오류
- 파일 경로에 특수문자가 있는지 확인
- 파일 크기가 100MB를 초과하지 않는지 확인

### 업로드가 느림
- 파일이 많거나 크면 업로드에 시간이 걸릴 수 있습니다
- 네트워크 상태를 확인하세요

## 📝 예시

### TEROS 프로젝트 업로드

1. **GitHub 설정:**
   - Token: `ghp_xxxxxxxxxxxx`
   - 사용자명: `your-username`
   - 저장소: `teros-project`
   - 브랜치: `main`
   - 경로: (비워두기)

2. **폴더 선택:**
   - `D:\AI\TEROS` 폴더 드래그 앤 드롭

3. **결과:**
   - `https://github.com/your-username/teros-project`에 업로드됨

## 🔒 보안

- 토큰은 브라우저에 저장되지 않습니다
- 페이지를 새로고침하면 입력한 정보가 초기화됩니다
- 토큰은 절대 공유하지 마세요
- 사용 후 불필요한 토큰은 삭제하는 것을 권장합니다

## 📞 지원

문제가 발생하면 GitHub Issues에 문의하세요.

---

**Made with ❤️ for TEROS Project**

