#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PDF 분석 및 요약 생성 스크립트
"""

import sys
import os
from pathlib import Path

def install_required_packages():
    """필요한 패키지 설치"""
    try:
        import pypdf
    except ImportError:
        print("pypdf 라이브러리 설치 중...")
        os.system(f"{sys.executable} -m pip install pypdf")
    
    try:
        import pdfplumber
    except ImportError:
        print("pdfplumber 라이브러리 설치 중...")
        os.system(f"{sys.executable} -m pip install pdfplumber")

def extract_text_from_pdf(pdf_path):
    """PDF에서 텍스트 추출"""
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        return text
    except Exception as e:
        print(f"pdfplumber로 추출 실패, pypdf로 시도: {e}")
        try:
            import pypdf
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n\n"
            return text
        except Exception as e2:
            print(f"pypdf로도 추출 실패: {e2}")
            return None

def analyze_pdf_content(text):
    """PDF 내용 분석"""
    if not text:
        return None
    
    lines = text.split('\n')
    non_empty_lines = [line.strip() for line in lines if line.strip()]
    
    # 기본 통계
    word_count = len(text.split())
    char_count = len(text)
    line_count = len(non_empty_lines)
    
    # 섹션 찾기 (제목 패턴)
    sections = []
    for i, line in enumerate(non_empty_lines):
        if len(line) < 100 and (line.isupper() or 
                               any(keyword in line for keyword in ['제목', 'Abstract', '요약', '서론', '결론', '참고문헌', 'Reference'])):
            sections.append((i, line))
    
    analysis = {
        'word_count': word_count,
        'char_count': char_count,
        'line_count': line_count,
        'sections': sections[:20],  # 처음 20개 섹션만
        'preview': non_empty_lines[:50]  # 처음 50줄 미리보기
    }
    
    return analysis

def create_summary(text, analysis):
    """요약본 생성"""
    summary = []
    summary.append("=" * 80)
    summary.append("PDF 문서 분석 요약")
    summary.append("=" * 80)
    summary.append("")
    
    if analysis:
        summary.append("📊 문서 통계:")
        summary.append(f"  - 총 단어 수: {analysis['word_count']:,}개")
        summary.append(f"  - 총 문자 수: {analysis['char_count']:,}개")
        summary.append(f"  - 총 줄 수: {analysis['line_count']:,}줄")
        summary.append("")
    
    summary.append("📑 주요 섹션:")
    if analysis and analysis['sections']:
        for idx, (line_num, section) in enumerate(analysis['sections'], 1):
            summary.append(f"  {idx}. {section}")
    summary.append("")
    
    summary.append("=" * 80)
    summary.append("전체 내용:")
    summary.append("=" * 80)
    summary.append("")
    summary.append(text)
    
    return "\n".join(summary)

def main():
    pdf_path = Path("DOC/자기 진화형 AI 아키텍처 기반의 개인 맞춤형 재능 진단 연구.pdf")
    output_path = Path("DOC/PDF_분석_요약.txt")
    
    if not pdf_path.exists():
        print(f"오류: PDF 파일을 찾을 수 없습니다: {pdf_path}")
        return
    
    print("필요한 패키지 확인 중...")
    install_required_packages()
    
    print(f"PDF 파일 분석 중: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)
    
    if not text:
        print("오류: PDF에서 텍스트를 추출할 수 없습니다.")
        return
    
    print("PDF 내용 분석 중...")
    analysis = analyze_pdf_content(text)
    
    print("요약본 생성 중...")
    summary = create_summary(text, analysis)
    
    print(f"요약본 저장 중: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"✅ 완료! 요약본이 생성되었습니다: {output_path}")
    print(f"\n문서 길이: {len(text):,} 문자")
    if analysis:
        print(f"통계: 단어 {analysis['word_count']:,}개, 줄 {analysis['line_count']:,}개")

if __name__ == "__main__":
    main()

