import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

# --- 물리 상수 및 파라미터 (단위는 임의로 설정) ---
STAR_RADIUS = 0.5  # 중심 항성 크기
PLANET_RADIUS = 0.1 # 행성 크기
BG_STAR_RADIUS = 0.2 # 배경별 크기

ORBIT_RADIUS = 3.0 # 행성의 공전 궤도 반지름
ORBIT_PERIOD_FRAMES = 100 # 행성이 한 바퀴 도는 데 걸리는 애니메이션 프레임 수

# 배경별의 초기 위치 (중심 항성 뒤에 있다고 가정)
# 실제 렌즈 효과는 시선에 따라 달라지지만, 여기서는 고정된 배경으로 설정
BG_STAR_X = 0.0
BG_STAR_Y = -5.0 # 중심 항성보다 아래에 위치

# --- 중력렌즈 광도 변화 모델 (매우 단순화된 근사) ---
# 실제 미세중력렌즈는 복잡한 계산이 필요하며, 여기서는 개념적인 변화를 보여줍니다.
def calculate_magnification(planet_x, planet_y):
    # 중심 항성(0,0)과 배경별(BG_STAR_X, BG_STAR_Y) 사이의 시선에 행성이 가까워질 때
    # 광도가 증가하는 것을 시뮬레이션합니다.
    
    # 행성과 배경별의 상대적인 거리 (시선 방향으로의 근접성)
    # 여기서는 행성의 Y좌표가 배경별의 Y좌표에 가까워질수록, 그리고 X좌표가 0에 가까워질수록
    # 렌즈 효과가 강해진다고 가정합니다.
    
    # 배경별과 행성 사이의 "가상의" 거리 (시선 방향으로의 정렬 정도)
    # 행성이 (0, BG_STAR_Y)에 가까워질수록 강한 렌즈 효과를 낸다고 가정
    dist_to_alignment = np.sqrt((planet_x - BG_STAR_X)**2 + (planet_y - BG_STAR_Y)**2)
    
    # 기본 광도 (렌즈 효과 없을 때)
    base_magnification = 1.0
    
    # 행성이 시선에 가까워질수록 광도 증가 (가우시안 함수 형태)
    # dist_to_alignment가 작을수록 (가까울수록) magnification_boost가 커집니다.
    magnification_boost = np.exp(-dist_to_alignment**2 / 0.5) * 0.5 # 최대 0.5 증가
    
    total_magnification = base_magnification + magnification_boost
    
    return total_magnification

# --- Streamlit 앱 시작 ---
st.set_page_config(layout="wide")
st.title("외계행성 탐사: 중력렌즈 효과 시뮬레이션 (Plotly)")

st.markdown("""
이 시뮬레이션은 배경별 중심 항성 주위를 공전하는 외계행성에 의한 **미세중력렌즈(Microlensing)** 효과를 개념적으로 보여줍니다.
'Play' 버튼을 누르면 외계행성이 공전하며, 배경별의 겉보기 밝기(광도)가 어떻게 변하는지 그래프로 확인할 수 있습니다.
""")

# --- 레이아웃 설정 ---
col1, col2 = st.columns(2)

with col1:
    st.header("1. 중력렌즈 시각화")
    # 렌즈 시각화 Plotly Figure 초기화
    fig_lensing = go.Figure()
    
    # 중심 항성 (렌즈)
    fig_lensing.add_trace(go.Scatter(
        x=[0], y=[0], mode='markers',
        marker=dict(size=STAR_RADIUS*50, color='orange', symbol='circle'),
        name='중심 항성 (렌즈)'
    ))
    
    # 배경별
    fig_lensing.add_trace(go.Scatter(
        x=[BG_STAR_X], y=[BG_STAR_Y], mode='markers',
        marker=dict(size=BG_STAR_RADIUS*50, color='yellow', symbol='star'),
        name='배경별'
    ))
    
    # 외계행성 (초기 위치)
    planet_x_init = ORBIT_RADIUS
    planet_y_init = 0
    fig_lensing.add_trace(go.Scatter(
        x=[planet_x_init], y=[planet_y_init], mode='markers',
        marker=dict(size=PLANET_RADIUS*50, color='blue', symbol='circle'),
        name='외계행성'
    ))
    
    fig_lensing.update_layout(
        xaxis_range=[-ORBIT_RADIUS*1.5, ORBIT_RADIUS*1.5],
        yaxis_range=[-ORBIT_RADIUS*1.5, ORBIT_RADIUS*1.5],
        width=500, height=500,
        showlegend=True,
        xaxis_title="X 위치",
        yaxis_title="Y 위치",
        plot_bgcolor='black',
        paper_bgcolor='black',
        font_color='white'
    )
    
    lensing_plot_placeholder = st.empty() # 렌즈 시각화 업데이트를 위한 placeholder

with col2:
    st.header("2. 배경별 광도 변화")
    # 광도 변화 그래프 Plotly Figure 초기화
    fig_light_curve = go.Figure()
    fig_light_curve.add_trace(go.Scatter(
        x=[], y=[], mode='lines+markers', name='광도',
        line=dict(color='lime', width=2),
        marker=dict(color='lime', size=5)
    ))
    fig_light_curve.update_layout(
        xaxis_title="시간 (프레임)",
        yaxis_title="광도 (배경별 밝기)",
        yaxis_range=[0.9, 1.6], # 광도 범위 설정
        width=500, height=500,
        showlegend=False,
        plot_bgcolor='black',
        paper_bgcolor='black',
        font_color='white'
    )
    light_curve_plot_placeholder = st.empty() # 광도 그래프 업데이트를 위한 placeholder

st.markdown("---")
st.header("시뮬레이션 제어")
play_button = st.button("▶️ Play 시뮬레이션")

# --- 애니메이션 로직 ---
if play_button:
    magnifications = []
    
    for frame in range(ORBIT_PERIOD_FRAMES):
        # 행성 위치 업데이트 (원형 궤도)
        angle = 2 * np.pi * frame / ORBIT_PERIOD_FRAMES
        planet_x = ORBIT_RADIUS * np.cos(angle)
        planet_y = ORBIT_RADIUS * np.sin(angle)
        
        # 광도 계산
        current_magnification = calculate_magnification(planet_x, planet_y)
        magnifications.append(current_magnification)
        
        # 렌즈 시각화 업데이트
        fig_lensing.data[2].x = [planet_x]
        fig_lensing.data[2].y = [planet_y]
        
        lensing_plot_placeholder.plotly_chart(fig_lensing, use_container_width=True)
        
        # 광도 그래프 업데이트
        fig_light_curve.data[0].x = list(range(len(magnifications)))
        fig_light_curve.data[0].y = magnifications
        
        light_curve_plot_placeholder.plotly_chart(fig_light_curve, use_container_width=True)
        
        time.sleep(0.05) # 애니메이션 속도 조절

st.markdown("""
### 시뮬레이션 설명
* **중심 항성 (주황색 원):** 렌즈 역할을 하는 별입니다.
* **외계행성 (파란색 원):** 중심 항성 주위를 공전하며 미세중력렌즈 효과를 유발합니다.
* **배경별 (노란색 별):** 멀리 떨어져 있는 광원 별입니다.
* **광도 변화 그래프:** 외계행성이 배경별과 중심 항성 사이의 시선에 정렬될 때, 배경별의 빛이 중력렌즈 효과로 인해 증폭되어 겉보기 밝기(광도)가 일시적으로 증가하는 것을 보여줍니다. 이 광도 변화 패턴을 통해 외계행성의 존재를 유추할 수 있습니다.

**참고:** 이 시뮬레이션은 중력렌즈 효과의 개념을 설명하기 위한 **매우 단순화된 모델**입니다. 실제 천체물리학적 계산은 훨씬 복잡하며, 빛의 왜곡된 이미지나 아인슈타인 링/십자가 같은 현상을 정확히 렌더링하려면 고급 물리 시뮬레이션 및 그래픽 기술이 필요합니다.
""")
