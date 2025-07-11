import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# Streamlit 페이지 설정
st.set_page_config(layout="wide", page_title="외계 행성 중력렌즈 시뮬레이션")

st.title("외계 행성 중력렌즈 효과 시뮬레이션")
st.markdown("중심 항성 주위를 공전하는 외계 행성에 의한 중력렌즈 효과와 광도 변화를 시뮬레이션합니다.")

# --- 사이드바: 시뮬레이션 파라미터 설정 ---
st.sidebar.header("시뮬레이션 설정")
planet_mass_factor = st.sidebar.slider("행성 질량 (중력렌즈 강도)", 0.1, 5.0, 1.0, 0.1)
orbit_radius = st.sidebar.slider("행성 궤도 반지름", 5.0, 15.0, 10.0, 0.5)
animation_speed = st.sidebar.slider("애니메이션 속도", 0.1, 2.0, 1.0, 0.1)

# --- 중력렌즈 광도 증폭률 계산 함수 (단순화된 모델) ---
# 실제 마이크로렌징 공식은 더 복잡하지만, 여기서는 시각적 이해를 돕기 위해 단순화합니다.
# 행성이 중심에 가까워질수록 광도가 증폭됩니다.
def calculate_magnification(distance_from_star, planet_mass_factor):
    # 아인슈타인 반경 (질량에 비례)
    # 실제 아인슈타인 반경은 질량, 거리 등에 따라 달라지지만, 여기서는 시뮬레이션 목적상 단순화
    einstein_radius = 1.0 * planet_mass_factor # 행성 질량 계수에 비례하도록 설정

    # 행성이 중심 항성에 매우 가까울 때 최대 증폭
    if distance_from_star < 0.1: # 너무 가까우면 무한대 발산 방지
        return 1.0 + 5.0 * planet_mass_factor # 최대 증폭값
    
    # 아인슈타인 반경에 대한 상대 거리 u = distance_from_star / einstein_radius
    u = distance_from_star / einstein_radius
    
    # 마이크로렌징 광도 증폭 공식 (u가 0에 가까울수록 증폭)
    # A = (u^2 + 2) / (u * sqrt(u^2 + 4))
    # u가 0에 가까워지면 A는 무한대로 발산하므로, 작은 값에 대한 처리 필요
    if u < 0.01: # u가 매우 작을 때 (거의 중심일 때)
        magnification = 1.0 + 5.0 * planet_mass_factor # 최대 증폭값
    else:
        magnification = (u**2 + 2) / (u * np.sqrt(u**2 + 4))
    
    # 최소 1.0 (증폭 없음)
    return max(1.0, magnification)

# --- Streamlit 세션 상태 초기화 ---
if 'animation_running' not in st.session_state:
    st.session_state.animation_running = False
if 'current_angle' not in st.session_state:
    st.session_state.current_angle = 0
if 'luminosity_history' not in st.session_state:
    st.session_state.luminosity_history = []
if 'time_points_history' not in st.session_state:
    st.session_state.time_points_history = []
if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()

# --- 사용자 조작 UI ---
st.header("수동 조작: 행성 위치 조절")
manual_angle = st.slider("행성 각도 (도)", 0, 360, st.session_state.current_angle, key="manual_angle_slider")

st.header("자동 시뮬레이션")
play_button_label = "애니메이션 정지" if st.session_state.animation_running else "애니메이션 시작"
if st.button(play_button_label, key="play_button"):
    st.session_state.animation_running = not st.session_state.animation_running
    if not st.session_state.animation_running: # 정지 시 현재 각도 저장
        st.session_state.current_angle = manual_angle # 슬라이더의 현재 각도로 동기화
    st.rerun() # 버튼 클릭 시 즉시 UI 업데이트

# --- 시뮬레이션 및 그래프 표시 영역 ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("중력렌즈 시뮬레이션")
    animation_placeholder = st.empty() # 애니메이션이 그려질 곳

with col2:
    st.subheader("광도 변화 그래프")
    luminosity_chart_placeholder = st.empty() # 그래프가 그려질 곳

# --- Matplotlib 초기 설정 ---
fig_anim, ax_anim = plt.subplots(figsize=(6,6))
ax_anim.set_xlim(-orbit_radius*1.5, orbit_radius*1.5)
ax_anim.set_ylim(-orbit_radius*1.5, orbit_radius*1.5)
ax_anim.set_aspect('equal')
ax_anim.set_facecolor('black') # 우주 배경
ax_anim.axis('off')
ax_anim.set_title("행성 공전 및 중력렌즈 효과", color='white')

# 중심 항성 (광원)
star, = ax_anim.plot(0, 0, 'o', color='yellow', markersize=10, label='중심 항성')
# 행성 (렌즈)
planet, = ax_anim.plot([], [], 'o', color='blue', markersize=8, label='외계 행성')

# 광도 그래프 초기화
fig_chart, ax_chart = plt.subplots(figsize=(6,3))
ax_chart.set_xlim(0, 360) # 0도부터 360도까지
ax_chart.set_ylim(0.9, 1.0 + 5.0 * 5.0 + 0.1) # 예상되는 최대 증폭률 + 여유
ax_chart.set_xlabel("행성 각도 (도)")
ax_chart.set_ylabel("광도 증폭률")
ax_chart.grid(True, linestyle='--', alpha=0.6)
luminosity_line, = ax_chart.plot([], [], color='red', linewidth=2)
current_angle_marker, = ax_chart.plot([], [], 'o', color='blue', markersize=8) # 현재 각도 마커

# --- 애니메이션 및 그래프 업데이트 함수 ---
def update_simulation(angle):
    # 행성 위치 계산 (원형 궤도)
    x_planet = orbit_radius * np.cos(np.radians(angle))
    y_planet = orbit_radius * np.sin(np.radians(angle))
    planet.set_data(x_planet, y_planet)

    # 중심 항성과의 거리 (행성과 항성 사이의 유클리드 거리)
    distance_from_star = np.sqrt(x_planet**2 + y_planet**2)
    
    # 광도 계산
    current_luminosity = calculate_magnification(distance_from_star, planet_mass_factor)
    
    # 광도 이력 업데이트
    # 자동 재생 시에만 이력을 쌓고, 수동 조작 시에는 현재 값만 표시
    if st.session_state.animation_running:
        st.session_state.luminosity_history.append(current_luminosity)
        st.session_state.time_points_history.append(angle) # 각도를 시간 축으로 사용

        # 일정 길이 이상이면 오래된 데이터 제거 (그래프 스크롤링 효과)
        max_history_points = 360 # 한 바퀴 (360도) 데이터 유지
        if len(st.session_state.luminosity_history) > max_history_points:
            st.session_state.luminosity_history.pop(0)
            st.session_state.time_points_history.pop(0)
    else: # 수동 조작 시에는 현재 값만 그래프에 표시
        st.session_state.luminosity_history = [current_luminosity]
        st.session_state.time_points_history = [angle]

    luminosity_line.set_data(st.session_state.time_points_history, st.session_state.luminosity_history)
    current_angle_marker.set_data([angle], [current_luminosity]) # 현재 각도에 해당하는 광도에 마커 표시

    # 광원 크기 변화 (중력렌즈 효과 시각화 보조)
    # 행성이 가까이 올수록 항성 크기 살짝 키우기
    star_size = 10 + (current_luminosity - 1.0) * 5 # 증폭률에 따라 크기 조절
    star.set_markersize(star_size)

    # 그래프 Y축 범위 동적 조절 (데이터에 따라)
    if st.session_state.luminosity_history:
        min_lum = min(st.session_state.luminosity_history)
        max_lum = max(st.session_state.luminosity_history)
        ax_chart.set_ylim(max(0.9, min_lum * 0.95), max_lum * 1.05)
    
    # 그래프 X축 범위 동적 조절 (자동 재생 시)
    if st.session_state.animation_running and st.session_state.time_points_history:
        ax_chart.set_xlim(min(st.session_state.time_points_history), max(360, max(st.session_state.time_points_history)))
    elif not st.session_state.animation_running: # 수동 조작 시 전체 범위
        ax_chart.set_xlim(0, 360)


# --- 시뮬레이션 실행 루프 ---
if st.session_state.animation_running:
    # 자동 재생 모드
    while st.session_state.animation_running:
        st.session_state.current_angle = (st.session_state.current_angle + animation_speed) % 360
        update_simulation(st.session_state.current_angle)
        
        with col1:
            animation_placeholder.pyplot(fig_anim)
        with col2:
            luminosity_chart_placeholder.pyplot(fig_chart)
        
        time.sleep(0.05) # 애니메이션 프레임 딜레이
        st.rerun() # Streamlit 앱을 다시 실행하여 UI 업데이트
else:
    # 수동 조작 모드 또는 정지 상태
    update_simulation(manual_angle)
    with col1:
        animation_placeholder.pyplot(fig_anim)
    with col2:
        luminosity_chart_placeholder.pyplot(fig_chart)

st.sidebar.markdown("---")
st.sidebar.markdown("**참고:** 이 시뮬레이션은 중력렌즈 효과를 시각적으로 이해하기 위한 단순화된 모델입니다. 실제 천체 물리학적 계산과는 차이가 있을 수 있습니다.")
