import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# --- Streamlit 페이지 설정 ---
st.set_page_config(layout="wide", page_title="외계 행성 중력렌즈 시뮬레이션")

st.title("외계 행성 중력렌즈 효과 시뮬레이션")
st.markdown("중심 항성 주위를 공전하는 외계 행성에 의한 중력렌즈 효과와 광도 변화를 시뮬레이션합니다.")
st.markdown("---")

# --- 사이드바: 시뮬레이션 파라미터 설정 ---
st.sidebar.header("시뮬레이션 설정")
st.sidebar.markdown("각 파라미터를 조절하여 중력렌즈 효과의 변화를 관찰해보세요.")

# 행성 질량 (중력렌즈 강도)
# 이 값이 클수록 광도 증폭 효과가 커집니다.
planet_mass_factor = st.sidebar.slider(
    "행성 질량 (중력렌즈 강도)", # 슬라이더 레이블
    0.1,                          # 최소값
    5.0,                          # 최대값
    1.0,                          # 기본값
    0.1,                          # 스텝 크기
    help="행성의 상대적 질량을 나타내며, 중력렌즈 효과의 강도에 영향을 줍니다."
)

# 행성 궤도 반지름
# 이 값이 클수록 행성이 중심 항성으로부터 멀리 떨어져 공전합니다.
orbit_radius = st.sidebar.slider(
    "행성 궤도 반지름",
    5.0,
    15.0,
    10.0,
    0.5,
    help="행성이 공전하는 궤도의 반지름입니다. 시뮬레이션 화면 크기에 영향을 줍니다."
)

# 애니메이션 속도
# 이 값이 클수록 행성이 더 빠르게 공전합니다.
animation_speed = st.sidebar.slider(
    "애니메이션 속도",
    0.1,
    2.0,
    1.0,
    0.1,
    help="자동 애니메이션 재생 시 행성이 움직이는 속도입니다."
)

st.sidebar.markdown("---")
st.sidebar.markdown("**참고:** 이 시뮬레이션은 중력렌즈 효과를 시각적으로 이해하기 위한 단순화된 모델입니다. 실제 천체 물리학적 계산과는 차이가 있을 수 있습니다.")

# --- 중력렌즈 광도 증폭률 계산 함수 (단순화된 모델) ---
# 마이크로렌징의 광도 증폭 공식: A = (u^2 + 2) / (u * sqrt(u^2 + 4))
# 여기서 u는 아인슈타인 반경에 대한 상대 거리입니다.
# u = distance_from_star / einstein_radius
def calculate_magnification(distance_from_star, planet_mass_factor):
    # 시뮬레이션 목적상 임의의 아인슈타인 반경 설정 (행성 질량 계수에 비례)
    # 실제 아인슈타인 반경은 렌즈 질량, 광원-렌즈-관측자 간의 거리에 따라 달라집니다.
    einstein_radius = 1.0 * planet_mass_factor # 질량 계수가 클수록 아인슈타인 반경이 커집니다.

    # 행성이 중심 항성(광원)에 매우 가까워질 때 (u가 0에 가까워질 때)
    # 광도 증폭률이 무한대로 발산하므로, 작은 값에 대한 처리가 필요합니다.
    u = distance_from_star / einstein_radius

    if u < 0.05: # u가 매우 작을 때 (중심에 매우 가까울 때)
        # 최대 증폭값을 제한하여 시각적으로 보기 좋게 만듭니다.
        magnification = 1.0 + 10.0 * planet_mass_factor # 행성 질량 계수에 비례하여 최대 증폭
    else:
        magnification = (u**2 + 2) / (u * np.sqrt(u**2 + 4))
    
    # 광도 증폭률은 최소 1.0 (증폭 없음)
    return max(1.0, magnification)

# --- Streamlit 세션 상태 초기화 ---
# 앱이 다시 로드되거나 rerunning 될 때도 상태를 유지하기 위함
if 'animation_running' not in st.session_state:
    st.session_state.animation_running = False
if 'current_angle' not in st.session_state:
    st.session_state.current_angle = 0
if 'luminosity_history' not in st.session_state:
    st.session_state.luminosity_history = []
if 'angle_history' not in st.session_state: # 광도 그래프의 X축 (각도) 기록
    st.session_state.angle_history = []
if 'max_luminosity_in_history' not in st.session_state:
    st.session_state.max_luminosity_in_history = 1.0

# --- 사용자 조작 UI ---
st.header("수동 조작: 행성 위치 조절")
# 슬라이더의 기본값을 st.session_state.current_angle로 설정
# int()와 % 360을 사용하여 값이 0-359 범위 내에 있도록 안전장치
manual_angle = st.slider(
    "행성 각도 (도)",
    0,
    360,
    int(st.session_state.current_angle % 360), # 슬라이더 기본값
    key="manual_angle_slider" # 고유 키
)

st.header("자동 시뮬레이션")
play_button_label = "애니메이션 정지" if st.session_state.animation_running else "애니메이션 시작"
if st.button(play_button_label, key="play_button"):
    st.session_state.animation_running = not st.session_state.animation_running
    if not st.session_state.animation_running: # 정지 시 현재 각도 저장 (수동 슬라이더와 동기화)
        st.session_state.current_angle = manual_angle
    # 버튼 클릭 시 즉시 UI 업데이트를 위해 rerun
    st.rerun() 

# --- 시뮬레이션 및 그래프 표시 영역 ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("중력렌즈 시뮬레이션")
    animation_placeholder = st.empty() # 애니메이션이 그려질 곳

with col2:
    st.subheader("광도 변화 그래프")
    luminosity_chart_placeholder = st.empty() # 그래프가 그려질 곳

# --- Matplotlib 초기 설정 ---
# 1. 애니메이션 그림 (plt.figure() 대신 plt.subplots() 사용 권장)
fig_anim, ax_anim = plt.subplots(figsize=(6,6))
ax_anim.set_xlim(-orbit_radius * 1.5, orbit_radius * 1.5)
ax_anim.set_ylim(-orbit_radius * 1.5, orbit_radius * 1.5)
ax_anim.set_aspect('equal') # X, Y축 비율을 동일하게 유지
ax_anim.set_facecolor('black') # 우주 배경색
ax_anim.axis('off') # 축 보이지 않게 설정
ax_anim.set_title("행성 공전 및 중력렌즈 효과", color='white', fontsize=16)

# 중심 항성 (광원) 플롯
star, = ax_anim.plot(0, 0, 'o', color='yellow', markersize=10, label='중심 항성')
# 행성 (렌즈) 플롯
# 단일 점이지만 set_data에서 시퀀스를 기대하므로 초기값도 리스트로 설정
planet, = ax_anim.plot([], [], 'o', color='blue', markersize=8, label='외계 행성')


# 2. 광도 그래프 그림
fig_chart, ax_chart = plt.subplots(figsize=(6,3))
ax_chart.set_xlim(0, 360) # X축 범위: 0도부터 360도까지
# Y축 초기 범위: 최소 0.9 (기본 광도 1.0 아래), 최대 예상 최대 증폭값 + 여유
ax_chart.set_ylim(0.9, 1.0 + 10.0 * 5.0 + 1.0) 
ax_chart.set_xlabel("행성 각도 (도)", fontsize=12)
ax_chart.set_ylabel("광도 증폭률", fontsize=12)
ax_chart.grid(True, linestyle='--', alpha=0.6)
ax_chart.set_title("시간에 따른 광도 변화", fontsize=14)

# 광도 변화를 그릴 라인 플롯
luminosity_line, = ax_chart.plot([], [], color='red', linewidth=2)
# 현재 행성의 각도에 해당하는 광도 위치를 표시할 마커
current_angle_marker, = ax_chart.plot([], [], 'o', color='cyan', markersize=8)


# --- 애니메이션 및 그래프 업데이트 함수 ---
def update_simulation(angle):
    # 1. 행성 위치 계산 (원형 궤도)
    x_planet = orbit_radius * np.cos(np.radians(angle))
    y_planet = orbit_radius * np.sin(np.radians(angle))
    
    # planet.set_data()는 시퀀스를 기대하므로, 단일 값도 리스트로 감싸서 전달
    planet.set_data([x_planet], [y_planet])

    # 2. 중심 항성과의 거리 계산 (행성과 항성 사이의 유클리드 거리)
    distance_from_star = np.sqrt(x_planet**2 + y_planet**2)
    
    # 3. 광도 계산
    current_luminosity = calculate_magnification(distance_from_star, planet_mass_factor)
    
    # 4. 광도 이력 업데이트
    # 자동 재생 시에만 이력을 쌓고, 수동 조작 시에는 현재 값만 표시하여 그래프를 비웁니다.
    if st.session_state.animation_running:
        st.session_state.luminosity_history.append(current_luminosity)
        st.session_state.angle_history.append(angle)

        # 그래프에 표시할 최대 데이터 포인트 수 (예: 한 바퀴 공전 데이터)
        max_history_points = 360 # 0도부터 360도까지의 데이터

        # 데이터가 너무 길어지면 오래된 데이터 제거 (그래프가 스크롤되는 효과)
        if len(st.session_state.luminosity_history) > max_history_points:
            st.session_state.luminosity_history.pop(0)
            st.session_state.angle_history.pop(0)
    else: # 수동 조작 모드 또는 애니메이션 정지 상태
        # 현재 값만 그래프에 표시하여 깨끗한 상태 유지
        st.session_state.luminosity_history = [current_luminosity]
        st.session_state.angle_history = [angle]
    
    # 광도 라인 플롯 업데이트
    luminosity_line.set_data(st.session_state.angle_history, st.session_state.luminosity_history)
    # 현재 각도에 해당하는 광도 위치에 마커 표시
    current_angle_marker.set_data([angle], [current_luminosity])

    # 5. 광원 크기 변화 (중력렌즈 효과 시각화 보조)
    # 행성이 가까이 올수록 항성의 겉보기 크기 살짝 키우기
    star_size = 10 + (current_luminosity - 1.0) * 8 # 증폭률에 따라 크기 조절 (조절 계수 8)
    star.set_markersize(star_size)

    # 6. 광도 그래프 Y축 범위 동적 조절
    if st.session_state.luminosity_history:
        min_lum = min(st.session_state.luminosity_history)
        max_lum = max(st.session_state.luminosity_history)
        
        # Y축 최소/최대 값 설정에 여유를 줌
        ax_chart.set_ylim(max(0.9, min_lum * 0.95), max_lum * 1.15)
        st.session_state.max_luminosity_in_history = max_lum # 최대 증폭값 기록 (나중에 사용 가능)
    
    # 7. 광도 그래프 X축 범위 동적 조절 (자동 재생 시)
    if st.session_state.animation_running and st.session_state.angle_history:
        # 애니메이션 중에는 마지막 각도 근처를 보여주거나 전체 360도를 유지
        ax_chart.set_xlim(min(st.session_state.angle_history), max(360, max(st.session_state.angle_history)))
    elif not st.session_state.animation_running: # 수동 조작 시 전체 0-360 범위 유지
        ax_chart.set_xlim(0, 360)


# --- 시뮬레이션 실행 루프 ---
if st.session_state.animation_running:
    # 자동 재생 모드
    while st.session_state.animation_running:
        # 현재 각도를 업데이트하고 360으로 나눈 나머지 값을 사용 (0~359.99... 유지)
        st.session_state.current_angle = (st.session_state.current_angle + animation_speed) % 360
        
        # 시뮬레이션 및 그래프 업데이트
        update_simulation(st.session_state.current_angle)
        
        # Matplotlib 그림을 Streamlit에 표시
        with col1:
            animation_placeholder.pyplot(fig_anim)
        with col2:
            luminosity_chart_placeholder.pyplot(fig_chart)
        
        # 애니메이션 프레임 간 딜레이 (속도 조절)
        time.sleep(0.05) 
        
        # Streamlit 앱을 다시 실행하여 UI를 업데이트 (매 프레임마다)
        st.rerun()
else:
    # 수동 조작 모드 또는 정지 상태
    # 슬라이더의 현재 값(manual_angle)을 사용하여 시뮬레이션 업데이트
    update_simulation(manual_angle)
    
    # Matplotlib 그림을 Streamlit에 표시
    with col1:
        animation_placeholder.pyplot(fig_anim)
    with col2:
        luminosity_chart_placeholder.pyplot(fig_chart)

# 그래프 그리기 후 Matplotlib 그림을 닫아 메모리 누수 방지
plt.close(fig_anim)
plt.close(fig_chart)
