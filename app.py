# app.py
# ... (most of the file is unchanged, only the simulation loop logic needs updating) ...
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import time
import pandas as pd
import sys, os
from shapely.geometry import LineString, Polygon
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from environment import Environment, Building, WeatherSystem
from ml_predictor.predictor import EnergyTimePredictor
from path_planner import PathPlanner3D

# --- App State Initialization & Helper Functions (Unchanged) ---
@st.cache_resource
def load_planner():
    # ...
    log_event("Loading planner and environment...")
    initial_weather=WeatherSystem(scale=150.0, max_speed=10.0)
    env=Environment(weather_system=initial_weather)
    predictor=EnergyTimePredictor()
    planner=PathPlanner3D(env, predictor)
    log_event("Planner loaded successfully.")
    return planner
defaults={'stage':'setup','log':[],'mission_running':False,'planned_path':None,'mission_log':[],'total_time':0.0,'total_energy':0.0,'drone_pos':None,'path_index':0,'initial_payload':0.0}
for k,v in defaults.items():
    if k not in st.session_state:st.session_state[k]=v
def log_event(m):st.session_state.log.insert(0,f"{time.strftime('%H:%M:%S')} - {m}")
def reset_mission_state():
    for k,v in defaults.items():
        if k not in['stage','log']:st.session_state[k]=v
    st.session_state.stage='setup'
planner=load_planner()
st.set_page_config(layout="wide");st.title("🚁 Q-DOP: Dynamic Replanning with LPA*")
def create_box(b):
    # ...
    x,y=b.center_xy;dx,dy=b.size_xy[0]/2,b.size_xy[1]/2;h=b.height;x_c=[x-dx,x+dx,x+dx,x-dx,x-dx,x+dx,x+dx,x-dx];y_c=[y-dy,y-dy,y+dy,y+dy,y-dy,y-dy,y+dy,y+dy];z_c=[0,0,0,0,h,h,h,h];return go.Mesh3d(x=x_c,y=y_c,z=z_c,i=[7,0,0,0,4,4,6,6,4,0,3,2],j=[3,4,1,2,5,6,5,2,0,1,6,3],k=[0,7,2,3,6,7,2,5,1,2,5,6],color='grey',opacity=0.7,name='Building',hoverinfo='none')
def create_nfz_box(z,c='red',o=0.15):
    # ...
    x_c=[z[0],z[2],z[2],z[0],z[0],z[2],z[2],z[0]];y_c=[z[1],z[1],z[3],z[3],z[1],z[1],z[3],z[3]];z_c=[0,0,0,0,config.MAX_ALTITUDE,config.MAX_ALTITUDE,config.MAX_ALTITUDE,config.MAX_ALTITUDE];return go.Mesh3d(x=x_c,y=y_c,z=z_c,color=c,opacity=o,name='No-Fly Zone',hoverinfo='name')
def check_replanning_triggers(current_pos, next_pos, planner):
    segment=LineString([current_pos[:2],next_pos[:2]])
    for nfz in planner.env.dynamic_nfzs:
        if planner.env.was_nfz_just_added:
            zone_poly=Polygon([(nfz[0],nfz[1]),(nfz[2],nfz[1]),(nfz[2],nfz[3]),(nfz[0],nfz[3])])
            if segment.intersects(zone_poly):
                log_event("🚨 Path invalid! Intersects new NFZ. Replanning with LPA*...")
                return True
    return False
# --- App Stages (Unchanged from previous version) ---
def setup_stage():
    st.header("1. Mission Parameters");# ... (UI code is the same)
    if not st.session_state.log:log_event("App initialized.")
    c1,c2=st.columns(2)
    with c1:st.session_state.hub_location=config.HUBS[st.selectbox("Hub",list(config.HUBS.keys()))];st.session_state.destination=config.DESTINATIONS[st.selectbox("Destination",list(config.DESTINATIONS.keys()))]
    with c2:st.session_state.payload_kg=st.slider("Payload (kg)",0.1,config.DRONE_MAX_PAYLOAD_KG,1.5,0.1)
    st.subheader("2. Environmental Conditions");wc1,wc2=st.columns(2)
    with wc1:st.session_state.max_wind_speed=st.slider("Max Wind (m/s)",0.0,25.0,10.0,0.5)
    with wc2:st.session_state.wind_complexity=st.slider("Wind Complexity",10.0,500.0,150.0,10.0)
    st.subheader("3. Optimization Priority");st.session_state.optimization_preference=st.radio("Optimize For:",["Balanced","Fastest Path","Most Battery Efficient"],horizontal=True,index=0)
    if st.session_state.optimization_preference=="Balanced":st.session_state.balance_weight=st.slider("Priority",0.0,1.0,0.5,0.05,format="%.2f",help="0.0=Energy, 1.0=Time")
    if st.button("🚀 Plan Initial Mission",type="primary",use_container_width=True):st.session_state.stage='planning';st.rerun()

def planning_stage():
    # ... (This logic is the same)
    log_event(f"Planning initial mission...")
    with st.spinner(f"Optimizing initial path with LPA*..."):
        planner.env.weather.max_speed=st.session_state.max_wind_speed;planner.env.weather.scale=st.session_state.wind_complexity
        m={"Fastest Path":"time","Most Battery Efficient":"energy","Balanced":"balanced"};sm=m[st.session_state.optimization_preference];bw=st.session_state.get('balance_weight',0.5)
        hub,order=st.session_state.hub_location,st.session_state.destination;payload=st.session_state.payload_kg;st.session_state.initial_payload=payload
        takeoff=(hub[0],hub[1],config.TAKEOFF_ALTITUDE)
        path,status=planner.find_path(takeoff,order,payload,sm,bw) # Initial plan to destination
        if path is None:st.error(f"Fatal Error: Path planning failed. Status: {status}.");st.button("⬅️ New",on_click=reset_mission_state);return
        # Create full round trip path for simulation
        full_path=[hub]+path+[hub] # Simplified for LPA* demo
        st.session_state.planned_path=full_path;st.session_state.drone_pos=full_path[0];st.session_state.path_index=0;log_event("✅ Initial mission plan found.")
    st.session_state.stage='simulation';st.rerun()

def simulation_stage():
    # ... (This UI is mostly the same)
    st.header(f"Mission Simulation (Optimized for: {st.session_state.optimization_preference})")
    c1,c2=st.columns([1,2.5])
    with c1:
        st.subheader("Mission Control")
        sb,stb=st.columns(2)
        if sb.button("▶️ Run",use_container_width=True,disabled=st.session_state.mission_running):st.session_state.mission_running=True;st.rerun()
        if stb.button("⏸️ Pause",use_container_width=True,disabled=not st.session_state.mission_running):st.session_state.mission_running=False;st.rerun()
        st.subheader("Stats")
        prog=(st.session_state.path_index)/(len(st.session_state.planned_path)-1)if len(st.session_state.planned_path)>1 else 0;st.progress(prog,text=f"Progress: {prog:.0%}")
        rem_bat=config.DRONE_BATTERY_WH-st.session_state.total_energy;st.metric("Battery",f"{rem_bat:.2f} Wh",delta=f"{-st.session_state.total_energy:.2f} Wh used")
        if st.session_state.path_index>=len(st.session_state.planned_path)-1 and st.session_state.planned_path:st.success("✅ Mission Complete!");st.session_state.mission_running=False
        if st.button("⬅️ New Mission",use_container_width=True):reset_mission_state();st.rerun()
    with c2:
        fig=go.Figure();hub,dest=st.session_state.hub_location,st.session_state.destination
        fig.add_traces([go.Scatter3d(x=[hub[0]],y=[hub[1]],z=[hub[2]],mode='markers',marker=dict(size=8,color='cyan',symbol='diamond'),name='Hub'),go.Scatter3d(x=[dest[0]],y=[dest[1]],z=[dest[2]],mode='markers+text',text=["Dest."],textposition='middle right',marker=dict(size=8,color='lime'),name='Destination')])
        for b in planner.env.buildings:fig.add_trace(create_box(b))
        for nfz in planner.env.static_nfzs:fig.add_trace(create_nfz_box(nfz))
        for dnfz in planner.env.dynamic_nfzs:fig.add_trace(create_nfz_box(dnfz,color='yellow',opacity=0.2))
        path_np=np.array(st.session_state.planned_path);fig.add_trace(go.Scatter3d(x=path_np[:,0],y=path_np[:,1],z=path_np[:,2],mode='lines',line=dict(width=4,color='yellow'),name='Planned Path'))
        dp=st.session_state.drone_pos;fig.add_trace(go.Scatter3d(x=[dp[0]],y=[dp[1]],z=[dp[2]],mode='markers',marker=dict(size=10,color='red',symbol='circle-open'),name='Live Drone'))
        fig.update_layout(margin=dict(l=0,r=0,b=0,t=0),scene=dict(aspectratio=dict(x=1,y=1,z=0.4),bgcolor='rgb(20,24,54)'),legend=dict(font=dict(color='white')));st.plotly_chart(fig,use_container_width=True,height=700)
# --- Sidebar ---
with st.sidebar:st.header("Ops Log");log_c=st.container(height=800);
with log_c:
    for e in st.session_state.log:st.text(e)

# --- Main Execution & Simulation Loop ---
if planner.predictor.models:
    if st.session_state.stage=='setup':setup_stage()
    elif st.session_state.stage=='planning':planning_stage()
    elif st.session_state.stage=='simulation':simulation_stage()

    if st.session_state.mission_running:
        path=st.session_state.planned_path;idx=st.session_state.path_index
        if idx<len(path)-1:
            p1,p2=path[idx],path[idx+1]
            planner.env.update_environment(st.session_state.total_time,0.5)

            # --- MODIFICATION: Call the new LPA* replanning logic ---
            if check_replanning_triggers(p1,p2,planner):
                with st.spinner("LPA* Replanning..."):
                    # The new NFZ is the last one added
                    changed_nfz=planner.env.dynamic_nfzs[-1]
                    new_path_segment,status=planner.replan_path(changed_nfz)
                
                if new_path_segment:
                    # Splice the new path into the existing plan
                    st.session_state.planned_path=path[:idx+1]+new_path_segment[1:]
                    log_event("✅ LPA* Replanning complete. Path updated.")
                else:
                    log_event(f"❌ LPA* Replanning failed: {status}. Stopping.");st.session_state.mission_running=False
                # Reset the trigger to prevent continuous replanning
                planner.env.was_nfz_just_added = False
            
            # Continue simulation with current/new path
            current_payload=st.session_state.initial_payload if st.session_state.total_time < 200 else 0 # Simplified payload drop
            p_prev=path[idx-1]if idx>0 else None;wind=planner.env.weather.get_wind_at_location(p1[0],p1[1])
            t,e=planner.predictor.predict(p1,p2,current_payload,wind,p_prev)
            st.session_state.total_time+=t;st.session_state.total_energy+=e;st.session_state.path_index+=1;st.session_state.drone_pos=p2
            time.sleep(0.1);st.rerun()
else:
    st.error("ML Model not found. Please run `data_generator.py` and `train_model.py` first.")