"""
sim_app.py — Acoustic Levitation Simulator v2.1
─────────────────────────────────────────────
Features:
  • Dual Object Simulation (2 Water Droplets)
  • Interactive Viewport (Orbit & Zoom)
  • Real-time Force Arrows (Acoustic, Gravity, Net)
  • Dear PyGui + PyVista Off-screen rendering
"""

import sys, os, threading, time
import numpy as np
import trimesh
import dearpygui.dearpygui as dpg
import pyvista as pv

# ── Acoustic physics ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AcousticFieldModeling'))
from SimAcousticField import compute_gorkov_force, compute_complex_pressure
from materials import get_material, get_contrast_factors, MATERIALS

# ── Config ────────────────────────────────────────────────────────────────────
VP_W, VP_H   = 860, 680          # PyVista viewport size
PANEL_W      = 320
WIN_W        = PANEL_W + VP_W + 20
WIN_H        = VP_H + 60
PHYSICS_HZ   = 200               # physics ticks / second
RENDER_HZ    = 30                # viewport refresh / second

# ── Load transducer array ─────────────────────────────────────────────────────
def _load_sources():
    p = os.path.join(os.path.dirname(__file__), 'AcousticFieldModeling', 'srcarray.txt')
    if os.path.exists(p):
        raw = np.loadtxt(p)
        return np.column_stack((raw[:, 0], raw[:, 2], raw[:, 1]))
    return np.array([[0., 0., 40.], [0., 0., -40.]])

SOURCES = _load_sources()

# ── Object State ─────────────────────────────────────────────────────────────
class ObjState:
    def __init__(self, idx, color='#1a7fff'):
        self.idx = idx
        self.pos = np.array([0.0, 0.0, 0.0])
        self.vel = np.array([0.0, 0.0, 0.0])
        self.stl_path = os.path.join('3D_Files', 'Sphere.stl')
        self.scale = 0.04
        self.mat_key = 'water_droplet'
        self.color = color
        self.base_mesh = None
        self.vol = self.mass = self.f1 = self.f2 = None
        self.f_acoustic = np.zeros(3)
        self.f_net = np.zeros(3)
        self.p_map = None
        self.eq_pos = np.array([0.0, 0.0, 0.0])

class SimState:
    def __init__(self):
        self.lock = threading.Lock()
        self.objs = [ObjState(1, '#1a7fff'), ObjState(2, '#ffcc00')]
        self.objs[1].pos = np.array([5.0, 0.0, 0.0]) # Start obj 2 slightly offset
        self.power = 1.0
        self.phase_deg = 0.0
        self.damping = 0.05
        self.gravity_on = True
        self.running = False
        self.dirty = True
        self.use_limit = True  # Default to True as requested
        
        # Camera State
        self.cam_dist = 120.0
        self.cam_pitch = 45.0
        self.cam_yaw = 45.0
        self.cam_center = np.array([0.0, 0.0, 0.0])
        self.last_drag_x = 0.0
        self.last_drag_y = 0.0

S = SimState()

# ── Mesh helpers ──────────────────────────────────────────────────────────────
_mesh_cache: dict = {}

def _build_base_mesh(stl_path, scale):
    key = (stl_path, scale)
    if key in _mesh_cache:
        return _mesh_cache[key].copy()
    try:
        m = trimesh.load(stl_path)
        if isinstance(m, trimesh.Scene):
            parts = [g for g in m.geometry.values() if isinstance(g, trimesh.Trimesh)]
            m = trimesh.util.concatenate(parts) if parts else list(m.geometry.values())[0]
        m.apply_scale(scale)
        m.vertices -= m.centroid
        _mesh_cache[key] = m.copy()
        return m
    except:
        return trimesh.creation.icosphere(radius=1.0)

def refresh_props(obj_idx):
    with S.lock:
        obj = S.objs[obj_idx-1]
        path, sc, mk = obj.stl_path, obj.scale, obj.mat_key
    m = _build_base_mesh(path, sc)
    mat = get_material(mk)
    f1, f2 = get_contrast_factors(mk)
    with S.lock:
        obj.base_mesh = m
        obj.vol = m.volume
        obj.mass = m.volume * mat['rho']
        obj.f1, obj.f2 = f1, f2
        S.dirty = True

# ── Physics thread ────────────────────────────────────────────────────────────
def _physics_loop():
    dt = 1.0 / PHYSICS_HZ
    while True:
        with S.lock:
            if not S.running:
                time.sleep(dt)
                continue
            
            power = S.power
            phase = S.phase_deg
            damp  = S.damping
            grav  = S.gravity_on
            phases = np.zeros(len(SOURCES))
            phases[SOURCES[:, 2] > 0] = np.radians(phase)

            for obj in S.objs:
                if obj.vol is None: continue
                
                pos = obj.pos.copy()
                vel = obj.vel.copy()
                vol, mass, f1, f2 = obj.vol, obj.mass, obj.f1, obj.f2

                fa = compute_gorkov_force(pos.reshape(1,3), SOURCES, vol, f1, f2, phases)[0] * (power**2)
                fg = np.array([0., 0., -(mass * 9806.65)]) if grav else np.zeros(3)
                fn = fa + fg

                # Constraint force (restoring spring)
                if S.use_limit:
                    dist_vec = pos - obj.eq_pos
                    dist = np.linalg.norm(dist_vec)
                    if dist > 5.0:
                        # Apply a strong restoring force back towards equilibrium
                        spring_k = 500.0 
                        fn -= dist_vec * spring_k

                # RK4
                def acc(v): return fn / mass - damp * v
                k1v = acc(vel);               k1p = vel
                k2v = acc(vel + .5*dt*k1v);   k2p = vel + .5*dt*k1v
                k3v = acc(vel + .5*dt*k2v);   k3p = vel + .5*dt*k2v
                k4v = acc(vel +    dt*k3v);   k4p = vel +    dt*k3v

                obj.vel = vel + (dt/6.)*(k1v + 2*k2v + 2*k3v + k4v)
                obj.pos = np.clip(pos + (dt/6.)*(k1p + 2*k2p + 2*k3p + k4p), -50, 50)
                obj.f_acoustic = fa
                obj.f_net = fn
            
            S.dirty = True
        time.sleep(dt)

# ── PyVista off-screen renderer ───────────────────────────────────────────────
class Renderer:
    def __init__(self):
        self.pl = pv.Plotter(off_screen=True, window_size=[VP_W, VP_H])
        self.pl.set_background('#080a14', top='#1a1c2c')
        src = pv.PolyData(SOURCES)
        self.pl.add_mesh(src, color='white', opacity=0.35, point_size=5, name='src')
        box = pv.Box(bounds=[-50,50,-50,50,-50,50])
        self.pl.add_mesh(box, style='wireframe', color='#1a1a3a', opacity=0.4, name='box')
        self.pl.add_axes(color='white')

    def update_camera(self):
        dist = S.cam_dist
        pitch = np.radians(S.cam_pitch)
        yaw = np.radians(S.cam_yaw)
        
        x = dist * np.cos(pitch) * np.cos(yaw)
        y = dist * np.cos(pitch) * np.sin(yaw)
        z = dist * np.sin(pitch)
        
        self.pl.camera.position = (x + S.cam_center[0], y + S.cam_center[1], z + S.cam_center[2])
        self.pl.camera.focal_point = S.cam_center
        self.pl.camera.up = (0, 0, 1)

    def frame(self):
        pl = self.pl
        with S.lock:
            objs = S.objs
            show_p = dpg.get_value('show_pressure')
            show_f = dpg.get_value('show_force')
            phase = S.phase_deg

        self.update_camera()
        
        # Clear dynamic actors
        for i in [1, 2]:
            pl.remove_actor(f'obj{i}_mesh')
            pl.remove_actor(f'obj{i}_wire')
            pl.remove_actor(f'obj{i}_f_acoustic')
            pl.remove_actor(f'obj{i}_f_net')

        for obj in objs:
            if obj.base_mesh is None: continue
            
            pos = obj.pos
            verts = obj.base_mesh.vertices + pos
            pv_faces = np.hstack([np.full((len(obj.base_mesh.faces),1), 3, dtype=np.int_), obj.base_mesh.faces])
            pvm = pv.PolyData(verts, pv_faces.ravel())

            if show_p:
                pts = obj.base_mesh.triangles_center + pos
                phases = np.zeros(len(SOURCES))
                phases[SOURCES[:,2] > 0] = np.radians(phase)
                p_map = np.abs(compute_complex_pressure(pts, SOURCES, phases))
                pvm.cell_data['Pa'] = p_map
                pl.add_mesh(pvm, scalars='Pa', cmap='viridis', opacity=0.8, name=f'obj{obj.idx}_mesh')
            else:
                pl.add_mesh(pvm, color=obj.color, opacity=0.7, name=f'obj{obj.idx}_mesh')
            
            pl.add_mesh(pvm, style='wireframe', color='white', opacity=0.2, name=f'obj{obj.idx}_wire')

            if show_f:
                # Acoustic Force (Magenta)
                famag = np.linalg.norm(obj.f_acoustic)
                if famag > 1e-5:
                    scale = float(np.clip(famag * 0.01, 2.0, 15.0))
                    arr_a = pv.Arrow(start=pos, direction=obj.f_acoustic/famag, scale=scale)
                    pl.add_mesh(arr_a, color='#ff00ff', name=f'obj{obj.idx}_f_acoustic')
                
                # Net Force (Orange)
                fnmag = np.linalg.norm(obj.f_net)
                if fnmag > 1e-5:
                    scale = float(np.clip(fnmag * 0.01, 2.0, 15.0))
                    arr_n = pv.Arrow(start=pos, direction=obj.f_net/fnmag, scale=scale)
                    pl.add_mesh(arr_n, color='#ff8800', name=f'obj{obj.idx}_f_net')

        pl.render()
        img = pl.screenshot(return_img=True)
        rgba = np.dstack([img, np.full(img.shape[:2], 255, dtype=np.uint8)])
        return rgba.flatten().astype(np.float32) / 255.0

# ── Interaction Handlers ──────────────────────────────────────────────────────
def cb_mouse_drag(sender, app_data):
    if not dpg.is_item_hovered('viewport_img'): return
    
    # app_data is [button, dx, dy] where dx/dy are cumulative for the drag
    curr_dx, curr_dy = app_data[1], app_data[2]
    delta_x = curr_dx - S.last_drag_x
    delta_y = curr_dy - S.last_drag_y
    S.last_drag_x, S.last_drag_y = curr_dx, curr_dy

    with S.lock:
        if dpg.is_mouse_button_down(dpg.mvMouseButton_Left):
            S.cam_yaw -= delta_x * 0.4
            S.cam_pitch = np.clip(S.cam_pitch + delta_y * 0.4, -89, 89)
        S.dirty = True

def cb_mouse_release(sender, app_data):
    S.last_drag_x = 0.0
    S.last_drag_y = 0.0

def cb_wheel(sender, app_data):
    if not dpg.is_item_hovered('viewport_img'): return
    with S.lock:
        S.cam_dist = np.clip(S.cam_dist - app_data * 5.0, 10, 300)
        S.dirty = True

# ── UI Callbacks ─────────────────────────────────────────────────────────────
def cb_play(): 
    with S.lock:
        for obj in S.objs:
            obj.eq_pos = obj.pos.copy() # Set current pos as equilibrium baseline
        S.running = True
    dpg.set_value('status', 'Running...')

def cb_stop(): S.running = False; dpg.set_value('status', 'Paused')
def cb_reset():
    with S.lock:
        S.running = False
        for i, obj in enumerate(S.objs):
            obj.pos = np.array([dpg.get_value(f'o{i+1}_x'), dpg.get_value(f'o{i+1}_y'), dpg.get_value(f'o{i+1}_z')])
            obj.vel = np.zeros(3)
        S.dirty = True
    dpg.set_value('status', 'Reset')

def cb_obj_param(sender, val, ud):
    obj_idx, param = ud
    with S.lock:
        obj = S.objs[obj_idx-1]
        if param == 'stl': obj.stl_path = val
        elif param == 'mat': obj.mat_key = val
        elif param == 'scale': obj.scale = val
    threading.Thread(target=refresh_props, args=(obj_idx,), daemon=True).start()

def cb_pos_sync(sender, val, ud):
    obj_idx, axis = ud
    with S.lock:
        if not S.running:
            S.objs[obj_idx-1].pos[axis] = val
            S.dirty = True

# ── Main UI ───────────────────────────────────────────────────────────────────
def build_ui():
    stl_files = [f for f in os.listdir('3D_Files') if f.endswith('.stl')]
    mat_keys = list(MATERIALS.keys())
    
    with dpg.window(tag='main_win', label='Simulator v2.1', width=WIN_W, height=WIN_H, no_move=True):
        with dpg.group(horizontal=True):
            # Left Panel
            with dpg.child_window(width=PANEL_W, border=True):
                dpg.add_text('Simulation Control', color=[0, 224, 255])
                with dpg.group(horizontal=True):
                    dpg.add_button(label='Play', callback=cb_play, width=80)
                    dpg.add_button(label='Stop', callback=cb_stop, width=80)
                    dpg.add_button(label='Reset', callback=cb_reset, width=80)
                dpg.add_text('Ready', tag='status')
                
                dpg.add_separator()
                for i in [1, 2]:
                    with dpg.collapsing_header(label=f'Object {i}', default_open=(i==1)):
                        dpg.add_combo(items=stl_files, default_value='Sphere.stl', label='STL', 
                                      callback=cb_obj_param, user_data=(i, 'stl'))
                        dpg.add_combo(items=mat_keys, default_value='water_droplet', label='Mat',
                                      callback=cb_obj_param, user_data=(i, 'mat'))
                        dpg.add_slider_float(label='Scale', default_value=0.04, min_value=0.01, max_value=0.2,
                                             callback=cb_obj_param, user_data=(i, 'scale'))
                        dpg.add_slider_float(label='X', tag=f'o{i}_x', min_value=-30, max_value=30, 
                                             default_value=(5.0 if i==2 else 0.0), callback=cb_pos_sync, user_data=(i, 0))
                        dpg.add_slider_float(label='Y', tag=f'o{i}_y', min_value=-30, max_value=30, callback=cb_pos_sync, user_data=(i, 1))
                        dpg.add_slider_float(label='Z', tag=f'o{i}_z', min_value=-30, max_value=30, callback=cb_pos_sync, user_data=(i, 2))

                with dpg.collapsing_header(label='Global Settings', default_open=True):
                    dpg.add_slider_float(label='Power', tag='power', default_value=100, min_value=0, max_value=100, 
                                         callback=lambda s,v: setattr(S, 'power', v/100.0))
                    dpg.add_slider_float(label='Phase', tag='phase', default_value=0, min_value=0, max_value=360,
                                         callback=lambda s,v: setattr(S, 'phase_deg', v))
                    dpg.add_checkbox(label='Gravity', tag='gravity', default_value=True,
                                     callback=lambda s,v: setattr(S, 'gravity_on', v))
                    dpg.add_checkbox(label='Constrain 5mm', tag='limit_toggle', default_value=True,
                                     callback=lambda s,v: setattr(S, 'use_limit', v))
                    dpg.add_checkbox(label='Show Pressure', tag='show_pressure', default_value=True)
                    dpg.add_checkbox(label='Show Forces', tag='show_force', default_value=True)

                dpg.add_separator()
                dpg.add_text('Stats', color=[0, 224, 255])
                dpg.add_text('...', tag='stats_txt')

            # Viewport
            with dpg.child_window(width=VP_W, height=VP_H, border=False, no_scrollbar=True):
                dpg.add_image('tex', width=VP_W, height=VP_H, tag='viewport_img')
                with dpg.item_handler_registry(tag='viewport_handler'):
                    dpg.add_item_clicked_handler(callback=lambda: dpg.focus_item('viewport_img'))
                dpg.bind_item_handler_registry('viewport_img', 'viewport_handler')

    # Mouse handlers
    with dpg.handler_registry():
        dpg.add_mouse_drag_handler(callback=cb_mouse_drag)
        dpg.add_mouse_wheel_handler(callback=cb_wheel)
        dpg.add_mouse_release_handler(callback=cb_mouse_release)

def main():
    global R
    dpg.create_context()
    
    with dpg.texture_registry():
        dpg.add_raw_texture(VP_W, VP_H, np.zeros(VP_W*VP_H*4, dtype=np.float32), tag='tex', format=dpg.mvFormat_Float_rgba)
    
    build_ui()
    dpg.create_viewport(title='Acoustic Sim v2.1', width=WIN_W, height=WIN_H, resizable=False)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    
    R = Renderer()
    for i in [1, 2]: refresh_props(i)
    
    threading.Thread(target=_physics_loop, daemon=True).start()
    
    while dpg.is_dearpygui_running():
        if S.dirty or S.running:
            data = R.frame()
            dpg.set_value('tex', data)
            S.dirty = False
            
            # Update stats string
            with S.lock:
                s = ""
                for obj in S.objs:
                    s += f"Obj {obj.idx}: Z={obj.pos[2]:.2f} F={np.linalg.norm(obj.f_acoustic):.1f}uN\n"
                    if S.running:
                        dpg.set_value(f'o{obj.idx}_x', obj.pos[0])
                        dpg.set_value(f'o{obj.idx}_y', obj.pos[1])
                        dpg.set_value(f'o{obj.idx}_z', obj.pos[2])
                dpg.set_value('stats_txt', s)
                
        dpg.render_dearpygui_frame()
    dpg.destroy_context()

if __name__ == '__main__':
    main()
