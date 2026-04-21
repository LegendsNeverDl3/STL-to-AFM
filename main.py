import numpy as np
import trimesh
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import argparse
import webbrowser
import dash
from dash import Dash, dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from flask_cors import CORS

# --- Import acoustic field module ---
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AcousticFieldModeling'))
from SimAcousticField import (
    compute_complex_pressure,
    compute_velocity_vector,
    compute_gorkov_potential,
    compute_gorkov_force,
)
from materials import (
    get_medium, get_material, get_contrast_factors,
    MATERIALS, AMPLITUDE,
    ACTIVE_MATERIAL as DEFAULT_MATERIAL,
)
import materials as mat_module


# --- Legacy simplified model (kept for "Simplified" mode) ---
# Iplements a basic radiation pressure model model that calculates pressure
# at the face of the object and converts it to a force vector pointing inward
# based on the normal of each face
def compute_simplified_forces(centroids, normals, face_areas, sources, phases=None):
    """Simplified radiation pressure model: F = p^2/(rho*c^2) * (-n) * A"""
    # Physical properties of medium(ususally air)
    med = get_medium()
    #grab density
    rho = med["rho"]
    #grab speed of sound
    c = med["c"]
    # sums up waves from all 100+ transducers to find the complex pressure at every
    # face(both volume and timing(phase)) of the sound wave
    p_complex = compute_complex_pressure(centroids, sources, phases)
    p_amplitude = np.abs(p_complex)
    # acoustic velocity
    v_scalar = p_amplitude / (rho * c)

    # Radiation pressure: <p^2> / (rho * c^2) = |p|^2 / (2 * rho * c^2)
    p_rad = (p_amplitude ** 2) / (2.0 * rho * c ** 2)
    # force = pressure * area
    f_acoustic = p_rad[:, np.newaxis] * (-normals) * face_areas[:, np.newaxis]

    return p_amplitude, v_scalar, f_acoustic


# --- Gorkov model ---
# Calculates the acoustic radiation force on an object using the Gorkov potential
# Note that this model calculates the force on the volume of the object, not the surface
# i.e. a sphere. Additionally, it accounts for interaction between the sound pressure
# and air velocity simultaneously
def compute_gorkov_forces(centroids, sources, mesh_volume, phases=None):
    """Full Gorkov potential model: F = -grad(U)"""
    # This is a dimensionless factor that accounts for the difference in acoustic
    # properties between the object and the medium
    f1, f2 = get_contrast_factors()
    # sums up waves from all 100+ transducers to find the complex pressure at every
    # face(both volume and timing(phase)) of the sound wave
    p_complex = compute_complex_pressure(centroids, sources, phases)
    p_amplitude = np.abs(p_complex)
    # acoustic velocity -> oscilations at a certain spot
    v_vectors, v_speed = compute_velocity_vector(centroids, sources, phases)
    # gorkov potential -> potential energy at a certain spot
    gorkov_U = compute_gorkov_potential(p_complex, v_speed, mesh_volume, f1, f2)
    # gradient call occurs in here
    gorkov_force = compute_gorkov_force(centroids, sources, mesh_volume, f1, f2, phases)

    return p_amplitude, v_speed, v_vectors, gorkov_U, gorkov_force


# --- DASH APPLICATION CONFIG ---
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
CORS(app.server)
app.title = "Acoustic Levitation Simulator"

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body { background: radial-gradient(circle at center, #1a1c2c 0%, #0a0b14 100%) !important; color: #e0e0e0; font-family: 'Inter', sans-serif; }
            .control-card { background-color: rgba(30, 32, 48, 0.85) !important; border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 15px !important; backdrop-filter: blur(10px); box-shadow: 0 8px 32px rgba(0,0,0,0.5); }
            .card-header { background: linear-gradient(90deg, #2a2d3e, #1e2030) !important; color: #00e0ff !important; font-weight: 800 !important; letter-spacing: 1px; border-bottom: 1px solid rgba(255,255,255,0.1) !important; }
            .section-header { color: #6c757d; font-size: 0.75rem; font-weight: 800; text-transform: uppercase; letter-spacing: 1.5px; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 4px; margin: 15px 0 10px 0; }
            .control-label { color: #b0b3b8 !important; font-weight: 600 !important; font-size: 0.85rem; margin-bottom: 4px; }
            .dark-input { background-color: #0d0e1a !important; color: #ffffff !important; border: 1px solid #30363d !important; font-weight: bold !important; height: 32px !important; }
            .dark-input:focus { border-color: #00e0ff !important; box-shadow: 0 0 10px rgba(0, 224, 255, 0.2) !important; }
            .stats-text { background-color: rgba(0,0,0,0.3) !important; color: #ffffff !important; border-left: 3px solid #00e0ff; padding: 12px; border-radius: 8px; font-family: 'Consolas', monospace; font-size: 0.8rem; line-height: 1.5; }
            .accordion-button { background-color: transparent !important; color: #ffffff !important; font-size: 0.9rem !important; font-weight: 700 !important; border: none !important; }
            .accordion-button:not(.collapsed) { background-color: rgba(0,224,255,0.1) !important; color: #00e0ff !important; box-shadow: none !important; }
            .accordion-item { background-color: transparent !important; border: none !important; border-bottom: 1px solid rgba(255,255,255,0.05) !important; }
            .accordion-body { padding: 15px 10px !important; }
            .rc-slider-rail { background-color: #30363d !important; }
            .rc-slider-track { background-color: #00e0ff !important; }
            .rc-slider-handle { border-color: #00e0ff !important; background-color: #00e0ff !important; }
            /* Broad Dash Dropdown Fix */
            .dash-dropdown, .dash-dropdown-search, .dash-dropdown-option {
                background-color: #0d0e1a !important;
                border: 1px solid #30363d !important;
                color: #ffffff !important;
            }
            /* Target all internal text elements regardless of tag */
            .dash-dropdown *, .dash-dropdown-search * {
                color: #ffffff !important;
                background-color: transparent !important;
            }
            .dash-dropdown-option:hover, .dash-options-list-option:hover, .dash-options-list-option.selected {
                background-color: #00e0ff !important;
                color: #000000 !important;
            }
            .dash-dropdown-option:hover *, .dash-options-list-option:hover * {
                color: #000000 !important;
            }
            /* Legacy fallback */
            .Select-control, .Select-menu-outer, .Select-value-label { 
                background-color: #0d0e1a !important; 
                color: #ffffff !important; 
            }
            /* End Dropdown Styling */
            .radio-group label { color: #e0e0e0 !important; cursor: pointer; }
        </style>
    </head>
    <body class="control-panel">
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Build material dropdown options
material_options = [{'label': v['name'], 'value': k} for k, v in MATERIALS.items()]

# Scan for available STL files
stl_files = [f for f in os.listdir('3D_Files') if f.endswith('.stl')]
stl_options = [{'label': f.replace('.stl', ''), 'value': os.path.join('3D_Files', f)} for f in stl_files]
default_stl = os.path.join('3D_Files', 'Tetrahedron.stl') if 'Tetrahedron.stl' in stl_files else stl_options[0]['value']

app.layout = dbc.Container([
    dcc.Store(id='camera-store', data={'eye': {'x': 1.25, 'y': 1.25, 'z': 1.25}, 'center': {'x': 0, 'y': 0, 'z': 0}}),
    
    dbc.Row([
        dbc.Col(html.H1("Acoustic Levitation Simulator", className="text-center my-4", style={'color': '#00e0ff', 'fontWeight': '200', 'letterSpacing': '4px'}), width=8),
        dbc.Col([
            html.Label("Quick Presets", className="control-label"),
            dcc.Dropdown(
                id='quick-presets',
                options=[
                    {'label': 'Default (Foam)', 'value': 'foam'},
                    {'label': 'Water Droplet', 'value': 'droplet'},
                ],
                placeholder="Apply Preset...",
                className="mb-3"
            ),
        ], width=4, className="d-flex flex-column justify-content-center")
    ]),
    
    dbc.Row([
        # --- LEFT SIDEBAR (Width 3) ---
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("CONTROLS"),
                dbc.CardBody([
                    dbc.Accordion([
                        # Category 1: Global Settings
                        dbc.AccordionItem([
                            html.Label("Interaction Mode", className="control-label"),
                            dbc.RadioItems(
                                id='rotation-mode',
                                options=[
                                    {'label': 'Object', 'value': 'object'},
                                    {'label': 'World', 'value': 'world'}
                                ],
                                value='world',
                                className="mb-3 radio-group",
                                inline=True, style={'fontSize': '0.8rem'}
                            ),
                            
                            html.Label("Force Model", className="control-label"),
                            dcc.Dropdown(
                                id='force-model',
                                options=[
                                    {'label': 'Simplified (Radiation)', 'value': 'simplified'},
                                    {'label': 'Gorkov Potential', 'value': 'gorkov'},
                                ],
                                value='simplified', clearable=False, className="mb-3"
                            ),
                            
                            html.Label("Gorkov Object Mode", id='gorkov-mode-label', className="control-label", style={'display': 'none'}),
                            dbc.RadioItems(
                                id='gorkov-object-mode',
                                options=[
                                    {'label': 'Single Point', 'value': 'single'},
                                    {'label': 'Integrate Mesh', 'value': 'mesh'}
                                ],
                                value='mesh',
                                className="mb-3 radio-group",
                                inline=True, style={'fontSize': '0.8rem', 'display': 'none'}
                            ),
                            
                            html.Label(["Sound Power: ", html.Span(id='sp-val', children='100'), "%"], className="control-label"),
                            dcc.Slider(id='sound-power', min=1, max=100, step=1, value=100, tooltip={"always_visible": True, "placement": "bottom"}),
                            
                            html.Label(["Phase Shift: ", html.Span(id='ps-val', children='0'), "°"], className="control-label"),
                            dcc.Slider(id='phase-shift', min=0, max=360, step=1, value=0, tooltip={"always_visible": True, "placement": "bottom"}),

                            html.Hr(style={'borderColor': 'rgba(255,255,255,0.05)'}),
                            dbc.Checklist(
                                id='enable-obj2',
                                options=[{'label': ' ENABLE SECOND OBJECT', 'value': 'enabled'}],
                                value=[],
                                switch=True,
                                className="radio-group mb-3", style={'fontWeight': 'bold', 'color': '#00e0ff'}
                            ),
                        ], title="Global Settings"),
                        
                        # Category 2: Object 1 Details
                        dbc.AccordionItem([
                            html.Label("3D Object", className="control-label"),
                            dcc.Dropdown(id='selected-object', options=stl_options, value=default_stl, clearable=False, className="mb-2"),
                            
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Scale", className="control-label"),
                                    dbc.Input(id='object-scale', type='number', value=0.04, step=0.001, className="dark-input mb-2"),
                                ], width=6),
                                dbc.Col([
                                    html.Label("Z-Squash", className="control-label"),
                                    dbc.Input(id='z-squash', type='number', value=1.0, step=0.05, className="dark-input mb-2"),
                                ], width=6),
                            ]),

                            html.Label("Material", className="control-label"),
                            dcc.Dropdown(id='material-preset', options=[{'label': m['name'], 'value': k} for k, m in MATERIALS.items()], value='polystyrene_foam', clearable=False, className="mb-2"),

                            html.Label(["Subdivision: ", html.Span(id='subdiv-val', children='0')], className="control-label"),
                            dcc.Slider(id='subdiv-level', min=0, max=3, step=1, value=0, marks={0:'0', 1:'1', 2:'2', 3:'3'}),
                            
                            html.Label("Translational Position (X, Y, Z)", className="control-label mt-2"),
                            dbc.Row([
                                dbc.Col(dbc.Input(id='pos-x', type='number', value=0, size="sm", className="dark-input"), width=4),
                                dbc.Col(dbc.Input(id='pos-y', type='number', value=0, size="sm", className="dark-input"), width=4),
                                dbc.Col(dbc.Input(id='pos-z', type='number', value=0, size="sm", className="dark-input"), width=4),
                            ], className="mb-2"),
                            dcc.Slider(id='slider-pos-x', min=-25, max=25, step=0.1, value=0, className="mb-1"),
                            dcc.Slider(id='slider-pos-y', min=-25, max=25, step=0.1, value=0, className="mb-1"),
                            dcc.Slider(id='slider-pos-z', min=-40, max=40, step=0.1, value=0, className="mb-1"),

                            html.Label("Rotation (Pitch, Yaw, Roll)", className="control-label mt-2"),
                            dbc.Row([
                                dbc.Col(dbc.Input(id='rot-x', type='number', value=0, size="sm", className="dark-input"), width=4),
                                dbc.Col(dbc.Input(id='rot-y', type='number', value=0, size="sm", className="dark-input"), width=4),
                                dbc.Col(dbc.Input(id='rot-z', type='number', value=0, size="sm", className="dark-input"), width=4),
                            ], className="mb-2"),
                        ], title="Object 1 (Primary)"),

                        # Category 3: Object 2 Details
                        dbc.AccordionItem([
                            html.Label("3D Object", className="control-label"),
                            dcc.Dropdown(id='selected-object-2', options=stl_options, value=os.path.join('3D_Files', 'Sphere.stl') if 'Sphere.stl' in stl_files else stl_options[0]['value'], clearable=False, className="mb-2"),
                            
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Scale", className="control-label"),
                                    dbc.Input(id='object-scale-2', type='number', value=0.04, step=0.001, className="dark-input mb-2"),
                                ], width=6),
                                dbc.Col([
                                    html.Label("Z-Squash", className="control-label"),
                                    dbc.Input(id='z-squash-2', type='number', value=1.0, step=0.05, className="dark-input mb-2"),
                                ], width=6),
                            ]),

                            html.Label("Material", className="control-label"),
                            dcc.Dropdown(id='material-preset-2', options=[{'label': m['name'], 'value': k} for k, m in MATERIALS.items()], value='water_droplet', clearable=False, className="mb-2"),

                            html.Label(["Subdivision: ", html.Span(id='subdiv-val-2', children='0')], className="control-label"),
                            dcc.Slider(id='subdiv-level-2', min=0, max=3, step=1, value=0, marks={0:'0', 1:'1', 2:'2', 3:'3'}),
                            
                            html.Label("Translational Position (X, Y, Z)", className="control-label mt-2"),
                            dbc.Row([
                                dbc.Col(dbc.Input(id='pos-x-2', type='number', value=5, size="sm", className="dark-input"), width=4),
                                dbc.Col(dbc.Input(id='pos-y-2', type='number', value=0, size="sm", className="dark-input"), width=4),
                                dbc.Col(dbc.Input(id='pos-z-2', type='number', value=0, size="sm", className="dark-input"), width=4),
                            ], className="mb-2"),
                            dcc.Slider(id='slider-pos-x-2', min=-25, max=25, step=0.1, value=5, className="mb-1"),
                            dcc.Slider(id='slider-pos-y-2', min=-25, max=25, step=0.1, value=0, className="mb-1"),
                            dcc.Slider(id='slider-pos-z-2', min=-40, max=40, step=0.1, value=0, className="mb-1"),

                            html.Label("Rotation (Pitch, Yaw, Roll)", className="control-label mt-2"),
                            dbc.Row([
                                dbc.Col(dbc.Input(id='rot-x-2', type='number', value=0, size="sm", className="dark-input"), width=4),
                                dbc.Col(dbc.Input(id='rot-y-2', type='number', value=0, size="sm", className="dark-input"), width=4),
                                dbc.Col(dbc.Input(id='rot-z-2', type='number', value=0, size="sm", className="dark-input"), width=4),
                            ], className="mb-2"),
                        ], title="Object 2 (Secondary)", id='obj2-accordion-item', style={'display': 'none'}),

                        # Category 4: Physics Visualization
                        dbc.AccordionItem([
                            dbc.Checklist(
                                id='field-toggles',
                                options=[
                                    {'label': ' Pressure Mapping', 'value': 'pressure_color'},
                                    {'label': ' Local Acoustic Force Map', 'value': 'acoustic_force_color'},
                                    {'label': ' Gorkov Potential Map', 'value': 'gorkov_color'},
                                    {'label': ' Velocity Field', 'value': 'velocity_arrows'},
                                    {'label': ' Acoustic Force', 'value': 'acoustic_arrows'},
                                    {'label': ' Gravitational F', 'value': 'gravity_arrows'},
                                    {'label': ' GLOBAL NET FORCE', 'value': 'net_force_arrows'},
                                ],
                                value=['pressure_color', 'acoustic_arrows'],
                                className="radio-group", style={'fontSize': '0.85rem'}
                            ),
                        ], title="Physics Visualization"),
                    ], start_collapsed=True, flush=True),
                    
                    html.Hr(style={'borderColor': 'rgba(255,255,255,0.1)'}),
                    dbc.ButtonGroup([
                        dbc.Button(" Find Equilibrium", id='btn-auto-levitate', color="info", style={'fontWeight': 'bold'}),
                        dbc.Button(" Focus Camera", id='btn-focus-camera', color="secondary", style={'fontWeight': 'bold'}),
                    ], className="w-100 mb-3"),
                    html.Div(id='stats-target', className="stats-text")
                ])
            ], className="control-card")
        ], width=3),
        
        # --- MAIN SIMULATOR VIEW (Width 9) ---
        dbc.Col([
            dbc.Card([
                dcc.Graph(id='live-graph', style={'height': '85vh'}, config={'displayModeBar': False, 'scrollZoom': True})
            ], style={'borderRadius': '15px', 'overflow': 'hidden', 'border': 'none', 'boxShadow': '0 4px 20px rgba(0,0,0,0.3)'})
        ], width=9)
    ])
], fluid=True)

@app.callback(
    [Output('selected-object', 'value'), Output('material-preset', 'value'),
     Output('force-model', 'value'), Output('object-scale', 'value'), Output('z-squash', 'value')],
    [Input('quick-presets', 'value')],
    prevent_initial_call=True
)
def apply_preset(preset):
    if preset == 'droplet':
        return os.path.join('3D_Files', 'Sphere.stl'), 'water_droplet', 'gorkov', 0.04, 0.7
    elif preset == 'foam':
        return os.path.join('3D_Files', 'Tetrahedron.stl'), 'polystyrene_foam', 'simplified', 0.04, 1.0
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

@app.callback(
    [Output('gorkov-object-mode', 'style'), Output('gorkov-mode-label', 'style'),
     Output('obj2-accordion-item', 'style')],
    [Input('force-model', 'value'), Input('enable-obj2', 'value')]
)
def toggle_gorkov_settings(model, obj2_enabled):
    gorkov_v = {'display': 'block'} if model == 'gorkov' else {'display': 'none'}
    obj2_v = {'display': 'block'} if 'enabled' in (obj2_enabled or []) else {'display': 'none'}
    return gorkov_v, gorkov_v, obj2_v

@app.callback(
    Output('pos-z', 'value', allow_duplicate=True),
    [Input('btn-auto-levitate', 'n_clicks')],
    [State('selected-object', 'value'), State('object-scale', 'value'), State('z-squash', 'value'),
     State('material-preset', 'value'), State('sound-power', 'value'), State('phase-shift', 'value'),
     State('pos-x', 'value'), State('pos-y', 'value'), State('force-model', 'value'), State('gorkov-object-mode', 'value')],
    prevent_initial_call=True
)
def find_equilibrium(n_clicks, stl_path, scale, squash, material_key, power, phase_deg, x, y, force_model, gorkov_mode):
    if not n_clicks:
        return dash.no_update
    
    # 1. Prepare physical parameters
    mat_info = get_material(material_key)
    power_mult = (power or 100) / 100.0
    phases = np.zeros(len(SOURCES))
    phases[SOURCES[:, 2] > 0] = np.radians(phase_deg)
    
    # 2. Load mesh and calculate gravity
    mesh = trimesh.load(stl_path)
    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(meshes) if meshes else mesh.to_geometry()
    mesh.apply_scale(scale or 0.04)
    if squash and squash != 1.0:
        # Volume-preserving squash: if Z scales by s, X and Y must scale by 1/sqrt(s)
        scaler_xy = 1.0 / np.sqrt(squash)
        v = mesh.vertices.copy()
        v[:, 0] *= scaler_xy
        v[:, 1] *= scaler_xy
        v[:, 2] *= squash
        mesh.vertices = v

    total_mass = mesh.volume * mat_info['rho']
    gravity_f_z = -(total_mass * 9806.65) # Total weight in micro-Newtons (g*mm/s^2)
    
    # 3. Optimized Sweep (Coarse Sweep + Refined Search)
    centroids = mesh.triangles_center - mesh.centroid
    step = max(1, len(centroids) // 200)
    sub_centroids = centroids[::step]
    sub_vol = mesh.volume / max(1, len(sub_centroids))
    f1, f2 = get_contrast_factors(material_key)
    
    z_coarse = np.linspace(-30, 30, 61)
    
    def get_net_force(z_pos):
        if force_model == 'gorkov' and gorkov_mode == 'single':
            test_pts = np.array([[x, y, z_pos]], dtype=float)
            f_acoustic = compute_gorkov_force(test_pts, SOURCES, mesh.volume, f1, f2, phases)
        else:
            test_pts = sub_centroids + np.array([x, y, z_pos])
            f_acoustic = compute_gorkov_force(test_pts, SOURCES, sub_vol, f1, f2, phases)
        
        total_acoustic_z = np.sum(f_acoustic[:, 2]) * (power_mult ** 2)
        return total_acoustic_z + gravity_f_z

    # Coarse sweep
    coarse_forces = [get_net_force(z) for z in z_coarse]
    
    # 4. Find stable crossings
    stable_z = []
    for i in range(len(coarse_forces)-1):
        if coarse_forces[i] > 0 and coarse_forces[i+1] <= 0:
            # We found a potential trap region! 
            # Use Bisection to refine this specific 1mm region to high precision
            low, high = z_coarse[i], z_coarse[i+1]
            for _ in range(6): # 6 iterations = 2^6 = 64x higher precision
                mid = (low + high) / 2
                if get_net_force(mid) > 0:
                    low = mid
                else:
                    high = mid
            stable_z.append((low + high) / 2)
    
    if not stable_z:
        # If no trap found, maybe increase range or return fallback
        return 0
        
    # Pick the one closest to current Z
    return float(stable_z[np.argmin(np.abs(stable_z))])

@app.callback(
    [Output('pos-x', 'value'), Output('pos-y', 'value'), Output('pos-z', 'value'),
     Output('rot-x', 'value'), Output('rot-y', 'value'), Output('rot-z', 'value')],
    [Input('slider-pos-x', 'value'), Input('slider-pos-y', 'value'), Input('slider-pos-z', 'value')],
    [State('pos-x', 'value'), State('pos-y', 'value'), State('pos-z', 'value')]
)
def sync_pos_sliders(sx, sy, sz, ix, iy, iz):
    ctx = callback_context
    if not ctx.triggered: return ix, iy, iz, 0, 0, 0
    return sx, sy, sz, dash.no_update, dash.no_update, dash.no_update

@app.callback(
    [Output('pos-x-2', 'value'), Output('pos-y-2', 'value'), Output('pos-z-2', 'value')],
    [Input('slider-pos-x-2', 'value'), Input('slider-pos-y-2', 'value'), Input('slider-pos-z-2', 'value')]
)
def sync_pos_sliders_2(sx, sy, sz):
    return sx, sy, sz

@app.callback(
    Output('camera-store', 'data', allow_duplicate=True),
    [Input('live-graph', 'relayoutData')],
    [State('rotation-mode', 'value'), State('camera-store', 'data')],
    prevent_initial_call=True
)
def sync_camera(relayout, mode, camera_data):
    if relayout and 'scene.camera' in relayout:
        return relayout['scene.camera']
    return dash.no_update


@app.callback(
    Output('camera-store', 'data', allow_duplicate=True),
    [Input('btn-focus-camera', 'n_clicks')],
    [State('pos-x', 'value'), State('pos-y', 'value'), State('pos-z', 'value'), State('camera-store', 'data')],
    prevent_initial_call=True
)
def focus_camera(n, x, y, z, current_cam):
    new_cam = current_cam.copy()
    # Normalize to scene limits (50) and offset Z downward by 1mm
    new_cam['center'] = {'x': x/50.0, 'y': y/50.0, 'z': (z - 1.0)/50.0}
    return new_cam

@app.callback(
    [Output('sp-val', 'children'), Output('ps-val', 'children'), Output('subdiv-val', 'children'), 
     Output('subdiv-val-2', 'children')],
    [Input('sound-power', 'value'), Input('phase-shift', 'value'), Input('subdiv-level', 'value'), 
     Input('subdiv-level-2', 'value')]
)
def update_slider_labels(sp, ps, sd, sd2):
    return f"{sp}", f"{ps}", f"{sd}", f"{sd2}"

@app.callback(
    [Output('live-graph', 'figure'), Output('stats-target', 'children')],
    [Input('pos-x', 'value'), Input('pos-y', 'value'), Input('pos-z', 'value'), 
     Input('rot-x', 'value'), Input('rot-y', 'value'), Input('rot-z', 'value'),
     Input('force-model', 'value'), Input('material-preset', 'value'),
     Input('field-toggles', 'value'), Input('phase-shift', 'value'), Input('sound-power', 'value'),
     Input('selected-object', 'value'), Input('object-scale', 'value'), Input('subdiv-level', 'value'),
     Input('z-squash', 'value'), Input('gorkov-object-mode', 'value'),
     Input('enable-obj2', 'value'),
     Input('selected-object-2', 'value'), Input('object-scale-2', 'value'), Input('subdiv-level-2', 'value'),
     Input('z-squash-2', 'value'), Input('material-preset-2', 'value'),
     Input('pos-x-2', 'value'), Input('pos-y-2', 'value'), Input('pos-z-2', 'value'),
     Input('rot-x-2', 'value'), Input('rot-y-2', 'value'), Input('rot-z-2', 'value'),
     Input('camera-store', 'data')],
    [State('rotation-mode', 'value')]
)
def update_physics(x1, y1, z1, rx1, ry1, rz1, force_model, material_key1, field_toggles, phase_shift_deg, sound_power_pct,
                    stl_path1, scale_factor1, subdiv_level1, z_squash1, gorkov_mode,
                    enable_obj2, stl_path2, scale_factor2, subdiv_level2, z_squash2, material_key2,
                    x2, y2, z2, rx2, ry2, rz2, camera_data, mode):
    
    # Scale base amplitude by power percentage
    power_mult = (sound_power_pct or 100) / 100.0
    phases = np.zeros(len(SOURCES))
    phases[SOURCES[:, 2] > 0] = np.radians(phase_shift_deg)
    
    fig = go.Figure()
    all_stats = []

    def process_object(idx, stl_p, scale, subdiv, squash, mat_k, x, y, z, rx, ry, rz):
        mesh = trimesh.load(stl_p)
        if isinstance(mesh, trimesh.Scene):
            meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
            mesh = trimesh.util.concatenate(meshes) if meshes else mesh.to_geometry()
        mesh.apply_scale(scale or 0.04)
        mesh.vertices -= mesh.centroid
        
        if subdiv and subdiv > 0:
            for _ in range(subdiv):
                v, f = trimesh.remesh.subdivide(mesh.vertices, mesh.faces)
                mesh = trimesh.Trimesh(vertices=v, faces=f)

        sq = squash if squash is not None else 1.0
        if sq != 1.0:
            scaler_xy = 1.0 / np.sqrt(sq)
            squash_mat = np.eye(4)
            squash_mat[0, 0] = scaler_xy
            squash_mat[1, 1] = scaler_xy
            squash_mat[2, 2] = sq
            mesh.apply_transform(squash_mat)
        
        r_x = trimesh.transformations.rotation_matrix(np.radians(rx), [1, 0, 0])
        r_y = trimesh.transformations.rotation_matrix(np.radians(ry), [0, 1, 0])
        r_z = trimesh.transformations.rotation_matrix(np.radians(rz), [0, 0, 1])
        t_mat = trimesh.transformations.translation_matrix([x, y, z])
        mesh.apply_transform(trimesh.transformations.concatenate_matrices(t_mat, r_z, r_y, r_x))
        
        c_pts, n_pts, a_pts = mesh.triangles_center, mesh.face_normals, mesh.area_faces
        mat_info = get_material(mat_k)
        f1, f2 = get_contrast_factors(mat_k)

        if force_model == 'gorkov':
            if gorkov_mode == 'single':
                p_amp_s, v_speed_s, v_vec_s, gorkov_U_s, gorkov_f_s = \
                    compute_gorkov_forces(np.array([[x, y, z]], dtype=float), SOURCES, mesh.volume, phases=phases)
                f_acoustic = np.tile(gorkov_f_s, (len(a_pts), 1)) / len(a_pts)
                p_amp = np.full(len(a_pts), p_amp_s[0])
                v_speed = np.full(len(a_pts), v_speed_s[0])
                v_vectors = np.tile(v_vec_s, (len(a_pts), 1))
                gorkov_U = np.full(len(a_pts), gorkov_U_s[0])
            else:
                vol_per_face = mesh.volume / max(1, len(a_pts))
                p_amp, v_speed, v_vectors, gorkov_U, gorkov_force = \
                    compute_gorkov_forces(c_pts, SOURCES, vol_per_face, phases=phases)
                f_acoustic = gorkov_force
            f_acoustic *= (power_mult ** 2)
            p_amp *= power_mult
        else:
            p_amp, v_scalar, f_acoustic = compute_simplified_forces(c_pts, n_pts, a_pts, SOURCES, phases=phases)
            f_acoustic *= (power_mult ** 2)
            p_amp *= power_mult
            v_speed = v_scalar * power_mult
            v_vectors = None
            gorkov_U = None

        total_mass = mesh.volume * mat_info['rho']
        f_gravity = np.zeros_like(f_acoustic)
        f_gravity[:, 2] = -(total_mass * 9806.65) / len(a_pts)
        f_net = f_acoustic + f_gravity
        net_v = np.sum(f_net, axis=0)

        # Rendering
        toggles = field_toggles or []
        if 'gorkov_color' in toggles and gorkov_U is not None:
            intensity, scale_name, colorscale = gorkov_U, 'Gorkov U', 'Portland'
        elif 'acoustic_force_color' in toggles:
            intensity, scale_name, colorscale = np.linalg.norm(f_acoustic, axis=1), 'Force Mag', 'Hot'
        elif 'pressure_color' in toggles:
            intensity, scale_name, colorscale = p_amp, 'Pressure (Pa)', 'Viridis'
        else:
            intensity, scale_name, colorscale = None, None, None

        mesh_color = '#0077ff' if mat_k == 'water_droplet' else ('#00e0ff' if idx==1 else '#ffcc00')
        mesh_opacity = 0.6 if mat_k == 'water_droplet' else 0.3
        
        fig.add_trace(go.Mesh3d(
            x=mesh.vertices[:, 0], y=mesh.vertices[:, 1], z=mesh.vertices[:, 2],
            i=mesh.faces[:, 0], j=mesh.faces[:, 1], k=mesh.faces[:, 2],
            intensity=intensity, intensitymode='cell' if intensity is not None else None,
            colorscale=colorscale, colorbar=dict(title=scale_name, x=1.00 + (idx-1)*0.08, len=0.4, y=0.7) if scale_name else None,
            color=mesh_color if intensity is None else None,
            opacity=mesh_opacity, flatshading=True,
            lighting=dict(ambient=0.45, diffuse=0.8, specular=1.0, roughness=0.1),
            name=f'Object {idx}'
        ))

        # Add white outline (wireframe) around each polygon to make it clearly visible
        if hasattr(mesh, 'edges_unique'):
            edge_vertices = mesh.vertices[mesh.edges_unique]
            lines_x = np.insert(edge_vertices[:, :, 0], 2, np.nan, axis=1).flatten()
            lines_y = np.insert(edge_vertices[:, :, 1], 2, np.nan, axis=1).flatten()
            lines_z = np.insert(edge_vertices[:, :, 2], 2, np.nan, axis=1).flatten()
            fig.add_trace(go.Scatter3d(
                x=lines_x, y=lines_y, z=lines_z,
                mode='lines', line=dict(color='white' if idx==1 else '#ffffaa', width=2),
                name=f'Obj{idx} Wireframe', showlegend=False,
                hoverinfo='none'
            ))

        if 'acoustic_arrows' in toggles:
            draw_arrows(fig, c_pts, f_acoustic, "#ff00ff" if idx==1 else "#ff55ff", f"Obj{idx} Acoustic", scale=2.0)
        if 'gravity_arrows' in toggles:
            draw_arrows(fig, np.array([[x, y, z]]), np.array([[0,0,-total_mass*9806.65]]), "#ff0000", f"Obj{idx} Gravity", scale=5.0, ref_mag=total_mass*9806.65)
        if 'net_force_arrows' in toggles:
            draw_arrows(fig, np.array([[x, y, z]]), np.array([net_v]), "#ff8800", f"Obj{idx} Net", scale=5.0, ref_mag=total_mass*9806.65)
        
        # Stats
        obj_stats = [
            html.Div([html.B(f"OBJECT {idx} ({mat_info['name']})", style={'color': mesh_color})]),
            html.Div([f"Vol: {mesh.volume:.2f} mm³ | Mass: {total_mass*1000:.2f} mg"]),
            html.Div([f"Net Force: {np.linalg.norm(net_v):.2f} μN | Z: {net_v[2]:+.2f} μN"]),
            html.Hr(style={'margin': '5px 0', 'opacity': '0.2'})
        ]
        return obj_stats

    def draw_arrows(f, origins, vecs, color, name, scale=6.0, ref_mag=None):
        mags = np.linalg.norm(vecs, axis=1)
        avg = ref_mag if ref_mag else (np.mean(mags[mags > 0]) if np.any(mags > 0) else 1.0)
        step = max(1, len(origins) // 150)
        idx = np.arange(0, len(origins), step)
        vis_mags = scale * (np.log1p(mags[idx] / avg) if ref_mag else mags[idx]*(scale/avg))
        
        u, v, w = vecs[idx, 0], vecs[idx, 1], vecs[idx, 2]
        non_zero = mags[idx] > 1e-12
        u[non_zero] *= vis_mags[non_zero] / mags[idx][non_zero]
        v[non_zero] *= vis_mags[non_zero] / mags[idx][non_zero]
        w[non_zero] *= vis_mags[non_zero] / mags[idx][non_zero]

        f.add_trace(go.Cone(
            x=origins[idx, 0], y=origins[idx, 1], z=origins[idx, 2],
            u=u, v=v, w=w, sizemode="absolute", sizeref=scale*1.5, anchor="tip",
            colorscale=[[0, color], [1, color]], showscale=False, name=name
        ))

    # Process Object 1
    stats1 = process_object(1, stl_path1, scale_factor1, subdiv_level1, z_squash1, material_key1, x1, y1, z1, rx1, ry1, rz1)
    all_stats.extend(stats1)

    # Process Object 2 if enabled
    if 'enabled' in (enable_obj2 or []):
        stats2 = process_object(2, stl_path2, scale_factor2, subdiv_level2, z_squash2, material_key2, x2, y2, z2, rx2, ry2, rz2)
        all_stats.extend(stats2)

    # Transducers
    fig.add_trace(go.Scatter3d(x=SOURCES[:, 0], y=SOURCES[:, 1], z=SOURCES[:, 2],
                               mode='markers', marker={'size': 3, 'color': 'white', 'opacity': 0.4}, name='Transducers'))

    fig.update_layout(
        template='plotly_dark',
        scene={'xaxis': {'range': [-50, 50]}, 'yaxis': {'range': [-50, 50]}, 'zaxis': {'range': [-50, 50]}, 'aspectmode': 'cube'},
        margin={'l': 0, 'r': 0, 't': 0, 'b': 0}, uirevision='stable', scene_camera=camera_data
    )
    
    return fig, all_stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="3D_Files/Tetrahedron.stl")
    parser.add_argument("--scale", type=float, default=0.04) # 50mm * 0.04 = 2mm object
    args = parser.parse_args()

    global STL_PATH, SOURCES, SCALE_FACTOR
    STL_PATH = args.file
    SCALE_FACTOR = args.scale
    
    src_file = "AcousticFieldModeling/srcarray.txt"
    if os.path.exists(src_file):
        raw = np.loadtxt(src_file)
        SOURCES = np.column_stack((raw[:, 0], raw[:, 2], raw[:, 1]))
    else:
        SOURCES = np.array([[0, 0, 40], [0, 0, -40]])

    print(f"Server is preparing to launch... Dash will automatically push physics updates.")
    print(f"Open your browser to: http://127.0.0.1:8095")
    app.run(debug=True, port=8095, host='0.0.0.0')
