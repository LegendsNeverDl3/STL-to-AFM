import numpy as np
import trimesh
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import argparse
import webbrowser
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
def compute_simplified_forces(centroids, normals, face_areas, sources, phases=None):
    """Simplified radiation pressure model: F = p^2/(rho*c^2) * (-n) * A"""
    med = get_medium()
    rho = med["rho"]
    c = med["c"]

    p_complex = compute_complex_pressure(centroids, sources, phases)
    p_amplitude = np.abs(p_complex)
    v_scalar = p_amplitude / (rho * c)

    p_rad = (p_amplitude ** 2) / (rho * c ** 2)
    f_acoustic = p_rad[:, np.newaxis] * (-normals) * face_areas[:, np.newaxis]

    return p_amplitude, v_scalar, f_acoustic


# --- Gorkov model ---
def compute_gorkov_forces(centroids, sources, mesh_volume, phases=None):
    """Full Gorkov potential model: F = -grad(U)"""
    f1, f2 = get_contrast_factors()

    p_complex = compute_complex_pressure(centroids, sources, phases)
    p_amplitude = np.abs(p_complex)
    v_vectors, v_speed = compute_velocity_vector(centroids, sources, phases)
    gorkov_U = compute_gorkov_potential(p_complex, v_speed, mesh_volume, f1, f2)
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
            .Select-control { background-color: #0d0e1a !important; border: 1px solid #30363d !important; height: 32px !important; }
            .Select-value-label { color: white !important; }
            .Select-menu-outer { background-color: #1e2030 !important; color: white !important; }
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

app.layout = dbc.Container([
    dcc.Store(id='camera-store', data={'eye': {'x': 1.25, 'y': 1.25, 'z': 1.25}}),
    
    dbc.Row([
        dbc.Col(html.H1("Acoustic Levitation Simulator", className="text-center my-4", style={'color': '#00e0ff', 'fontWeight': '200', 'letterSpacing': '4px'}), width=12)
    ]),
    
    dbc.Row([
        # --- LEFT SIDEBAR (Width 3) ---
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("CONTROLS"),
                dbc.CardBody([
                    dbc.Accordion([
                        # Category 1: Environment Setup
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
                            
                            html.Label("Object Material", className="control-label"),
                            dcc.Dropdown(
                                id='material-preset',
                                options=[{'label': m['name'], 'value': k} for k, m in MATERIALS.items()],
                                value='polystyrene_foam', clearable=False, className="mb-3"
                            ),
                            
                            html.Label("Sound Power (%)", className="control-label"),
                            dcc.Slider(id='sound-power', min=1, max=100, step=1, value=100, tooltip={"always_visible": False}),
                            
                            html.Label("Phase Shift (°)", className="control-label", style={'marginTop': '10px'}),
                            dcc.Slider(id='phase-shift', min=0, max=360, step=1, value=0),
                        ], title="Environment Setup"),
                        
                        # Category 2: Physics Toggles
                        dbc.AccordionItem([
                            dbc.Checklist(
                                id='field-toggles',
                                options=[
                                    {'label': ' Pressure Mapping', 'value': 'pressure_color'},
                                    {'label': ' Velocity Field', 'value': 'velocity_arrows'},
                                    {'label': ' Acoustic Force', 'value': 'acoustic_arrows'},
                                    {'label': ' Gravitational F', 'value': 'gravity_arrows'},
                                    {'label': ' GLOBAL NET FORCE', 'value': 'net_force_arrows'},
                                ],
                                value=['pressure_color', 'acoustic_arrows'],
                                className="radio-group", style={'fontSize': '0.85rem'}
                            ),
                        ], title="Physics Visualization"),
                        
                        # Category 3: Translation (6-DOF)
                        dbc.AccordionItem([
                            *[html.Div([
                                dbc.Row([
                                    dbc.Col(html.Label(f"{axis}", className="control-label"), width=6),
                                    dbc.Col(dbc.Input(id=f'inp-{axis.lower()}', type='number', value=0, size="sm", className="dark-input"), width=6)
                                ]),
                                dcc.Slider(id=f'pos-{axis.lower()}', min=-40 if axis=='Z' else -25, max=40 if axis=='Z' else 25, 
                                           step=0.1, value=0)
                            ], className="mb-3") for axis in ['X', 'Y', 'Z']],
                        ], title="Object Translation"),
                        
                        # Category 4: Rotation (6-DOF)
                        dbc.AccordionItem([
                            *[html.Div([
                                dbc.Row([
                                    dbc.Col(html.Label(f"{label}", className="control-label"), width=6),
                                    dbc.Col(dbc.Input(id=f'inp-r{axis}', type='number', value=0, size="sm", className="dark-input"), width=6)
                                ]),
                                dcc.Slider(id=f'rot-{axis}', min=0, max=360, step=1, value=0)
                            ], className="mb-3") for axis, label in zip(['x', 'y', 'z'], ['Pitch (X)', 'Yaw (Y)', 'Roll (Z)'])]
                        ], title="Object Rotation"),
                    ], start_collapsed=True, flush=True),
                    
                    html.Hr(style={'borderColor': 'rgba(255,255,255,0.1)'}),
                    html.Div(id='stats-target', className="stats-text")
                ])
            ], className="control-card")
        ], width=3),
        
        # --- MAIN SIMULATOR VIEW (Width 9) ---
        dbc.Col([
            dbc.Card([
                dcc.Graph(id='live-graph', style={'height': '85vh'}, config={'displayModeBar': False})
            ], style={'borderRadius': '15px', 'overflow': 'hidden', 'border': 'none', 'boxShadow': '0 4px 20px rgba(0,0,0,0.3)'})
        ], width=9)
    ])
], fluid=True)

@app.callback(
    [Output('pos-x', 'value'), Output('pos-y', 'value'), Output('pos-z', 'value'),
     Output('rot-x', 'value'), Output('rot-y', 'value'), Output('rot-z', 'value'),
     Output('inp-x', 'value'), Output('inp-y', 'value'), Output('inp-z', 'value'),
     Output('inp-rx', 'value'), Output('inp-ry', 'value'), Output('inp-rz', 'value'),
     Output('camera-store', 'data')],
    [Input('pos-x', 'value'), Input('pos-y', 'value'), Input('pos-z', 'value'),
     Input('rot-x', 'value'), Input('rot-y', 'value'), Input('rot-z', 'value'),
     Input('inp-x', 'value'), Input('inp-y', 'value'), Input('inp-z', 'value'),
     Input('inp-rx', 'value'), Input('inp-ry', 'value'), Input('inp-rz', 'value'),
     Input('live-graph', 'relayoutData')],
    [State('rotation-mode', 'value'), State('camera-store', 'data')]
)
def sync_controls(sx, sy, sz, srx, sry, srz, ix, iy, iz, irx, iry, irz, relayout, mode, camera_data):
    ctx = callback_context
    if not ctx.triggered:
        return sx, sy, sz, srx, sry, srz, ix, iy, iz, irx, iry, irz, camera_data
    
    tid = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if tid == 'live-graph' and relayout and 'scene.camera' in relayout:
        new_cam = relayout['scene.camera']
        if mode == 'world':
            return sx, sy, sz, srx, sry, srz, ix, iy, iz, irx, iry, irz, new_cam
        else:
            eye = new_cam.get('eye', camera_data['eye'])
            new_rz = (np.degrees(np.arctan2(eye['y'], eye['x'])) + 45) % 360
            new_rx = (np.degrees(np.arctan2(eye['z'], np.sqrt(eye['x']**2 + eye['y']**2)))) % 360
            return sx, sy, sz, round(new_rx), sry, round(new_rz), sx, sy, sz, round(new_rx), sry, round(new_rz), camera_data

    if tid.startswith('pos-') or tid.startswith('rot-'):
        return sx, sy, sz, srx, sry, srz, sx, sy, sz, srx, sry, srz, camera_data
    if tid.startswith('inp-'):
        return (ix or 0), (iy or 0), (iz or 0), (irx or 0), (iry or 0), (irz or 0), \
               (ix or 0), (iy or 0), (iz or 0), (irx or 0), (iry or 0), (irz or 0), camera_data

    return sx, sy, sz, srx, sry, srz, ix, iy, iz, irx, iry, irz, camera_data

@app.callback(
    [Output('live-graph', 'figure'), Output('stats-target', 'children')],
    [Input('pos-x', 'value'), Input('pos-y', 'value'), Input('pos-z', 'value'), 
     Input('rot-x', 'value'), Input('rot-y', 'value'), Input('rot-z', 'value'),
     Input('force-model', 'value'), Input('material-preset', 'value'),
     Input('field-toggles', 'value'), Input('phase-shift', 'value'), Input('sound-power', 'value')],
    [State('live-graph', 'relayoutData'), State('rotation-mode', 'value'), State('camera-store', 'data')]
)
def update_physics(x, y, z, rx, ry, rz, force_model, material_key, field_toggles, phase_shift_deg, sound_power_pct,
                   relayout, mode, camera_data):
    # --- Update active material and parameters ---
    mat_module.ACTIVE_MATERIAL = material_key
    
    # Scale base amplitude by power percentage
    power_mult = (sound_power_pct or 100) / 100.0
    effective_amplitude = AMPLITUDE * power_mult

    # --- Load and transform mesh ---
    mesh = trimesh.load(STL_PATH)
    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(meshes) if meshes else mesh.to_geometry()
    mesh.apply_scale(SCALE_FACTOR)
    mesh.vertices -= mesh.bounding_box.centroid
    
    # Apply Standard Euler Sequential Rotations (X -> Y -> Z)
    r_x = trimesh.transformations.rotation_matrix(np.radians(rx), [1, 0, 0])
    r_y = trimesh.transformations.rotation_matrix(np.radians(ry), [0, 1, 0])
    r_z = trimesh.transformations.rotation_matrix(np.radians(rz), [0, 0, 1])
    t_mat = trimesh.transformations.translation_matrix([x, y, z])
    mesh.apply_transform(trimesh.transformations.concatenate_matrices(t_mat, r_z, r_y, r_x))
    
    c_pts, n_pts, a_pts = mesh.triangles_center, mesh.face_normals, mesh.area_faces

    # --- Parse toggles ---
    show_pressure = 'pressure_color' in (field_toggles or [])
    show_velocity = 'velocity_arrows' in (field_toggles or [])
    show_acoustic_force = 'acoustic_arrows' in (field_toggles or [])
    show_gravity = 'gravity_arrows' in (field_toggles or [])
    show_net_force = 'net_force_arrows' in (field_toggles or [])

    # --- Compute phase array ---
    phases = np.zeros(len(SOURCES))
    phases[SOURCES[:, 2] > 0] = np.radians(phase_shift_deg)

    # --- Compute acoustic field based on selected model ---
    stats_lines = []
    mat_info = get_material(material_key)
    f1, f2 = get_contrast_factors(material_key)

    if force_model == 'gorkov':
        # Gorkov evaluates body forces. To map this to surface nodes like Simplified model,
        # we treat each face as a discrete sub-particle containing an equal fraction of the mass/volume.
        vol_per_face = mesh.volume / max(1, len(a_pts))
        p_amp, v_speed, v_vectors, gorkov_U, gorkov_force = \
            compute_gorkov_forces(c_pts, SOURCES, vol_per_face, phases=phases)
            
        # Manually scale forces by amp multiplier squared for Gorkov (U ~ P^2)
        f_acoustic = gorkov_force * (power_mult ** 2)
        p_amp *= power_mult # p scales linearly
        model_label = "Gorkov"
        stats_lines.append(html.Div([html.Span("MODEL: ", style={'color': '#888'}), "Gorkov Potential"]))
        stats_lines.append(html.Div([html.Span("PHASE: ", style={'color': '#888'}), f"Top offset {phase_shift_deg}°"]))
        stats_lines.append(html.Div([html.Span("f1: ", style={'color': '#888'}), f"{f1:.4f}",
                                     html.Span("  f2: ", style={'color': '#888'}), f"{f2:.4f}"]))
        if gorkov_U is not None:
            stats_lines.append(html.Div([html.Span("GORKOV U: ", style={'color': '#888'}),
                                         f"min={np.min(gorkov_U):.2e} max={np.max(gorkov_U):.2e}"]))
    else:
        p_amp, v_scalar, f_acoustic = compute_simplified_forces(c_pts, n_pts, a_pts, SOURCES, phases=phases)
        # Simplified radiation pressure scales with P^2
        f_acoustic *= (power_mult ** 2)
        p_amp *= power_mult
        v_speed = v_scalar * power_mult
        v_vectors = None
        gorkov_U = None
        model_label = "Simplified"
        stats_lines.append(html.Div([html.Span("MODEL: ", style={'color': '#888'}), "Simplified Radiation Pressure"]))

    # --- Gravity ---
    total_mass = mesh.volume * mat_info['rho']  # g (in mm/g unit system)
    f_gravity = np.zeros_like(f_acoustic)
    # The trandducers were mapped to the Z-axis on load, so gravity should pull down -Z
    # TODO: Make gravity direction dependent on transducer orientation / Fixed
    f_gravity[:, 2] = -(total_mass * 9806.65) / len(a_pts)  # gravity in mm/s^2

    
    f_net = f_acoustic + f_gravity
    net_v = np.sum(f_net, axis=0)
    
    # Calculate global sums for debugging
    total_acoustic_f = np.sum(f_acoustic, axis=0)
    total_gravity_f = np.sum(f_gravity, axis=0)

    # --- Build Figure ---
    fig = go.Figure()

    # 1. Mesh — colored by pressure or plain
    if show_pressure:
        fig.add_trace(go.Mesh3d(
            x=mesh.vertices[:, 0], y=mesh.vertices[:, 1], z=mesh.vertices[:, 2],
            i=mesh.faces[:, 0], j=mesh.faces[:, 1], k=mesh.faces[:, 2],
            intensity=p_amp, intensitymode='cell',
            colorscale='Viridis', colorbar=dict(title='Pressure (Pa)', x=1.0, len=0.5, y=0.75),
            opacity=1.0, flatshading=True,
            lighting=dict(ambient=0.45, diffuse=0.8, specular=0.4, roughness=0.2),
            name='Acoustic Target',
        ))
    else:
        fig.add_trace(go.Mesh3d(
            x=mesh.vertices[:, 0], y=mesh.vertices[:, 1], z=mesh.vertices[:, 2],
            i=mesh.faces[:, 0], j=mesh.faces[:, 1], k=mesh.faces[:, 2],
            color='#00e0ff', opacity=0.3, flatshading=True,
            lighting=dict(ambient=0.45, diffuse=0.8, specular=0.4, roughness=0.2),
            name='Acoustic Target',
        ))

    # Add white outline (wireframe) around each polygon to make it clearly visible
    if hasattr(mesh, 'edges_unique'):
        edge_vertices = mesh.vertices[mesh.edges_unique]
        lines_x = np.insert(edge_vertices[:, :, 0], 2, np.nan, axis=1).flatten()
        lines_y = np.insert(edge_vertices[:, :, 1], 2, np.nan, axis=1).flatten()
        lines_z = np.insert(edge_vertices[:, :, 2], 2, np.nan, axis=1).flatten()
        fig.add_trace(go.Scatter3d(
            x=lines_x, y=lines_y, z=lines_z,
            mode='lines', line=dict(color='white', width=2),
            name='Polygon Lines', showlegend=False,
            hoverinfo='none'
        ))

    # --- Arrow drawing helper using 3D cones ---
    def draw_arrows(origins, vecs, color, name, scale=6.0, ref_mag=None):
        mags = np.linalg.norm(vecs, axis=1)
        if ref_mag is None:
            # Native auto-scaling: find the average non-zero magnitude
            avg = np.mean(mags[mags > 0]) if np.any(mags > 0) else 1.0
        else:
            avg = ref_mag
            
        # Subsample for performance
        step = max(1, len(origins) // 300)
        idx = np.arange(0, len(origins), step)
        
        ox, oy, oz = origins[idx, 0], origins[idx, 1], origins[idx, 2]
        u, v, w = vecs[idx, 0], vecs[idx, 1], vecs[idx, 2]
        m_sub = mags[idx]
        # Linearly scale relative to the reference magnitude
        vis_scale = scale / avg
        
        # Logarithmic magnitude scaling for wide dynamic range
        # Result: Arrow shrinks clearly when approaching zero, but doesn't grow infinitely large.
        if ref_mag is not None:
            # Normalized log scale: y = scale * log1p(mag / ref)
            log_mags = np.log1p(mags[idx] / ref_mag)
            vis_mags = scale * log_mags
        else:
            vis_mags = mags[idx] * vis_scale
        
        max_multiplier = 15.0
        
        with np.errstate(divide='ignore', invalid='ignore'):
            cap_factors = np.ones_like(vis_mags)
            mask = vis_mags > (scale * max_multiplier)
            if np.any(mask):
                cap_factors[mask] = (scale * max_multiplier) / vis_mags[mask]
                
            # Compute unit vectors and scale by vis_mags
            norm_u = np.zeros_like(u)
            norm_v = np.zeros_like(v)
            norm_w = np.zeros_like(w)
            
            non_zero = mags[idx] > 1e-12
            norm_u[non_zero] = u[non_zero] / mags[idx][non_zero]
            norm_v[non_zero] = v[non_zero] / mags[idx][non_zero]
            norm_w[non_zero] = w[non_zero] / mags[idx][non_zero]
            
            u = norm_u * vis_mags * cap_factors
            v = norm_v * vis_mags * cap_factors
            w = norm_w * vis_mags * cap_factors
            
            # Additional visual logic: if the vector is nearly zero, hide it
            if np.all(vis_mags < 1e-9):
                return
            
        dynamic_sizeref = scale * 1.5
            
        fig.add_trace(go.Cone(
            x=ox, y=oy, z=oz,
            u=u, v=v, w=w,
            sizemode="absolute",
            sizeref=dynamic_sizeref, 
            anchor="tip",          
            colorscale=[[0, color], [1, color]], 
            showscale=False,
            name=name,
            showlegend=True
        ))

    # 2. Velocity arrows
    if show_velocity:
        if v_vectors is not None:
            v_real = np.real(v_vectors)
            draw_arrows(c_pts, v_real, "#00e0ff", "Velocity", scale=4.0)
        else:
            # For simplified mode, velocity is scalar along -normal 
            v_arrow = v_speed[:, np.newaxis] * (-n_pts)
            draw_arrows(c_pts, v_arrow, "#00e0ff", "Velocity (scalar)", scale=4.0)

    # Disable dynamic auto-scaling. A dynamic scale causes arrows to "blow up" to
    # the maximum screen size even when they are physically near zero.
    # By using a fixed reference magnitude, 1 unit of visual arrow length 
    # perfectly equals a fixed physical unit of force (g mm / s^2) across the board.
    fixed_physics_scale = 10000.0

    # 3. Acoustic force arrows (Local Surface Squeezing Forces)
    if show_acoustic_force:
        draw_arrows(c_pts, f_acoustic, "#ff00ff", f"Acoustic Force ({model_label})", scale=2.0)

    # Calculate absolute global gravity for rendering scale reference
    global_gravity_mag = total_mass * 9806.65
    
    # 4. Gravity arrow (Single Global Free-Body Force)
    if show_gravity:
        global_g_vec = np.array([[0.0, 0.0, -global_gravity_mag]])
        draw_arrows(np.array([[x, y, z]]), global_g_vec, "#ff0000", "Global Gravity", scale=5.0, ref_mag=global_gravity_mag)

    # 5. Net force arrow (Single Global Free-Body Force)
    if show_net_force:
        draw_arrows(np.array([[x, y, z]]), np.array([net_v]), "#ff8800", "Global Net Force", scale=5.0, ref_mag=global_gravity_mag)

    # 6. Transducer positions
    fig.add_trace(go.Scatter3d(x=SOURCES[:, 0], y=SOURCES[:, 1], z=SOURCES[:, 2],
                               mode='markers',
                               marker={'size': 4, 'color': 'white', 'opacity': 0.7},
                               name='Transducers'))

    limit = 50
    fig.update_layout(
        template='plotly_dark',
        scene={'xaxis': {'range': [-limit, limit], 'gridcolor': '#333'},
               'yaxis': {'range': [-limit, limit], 'gridcolor': '#333'},
               'zaxis': {'range': [-limit, limit], 'gridcolor': '#333'},
               'aspectmode': 'cube'},
        margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
        uirevision='stable',
        scene_camera=camera_data,
        legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.05, bgcolor="rgba(0,0,0,0.5)")
    )
    
    # --- Stats panel ---
    stats_lines.append(html.Div([html.Span("MATERIAL: ", style={'color': '#888'}), mat_info['name']]))
    stats_lines.append(html.Div([html.Span("MASS: ", style={'color': '#888'}), f"{total_mass:.4e} g ({total_mass*1000:.2f} mg)"]))
    stats_lines.append(html.Hr(style={'margin': '5px 0'}))
    stats_lines.append(html.Div([html.Span("NET FORCE: ", style={'color': '#888'}), f"{np.linalg.norm(net_v):.4e}"]))
    stats_lines.append(html.Div([html.Span("  Acoustic (Z): ", style={'color': '#888'}), f"{total_acoustic_f[2]:+.2e}"]))
    stats_lines.append(html.Div([html.Span("  Gravity  (Z): ", style={'color': '#888'}), f"{total_gravity_f[2]:+.2e}"]))
    stats_lines.append(html.Div([html.Span("VECTOR: ", style={'color': '#888'}),
                                  f"[{net_v[0]:.2e}, {net_v[1]:.2e}, {net_v[2]:.2e}]"]))
    stats_lines.append(html.Div([html.Span("MAX PRESSURE: ", style={'color': '#888'}), f"{np.max(p_amp):.0f} Pa"]))
    stats_lines.append(html.Div([html.Span("MAX VELOCITY: ", style={'color': '#888'}), f"{np.max(v_speed):.2e} mm/s"]))
    
    return fig, [html.Div(stats_lines)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="3D_Files/cube_50mm.stl")
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
