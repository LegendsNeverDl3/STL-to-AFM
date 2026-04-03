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
            .black-text-input input { color: black !important; background-color: white !important; font-weight: bold !important; height: 32px !important; border: 1px solid #ced4da !important; border-radius: 4px !important; }
            .black-text-label { color: #212529 !important; font-weight: 700 !important; font-size: 0.9rem; }
            .stats-text { color: #000 !important; background-color: #f8f9fa !important; padding: 15px; border-radius: 10px; font-family: 'Consolas', monospace; font-size: 0.85rem; border-left: 5px solid #28a745; box-shadow: inset 0 1px 3px rgba(0,0,0,0.1); }
            .card-header { color: #212529 !important; background-color: #e9ecef !important; font-weight: 800 !important; font-size: 1.1rem; padding: 12px; border-bottom: 2px solid #dee2e6 !important; }
            .control-card { border: none !important; border-radius: 15px !important; box-shadow: 0 4px 20px rgba(0,0,0,0.4); overflow: hidden; }
            .section-header { color: #6c757d; font-size: 0.8rem; font-weight: 800; text-transform: uppercase; letter-spacing: 1px; margin-top: 15px; border-bottom: 1px solid #dee2e6; padding-bottom: 5px; margin-bottom: 10px; }
            .control-row { margin-bottom: 12px; padding: 0 5px; }
            body { background: radial-gradient(circle at center, #1e1e2f 0%, #0d0d12 100%) !important; }
            .radio-group label { color: #495057 !important; font-weight: 600 !important; }
            .dropdown-dark .Select-control { background-color: white !important; }
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
        dbc.Col(html.H1("Force/Mapping Simulator", className="text-center my-4", style={'color': 'white', 'fontWeight': '200', 'letterSpacing': '2px'}), width=12)
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Simulator Controls"),
                dbc.CardBody([
                    html.Div([
                        html.Label("Interaction Mode", className="black-text-label"),
                        dbc.RadioItems(
                            id='rotation-mode',
                            options=[
                                {'label': 'Independent (Rotate Object)', 'value': 'object'},
                                {'label': 'Global (Rotate World)', 'value': 'world'}
                            ],
                            value='world',
                            className="mb-3 radio-group",
                            inline=True
                        ),
                    ]),
                    
                    # --- Acoustic Field Controls ---
                    html.Div("Acoustic Field", className="section-header"),
                    
                    html.Div([
                        html.Label("Force Model", className="black-text-label"),
                        dcc.Dropdown(
                            id='force-model',
                            options=[
                                {'label': 'Simplified (p^2/rho*c^2)', 'value': 'simplified'},
                                {'label': 'Gorkov Potential', 'value': 'gorkov'},
                            ],
                            value='simplified',
                            clearable=False,
                            style={'color': 'black', 'fontWeight': '600'},
                        ),
                    ], className="control-row"),
                    
                    html.Div([
                        html.Label("Material Preset", className="black-text-label"),
                        dcc.Dropdown(
                                id='material-preset',
                                options=[{'label': m['name'], 'value': k} for k, m in MATERIALS.items()],
                                value='polystyrene_foam',
                                clearable=False,
                                style={'color': 'black', 'fontWeight': '600'},
                            ),
                    ], className="control-row"),
                    
                    html.Div([
                        html.Label("Phase Difference (Top vs Bottom °)", className="black-text-label"),
                        dcc.Slider(
                            id='phase-shift',
                            min=0, max=360, step=1, value=0,
                            marks={0: '0', 90: '90', 180: '180', 270: '270', 360: '360'},
                            tooltip={"placement": "bottom", "always_visible": False}
                        )
                    ], className="control-row", style={'marginTop': '15px'}),
                    
                    html.Div([
                        dbc.Checklist(
                            id='field-toggles',
                            options=[
                                {'label': '  Color by Pressure', 'value': 'pressure_color'},
                                {'label': '  Show Velocity (Cyan)', 'value': 'velocity_arrows'},
                                {'label': '  Show Acoustic Force (Magenta)', 'value': 'acoustic_arrows'},
                                {'label': '  Show Gravity (Red)', 'value': 'gravity_arrows'},
                                {'label': '  Show Net Force (Orange)', 'value': 'net_force_arrows'},
                            ],
                            value=['pressure_color', 'acoustic_arrows'],
                            className="radio-group",
                            style={'fontSize': '0.85rem'},
                        ),
                    ], className="control-row"),
                    
                    html.Div("Translation (mm)", className="section-header"),
                    # X Position
                    html.Div([
                        dbc.Row([
                            dbc.Col(html.Label("X-Axis", className="black-text-label"), width=7),
                            dbc.Col(dbc.Input(id='inp-x', type='number', value=0, className="black-text-input"), width=5)
                        ]),
                        dcc.Slider(id='pos-x', min=-25, max=25, step=1, value=0, marks={-25:'-25',0:'0',25:'25'}),
                    ], className="control-row"),
                    
                    # Y Position
                    html.Div([
                        dbc.Row([
                            dbc.Col(html.Label("Y-Axis", className="black-text-label"), width=7),
                            dbc.Col(dbc.Input(id='inp-y', type='number', value=0, className="black-text-input"), width=5)
                        ]),
                        dcc.Slider(id='pos-y', min=-25, max=25, step=1, value=0, marks={-25:'-25',0:'0',25:'25'}),
                    ], className="control-row"),
                    
                    # Z Position
                    html.Div([
                        dbc.Row([
                            dbc.Col(html.Label("Z-Axis", className="black-text-label"), width=7),
                            dbc.Col(dbc.Input(id='inp-z', type='number', value=0, className="black-text-input"), width=5)
                        ]),
                        dcc.Slider(id='pos-z', min=-40, max=40, step=1, value=0, marks={-40:'-40',0:'0',40:'40'}),
                    ], className="control-row"),
                    
                    html.Div("Rotation (Degrees)", className="section-header"),
                    # X Rotation
                    html.Div([
                        dbc.Row([
                            dbc.Col(html.Label("Pitch (X)", className="black-text-label"), width=7),
                            dbc.Col(dbc.Input(id='inp-rx', type='number', value=0, className="black-text-input"), width=5)
                        ]),
                        dcc.Slider(id='rot-x', min=0, max=360, step=1, value=0, marks={0:'0', 180:'180', 360:'360'}),
                    ], className="control-row"),
                    
                    # Y Rotation
                    html.Div([
                        dbc.Row([
                            dbc.Col(html.Label("Yaw (Y)", className="black-text-label"), width=7),
                            dbc.Col(dbc.Input(id='inp-ry', type='number', value=0, className="black-text-input"), width=5)
                        ]),
                        dcc.Slider(id='rot-y', min=0, max=360, step=1, value=0, marks={0:'0', 180:'180', 360:'360'}),
                    ], className="control-row"),
                    
                    # Z Rotation
                    html.Div([
                        dbc.Row([
                            dbc.Col(html.Label("Roll (Z)", className="black-text-label"), width=7),
                            dbc.Col(dbc.Input(id='inp-rz', type='number', value=0, className="black-text-input"), width=5)
                        ]),
                        dcc.Slider(id='rot-z', min=0, max=360, step=1, value=0, marks={0:'0', 180:'180', 360:'360'}),
                    ], className="control-row"),
                    
                    html.Hr(),
                    html.Div(id='stats-target', className="stats-text")
                ], style={'backgroundColor': '#f8f9fa', 'padding': '15px'})
            ], className="control-card")
        ], width=3),
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
     Input('field-toggles', 'value'), Input('phase-shift', 'value')],
    [State('live-graph', 'relayoutData'), State('rotation-mode', 'value'), State('camera-store', 'data')]
)
def update_physics(x, y, z, rx, ry, rz, force_model, material_key, field_toggles, phase_shift_deg,
                   relayout, mode, camera_data):
    # --- Update active material ---
    mat_module.ACTIVE_MATERIAL = material_key

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
        f_acoustic = gorkov_force
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
        v_speed = v_scalar
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
        vis_mags = m_sub * vis_scale
        
        # Enforce a generous cap: no arrow can be drawn larger than 15x the scale
        # This keeps normal vectors large, but stops gravity from exploding the screen.
        max_multiplier = 15.0
        
        with np.errstate(divide='ignore', invalid='ignore'):
            cap_factors = np.ones_like(vis_mags)
            mask = vis_mags > (scale * max_multiplier)
            if np.any(mask):
                cap_factors[mask] = (scale * max_multiplier) / vis_mags[mask]
                
            u = u * vis_scale * cap_factors
            v = v * vis_scale * cap_factors
            w = w * vis_scale * cap_factors
            
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

    # 5. Net force arrow (Single Global Free-Body Lift Force)
    if show_net_force:
        draw_arrows(np.array([[x, y, z]]), np.array([net_v]), "#ff8800", "Global Net Lift", scale=5.0, ref_mag=global_gravity_mag)

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
    stats_lines.append(html.Hr(style={'margin': '5px 0'}))
    stats_lines.append(html.Div([html.Span("NET FORCE: ", style={'color': '#888'}), f"{np.linalg.norm(net_v):.4e}"]))
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
