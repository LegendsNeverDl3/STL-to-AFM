import numpy as np
import trimesh
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import argparse
import webbrowser
from dash import Dash, dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from flask_cors import CORS

def acoustic_pressure_coherent(points, sources, phases=None):
    rho = 1.225e-12 
    c = 343000.0   
    f = 40000.0    
    wl = c / f
    k = 2 * np.pi / wl
    diff = points[:, np.newaxis, :] - sources[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    r = np.clip(r, 1e-3, None)
    if phases is None:
        phases = np.zeros(len(sources))
    A = 4242.0 
    angle = k * r + phases[np.newaxis, :]
    complex_field = (A / r) * (np.cos(angle) + 1j * np.sin(angle))
    total_field = np.sum(complex_field, axis=1)
    pressure_amplitude = np.abs(total_field)
    velocity_amplitude = pressure_amplitude / (rho * c)
    return pressure_amplitude, velocity_amplitude

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
     Input('rot-x', 'value'), Input('rot-y', 'value'), Input('rot-z', 'value')],
    [State('live-graph', 'relayoutData'), State('rotation-mode', 'value'), State('camera-store', 'data')]
)
def update_physics(x, y, z, rx, ry, rz, relayout, mode, camera_data):
    mesh = trimesh.load(STL_PATH)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()
    mesh.apply_scale(SCALE_FACTOR)
    mesh.vertices -= mesh.bounding_box.centroid
    
    # Apply Standard Euler Sequential Rotations (X -> Y -> Z)
    r_x = trimesh.transformations.rotation_matrix(np.radians(rx), [1, 0, 0])
    r_y = trimesh.transformations.rotation_matrix(np.radians(ry), [0, 1, 0])
    r_z = trimesh.transformations.rotation_matrix(np.radians(rz), [0, 0, 1])
    t_mat = trimesh.transformations.translation_matrix([x, y, z])
    mesh.apply_transform(trimesh.transformations.concatenate_matrices(t_mat, r_z, r_y, r_x))
    
    c, n, a = mesh.triangles_center, mesh.face_normals, mesh.area_faces
    p, v = acoustic_pressure_coherent(c, SOURCES)
    p_rad = (p**2) / (1.225e-12 * (343000.0**2))
    f_acoustic = p_rad[:, np.newaxis] * (-n) * a[:, np.newaxis]
    
    total_mass = mesh.volume * 1000e-12
    f_gravity = np.zeros_like(f_acoustic)
    f_gravity[:, 2] = -(total_mass * 9806.65) / len(a)
    
    f_net = f_acoustic + f_gravity
    net_v = np.sum(f_net, axis=0)

    fig = go.Figure()
    fig.add_trace(go.Mesh3d(
        x=mesh.vertices[:,0], y=mesh.vertices[:,1], z=mesh.vertices[:,2],
        i=mesh.faces[:,0], j=mesh.faces[:,1], k=mesh.faces[:,2],
        color='#00e0ff', opacity=0.35, flatshading=True,
        lighting=dict(ambient=0.45, diffuse=0.8, specular=0.4, roughness=0.2),
        name='Acoustic Target'
    ))
    
    def draw_arrows(vecs, color, name, scale=6.0):
        mags = np.linalg.norm(vecs, axis=1)
        avg = np.mean(mags) if np.mean(mags) > 0 else 1
        u, v, w = (vecs / avg * scale).T 
        lx, ly, lz = [], [], []
        for i in range(len(c)):
            lx.extend([c[i,0], c[i,0] + u[i]*0.8, None])
            ly.extend([c[i,1], c[i,1] + v[i]*0.8, None])
            lz.extend([c[i,2], c[i,2] + w[i]*0.8, None])
        fig.add_trace(go.Scatter3d(x=lx, y=ly, z=lz, mode='lines', line={'color':color, 'width':5}, showlegend=False))

    draw_arrows(f_acoustic, "#00f0ff", "Acoustic Pressure")
    draw_arrows(f_gravity, "#ff5555", "Gravitational")
    draw_arrows(f_net, "#55ff55", "Net Force")

    fig.add_trace(go.Scatter3d(x=SOURCES[:,0], y=SOURCES[:,1], z=SOURCES[:,2], mode='markers', 
                               marker={'size': 4, 'color': 'white', 'opacity': 0.7}, name='Transducers'))

    limit = 50
    fig.update_layout(
        template='plotly_dark',
        scene={'xaxis':{'range':[-limit, limit], 'gridcolor': '#333'}, 
               'yaxis':{'range':[-limit, limit], 'gridcolor': '#333'}, 
               'zaxis':{'range':[-limit, limit], 'gridcolor': '#333'}, 
               'aspectmode':'cube'},
        margin={'l':0,'r':0,'t':0,'b':0},
        uirevision='stable',
        scene_camera=camera_data,
        legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.05, bgcolor="rgba(0,0,0,0.5)")
    )
    
    return fig, [
        html.Div([
            html.Div([html.Span("NET FORCE: ", style={'color': '#888'}), f"{np.linalg.norm(net_v):.2f} µN"]),
            html.Div([html.Span("VECTOR: ", style={'color': '#888'}), f"[{net_v[0]:.1f}, {net_v[1]:.1f}, {net_v[2]:.1f}]"]),
            html.Div([html.Span("MAX PRESSURE: ", style={'color': '#888'}), f"{np.max(p):.0f} Pa"])
        ])
    ]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="3D_Files/cube_50mm.stl")
    parser.add_argument("--scale", type=float, default=0.1)
    args = parser.parse_args()

    global STL_PATH, SOURCES, SCALE_FACTOR
    STL_PATH = args.file
    SCALE_FACTOR = args.scale
    
    src_file = "AcousticFieldModeling/srcarray.txt"
    if os.path.exists(src_file):
        raw = np.loadtxt(src_file)
        SOURCES = np.column_stack((raw[:, 0], raw[:, 2], raw[:, 1]))
    else:
        SOURCES = np.array([[0,0,40], [0,0,-40]])

    print(f"\n--- RESTARTING ON PORT 8095 ---\n")
    webbrowser.open("http://127.0.0.1:8095")
    app.run(debug=False, port=8095, host='0.0.0.0')
