from fasthtml.common import *
import base64
import io
from PIL import Image
import numpy as np
import torch
import cv2

# --- MiDaS Depth Estimation Setup ---
model_type = "DPT_Large"  # or "DPT_Hybrid" / "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
if model_type in ["DPT_Large", "DPT_Hybrid"]:
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

# --- Utility: Generate an OBJ file (as a string) from a depth map ---
def generate_obj_from_depth(depth_map, scale_x=1.0, scale_y=1.0, scale_z=0.1):
    """
    Generate a grid mesh OBJ string from a 2D depth map.
    Each pixel becomes a vertex; each grid cell becomes two triangles.
    """
    H, W = depth_map.shape
    vertices = []
    faces = []
    # Center the grid so (0,0) is at the center.
    for i in range(H):
        for j in range(W):
            z = float(depth_map[i, j]) * scale_z
            x = (j - W/2) * scale_x
            y = (i - H/2) * scale_y
            vertices.append(f"v {x} {y} {z}")
    # Create faces (two triangles per grid cell)
    for i in range(H - 1):
        for j in range(W - 1):
            v1 = i * W + j + 1
            v2 = i * W + (j + 1) + 1
            v3 = (i + 1) * W + j + 1
            v4 = (i + 1) * W + (j + 1) + 1
            faces.append(f"f {v1} {v2} {v3}")
            faces.append(f"f {v3} {v2} {v4}")
    return "\n".join(vertices + faces)

# --- FastHTML App Setup ---
app, rt = fast_app()

@rt('/')
def home():
    return HTMLResponse("""
    <h1>Upload a 2D Image to Convert to a 3D Mesh</h1>
    <form action="/convert" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required>
        <button type="submit">Convert to 3D Model</button>
    </form>
    """)

@rt('/convert', methods=['POST'])
async def convert(request):
    form = await request.form()
    uploaded_file = form.get('file')
    if not uploaded_file:
        return HTMLResponse("<p>No file uploaded!</p>")
    
    file_bytes = await uploaded_file.read()
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    image_np = np.array(image)
    input_batch = transform(image_np).to(device)
    
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image_np.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth_map = prediction.cpu().numpy()
    
    # Generate OBJ mesh from the depth map (pseudo-3D)
    obj_str = generate_obj_from_depth(depth_map, scale_x=1.0, scale_y=1.0, scale_z=0.1)
    
    # Build HTML that uses Three.js (with an import map) to load the OBJ model.
    # Note: All curly braces intended for the JS object are doubled.
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="UTF-8">
        <title>3D Mesh Viewer</title>
        <style>
          body {{ margin: 0; overflow: hidden; }}
          #viewer {{ width: 100vw; height: 100vh; }}
        </style>
        <script type="importmap">
        {{
          "imports": {{
            "three": "https://cdn.jsdelivr.net/npm/three@0.146.0/build/three.module.js",
            "three/examples/jsm/loaders/OBJLoader.js": "https://cdn.jsdelivr.net/npm/three@0.146.0/examples/jsm/loaders/OBJLoader.js",
            "three/examples/jsm/controls/OrbitControls.js": "https://cdn.jsdelivr.net/npm/three@0.146.0/examples/jsm/controls/OrbitControls.js"
          }}
        }}
        </script>
      </head>
      <body>
        <div id="viewer"></div>
        <script type="module">
          import * as THREE from "three";
          import {{ OBJLoader }} from "three/examples/jsm/loaders/OBJLoader.js";
          import {{ OrbitControls }} from "three/examples/jsm/controls/OrbitControls.js";
          
          console.log("Three.js, OBJLoader, and OrbitControls loaded");
          
          // Create scene, camera, and renderer
          const scene = new THREE.Scene();
          scene.background = new THREE.Color(0xf0f0f0);
          
          const camera = new THREE.PerspectiveCamera(75, window.innerWidth/window.innerHeight, 0.1, 1000);
          camera.position.set(0, 0, 100);
          camera.lookAt(0, 0, 0);
          
          const renderer = new THREE.WebGLRenderer({{ antialias: true }});
          renderer.setSize(window.innerWidth, window.innerHeight);
          document.getElementById("viewer").appendChild(renderer.domElement);
          
          // Add OrbitControls
          const controls = new OrbitControls(camera, renderer.domElement);
          controls.update();
          
          // Add basic lighting
          const ambientLight = new THREE.AmbientLight(0xcccccc, 0.4);
          scene.add(ambientLight);
          const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
          directionalLight.position.set(1, 1, 1);
          scene.add(directionalLight);
          
          // Create a blob URL for the OBJ data generated from the depth map
          const objData = `{obj_str}`;
          const blob = new Blob([objData], {{ type: 'text/plain' }});
          const objURL = URL.createObjectURL(blob);
          
          // Load the OBJ model using OBJLoader
          const loader = new OBJLoader();
          loader.load(objURL, function(object) {{
              scene.add(object);
              camera.lookAt(object.position);
          }}, function(xhr) {{
              console.log((xhr.loaded / xhr.total * 100) + '% loaded');
          }}, function(error) {{
              console.error('Error loading OBJ model:', error);
          }});
          
          // Animation loop
          function animate() {{
              requestAnimationFrame(animate);
              controls.update();
              renderer.render(scene, camera);
          }}
          animate();
          
          // Resize handler
          window.addEventListener('resize', () => {{
              camera.aspect = window.innerWidth / window.innerHeight;
              camera.updateProjectionMatrix();
              renderer.setSize(window.innerWidth, window.innerHeight);
          }});
        </script>
      </body>
    </html>
    """
    return HTMLResponse(html_content)

serve()







# from fasthtml.common import *
# import base64
# import io
# from PIL import Image
# import numpy as np
# import torch
# import cv2

# # --- MiDaS Depth Estimation Setup ---
# model_type = "DPT_Large"  # or "DPT_Hybrid" or "MiDaS_small"
# midas = torch.hub.load("intel-isl/MiDaS", model_type)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# midas.to(device)
# midas.eval()

# midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
# if model_type in ["DPT_Large", "DPT_Hybrid"]:
#     transform = midas_transforms.dpt_transform
# else:
#     transform = midas_transforms.small_transform

# # --- Utility: Generate an OBJ file (as a string) from a depth map ---
# def generate_obj_from_depth(depth_map, scale_x=1.0, scale_y=1.0, scale_z=0.1):
#     """
#     Generate a grid mesh OBJ string from a 2D depth map.
#     Each pixel becomes a vertex; each grid cell becomes two triangles.
#     """
#     H, W = depth_map.shape
#     vertices = []
#     faces = []
#     # Center the grid so (0,0) is at the center.
#     for i in range(H):
#         for j in range(W):
#             z = float(depth_map[i, j]) * scale_z
#             x = (j - W/2) * scale_x
#             y = (i - H/2) * scale_y
#             vertices.append(f"v {x} {y} {z}")
#     # Create faces (two triangles per grid cell)
#     for i in range(H - 1):
#         for j in range(W - 1):
#             v1 = i * W + j + 1
#             v2 = i * W + (j + 1) + 1
#             v3 = (i + 1) * W + j + 1
#             v4 = (i + 1) * W + (j + 1) + 1
#             faces.append(f"f {v1} {v2} {v3}")
#             faces.append(f"f {v3} {v2} {v4}")
#     return "\n".join(vertices + faces)

# # --- FastHTML App Setup ---
# app, rt = fast_app()

# @rt('/')
# def home():
#     return HTMLResponse("""
#     <h1>Upload a 2D Image to Convert to a 3D Mesh</h1>
#     <form action="/convert" method="post" enctype="multipart/form-data">
#         <input type="file" name="file" accept="image/*" required>
#         <button type="submit">Convert to 3D Model</button>
#     </form>
#     """)

# @rt('/convert', methods=['POST'])
# async def convert(request):
#     form = await request.form()
#     uploaded_file = form.get('file')
#     if not uploaded_file:
#         return HTMLResponse("<p>No file uploaded!</p>")
    
#     file_bytes = await uploaded_file.read()
#     image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
#     image_np = np.array(image)
#     input_batch = transform(image_np).to(device)
    
#     with torch.no_grad():
#         prediction = midas(input_batch)
#         prediction = torch.nn.functional.interpolate(
#             prediction.unsqueeze(1),
#             size=image_np.shape[:2],
#             mode="bicubic",
#             align_corners=False,
#         ).squeeze()
#     depth_map = prediction.cpu().numpy()
    
#     # Generate OBJ mesh from depth map
#     obj_str = generate_obj_from_depth(depth_map, scale_x=1.0, scale_y=1.0, scale_z=0.1)
    
#     # Build HTML to display the 3D model using Three.js (with an import map)
#     # Note: All curly braces not intended for Python interpolation are doubled.
#     html_content = f"""
#     <!DOCTYPE html>
#     <html lang="en">
#       <head>
#         <meta charset="UTF-8">
#         <title>3D Mesh Viewer</title>
#         <style>
#           body {{ margin: 0; overflow: hidden; }}
#           #viewer {{ width: 100vw; height: 100vh; }}
#         </style>
#         <script type="importmap">
#         {{
#           "imports": {{
#             "three": "https://cdn.jsdelivr.net/npm/three@0.146.0/build/three.module.js",
#             "three/examples/jsm/loaders/OBJLoader.js": "https://cdn.jsdelivr.net/npm/three@0.146.0/examples/jsm/loaders/OBJLoader.js",
#             "three/examples/jsm/controls/OrbitControls.js": "https://cdn.jsdelivr.net/npm/three@0.146.0/examples/jsm/controls/OrbitControls.js"
#           }}
#         }}
#         </script>
#       </head>
#       <body>
#         <div id="viewer"></div>
#         <script type="module">
#           import * as THREE from "three";
#           import {{ OBJLoader }} from "three/examples/jsm/loaders/OBJLoader.js";
#           import {{ OrbitControls }} from "three/examples/jsm/controls/OrbitControls.js";
          
#           console.log("Three.js, OBJLoader, and OrbitControls loaded");
          
#           // Create scene, camera, and renderer
#           const scene = new THREE.Scene();
#           scene.background = new THREE.Color(0xf0f0f0);
          
#           const camera = new THREE.PerspectiveCamera(75, window.innerWidth/window.innerHeight, 0.1, 1000);
#           camera.position.set(0, 0, 100);
#           camera.lookAt(0, 0, 0);
          
#           const renderer = new THREE.WebGLRenderer({{ antialias: true }});
#           renderer.setSize(window.innerWidth, window.innerHeight);
#           document.getElementById("viewer").appendChild(renderer.domElement);
          
#           // Add OrbitControls
#           const controls = new OrbitControls(camera, renderer.domElement);
#           controls.update();
          
#           // Add lights
#           const ambientLight = new THREE.AmbientLight(0xcccccc, 0.4);
#           scene.add(ambientLight);
#           const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
#           directionalLight.position.set(1, 1, 1);
#           scene.add(directionalLight);
          
#           // Create a blob URL for the OBJ data
#           const objData = `{obj_str}`;
#           const blob = new Blob([objData], {{ type: 'text/plain' }});
#           const objURL = URL.createObjectURL(blob);
          
#           // Load the OBJ model
#           const loader = new OBJLoader();
#           loader.load(objURL, function(object) {{
#               scene.add(object);
#           }}, undefined, function(error) {{
#               console.error('Error loading OBJ model:', error);
#           }});
          
#           // Animation loop
#           function animate() {{
#               requestAnimationFrame(animate);
#               controls.update();
#               renderer.render(scene, camera);
#           }}
#           animate();
          
#           // Resize handler
#           window.addEventListener('resize', () => {{
#               camera.aspect = window.innerWidth / window.innerHeight;
#               camera.updateProjectionMatrix();
#               renderer.setSize(window.innerWidth, window.innerHeight);
#           }});
#         </script>
#       </body>
#     </html>
#     """
#     return HTMLResponse(html_content)

# serve()
