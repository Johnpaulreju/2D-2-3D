from fasthtml.common import *

app = FastHTML()

@app.route("/")
def get():
    return Div(
        # Add an import map for bare module specifiers
        Script("""
        {
          "imports": {
            "three": "https://cdn.jsdelivr.net/npm/three@0.146.0/build/three.module.js"
          }
        }
        """, type="importmap"),
        # Container for the Three.js canvas
        Div(id="threejs-canvas", style="width:100%; height:500px;"),
        # Our main module script that creates the scene and loads a GLB model
        Script("""
          import * as THREE from 'three';
          import { GLTFLoader } from 'https://cdn.jsdelivr.net/npm/three@0.146.0/examples/jsm/loaders/GLTFLoader.js';
          import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.146.0/examples/jsm/controls/OrbitControls.js';
          
          console.log("Three.js, GLTFLoader, and OrbitControls loaded");
          
          // Create scene, camera, and renderer
          const scene = new THREE.Scene();
          scene.background = new THREE.Color(0xdddddd);
          
          // Use full window aspect ratio for the camera
          const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
          camera.position.set(0, 1, 5);
          camera.lookAt(0, 0, 0);
          
          const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
          renderer.setSize(window.innerWidth, 500);
          document.getElementById("threejs-canvas").appendChild(renderer.domElement);
          
          // Add OrbitControls for interactive navigation
          const controls = new OrbitControls(camera, renderer.domElement);
          controls.update();
          
          // Add lights to the scene
          const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
          scene.add(ambientLight);
          const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
          directionalLight.position.set(5, 10, 7.5);
          scene.add(directionalLight);
          
          // Load the GLB model using GLTFLoader
          const loader = new GLTFLoader();
          loader.load(
            'https://threejs.org/examples/models/gltf/DamagedHelmet/glTF/DamagedHelmet.gltf',
            function(gltf) {
              console.log("Model loaded");
              const model = gltf.scene;
              // Adjust scale and position as needed
              model.scale.set(2, 2, 2);
              model.position.set(0, 0, 0);
              scene.add(model);
              camera.lookAt(model.position);
            },
            function(xhr) {
              console.log((xhr.loaded / xhr.total * 100) + '% loaded');
            },
            function(error) {
              console.error('Error loading model:', error);
            }
          );
          
          // Animation loop
          function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
          }
          animate();
          
          // Update renderer and camera on window resize
          window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
          });
        """, type="module")
    )

serve()
