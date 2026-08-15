import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { gloveRowToBoneRotations } from './bone_mapping.js';

// ── Scene setup (Staging + Solid Drawing, same reasoning as the old Blender version:
// presentation only, never touches predicted data) ──────────────────────────────────
const canvas = document.getElementById('viewport');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setSize(canvas.clientWidth, canvas.clientHeight);
renderer.setPixelRatio(window.devicePixelRatio);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a1a);

const camera = new THREE.PerspectiveCamera(45, canvas.clientWidth / canvas.clientHeight, 0.01, 100);
camera.position.set(0.6, 0.5, 0.6);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0.1, 0);
controls.enableDamping = true;

const keyLight = new THREE.DirectionalLight(0xffffff, 2.5);
keyLight.position.set(1, 2, 1);
scene.add(keyLight);
const fillLight = new THREE.DirectionalLight(0xffffff, 1.0);
fillLight.position.set(-1, 0.5, -1);
scene.add(fillLight);
scene.add(new THREE.AmbientLight(0xffffff, 0.4));

function onResize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h, false);
}
window.addEventListener('resize', onResize);

// ── Load the rigged hand model ───────────────────────────────────────────────────────
let boneByName = {};
const loader = new GLTFLoader();
loader.load('./right_hand.glb', (gltf) => {
  const model = gltf.scene;
  scene.add(model);
  model.traverse((obj) => {
    if (obj.isBone) boneByName[obj.name] = obj;
  });
  console.log(`Model loaded: ${Object.keys(boneByName).length} bones found.`);

  // Frame the camera from the model's ACTUAL measured size, not a hardcoded guess -- the
  // original hardcoded camera position (0.6, 0.5, 0.6) turned out to be inside the mesh
  // (measured bounding box was ~2x2x2.5 units), which renders as solid black due to
  // backface culling. Computing this from the real geometry avoids that class of bug.
  const box = new THREE.Box3().setFromObject(model);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z);
  const distance = (maxDim / 2) / Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * 1.6;   // 1.6x margin

  controls.target.copy(center);
  camera.position.set(center.x + distance * 0.6, center.y + distance * 0.5, center.z + distance * 0.6);
  camera.near = maxDim / 100;
  camera.far = maxDim * 100;
  camera.updateProjectionMatrix();
  controls.update();

  console.log(`Model bounding box size: ${size.x.toFixed(2)}, ${size.y.toFixed(2)}, ${size.z.toFixed(2)} -- camera distance set to ${distance.toFixed(2)}`);
  document.getElementById('status').textContent = 'Model loaded. Ready.';
}, undefined, (err) => {
  console.error('Failed to load right_hand.glb', err);
  document.getElementById('status').textContent = 'ERROR loading model -- see browser console.';
});

function applyPredictionToBones(prediction, handSide) {
  const rotations = gloveRowToBoneRotations(prediction.joint_angles, handSide === 'left' ? 'Left' : 'Right');
  for (const [boneName, { axis, angleDeg }] of Object.entries(rotations)) {
    const bone = boneByName[boneName];
    if (!bone) continue;   // this bone doesn't exist on this rig
    const rad = THREE.MathUtils.degToRad(angleDeg);
    if (axis === 'X') bone.rotation.x = rad;
    else if (axis === 'Y') bone.rotation.y = rad;
    else if (axis === 'Z') bone.rotation.z = rad;
  }
}

// ── HUD ───────────────────────────────────────────────────────────────────────────────
function updateHud(prediction) {
  const angles = prediction.joint_angles;
  const labelForChannel = { Thumb: 1, Index: 4, Middle: 8, Ring: 12, Little: 16 };
  let lines = [`Frame ${prediction.frame_index}  t=${(prediction.elapsed_ms / 1000).toFixed(1)}s`];
  for (const [label, ch] of Object.entries(labelForChannel)) {
    lines.push(`${label} MCP: ${angles[ch].toFixed(0)} deg`);
  }
  if (prediction.forearm_emg) {
    const avg = prediction.forearm_emg.reduce((a, b) => a + b, 0) / prediction.forearm_emg.length;
    lines.push(`Forearm EMG (real): ${avg.toFixed(2)}`);
  }
  if (prediction.bicep_emg_simulated) {
    const avg = prediction.bicep_emg_simulated.reduce((a, b) => a + b, 0) / prediction.bicep_emg_simulated.length;
    lines.push(`Bicep EMG (SIMULATED, no real channel in dataset): ${avg.toFixed(2)}`);
  }
  document.getElementById('hud').textContent = lines.join('\n');
}

// ── Session control ───────────────────────────────────────────────────────────────────
let ws = null;

async function loadTasks() {
  const res = await fetch('/api/tasks');
  const { tasks } = await res.json();
  const select = document.getElementById('task');
  select.innerHTML = '';
  for (const t of tasks) {
    const opt = document.createElement('option');
    opt.value = t; opt.textContent = t;
    select.appendChild(opt);
  }
}

function startSession() {
  if (ws) ws.close();
  const config = {
    task: document.getElementById('task').value,
    age: document.getElementById('age').value,
    hand: document.getElementById('hand').value,
    amputation: document.getElementById('amputation').value,
  };
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws/session`);
  ws.onopen = () => {
    ws.send(JSON.stringify(config));
    document.getElementById('status').textContent = `Running: ${config.task}`;
  };
  ws.onmessage = (event) => {
    const prediction = JSON.parse(event.data);
    if (prediction.error) {
      document.getElementById('status').textContent = `ERROR: ${prediction.error}`;
      return;
    }
    applyPredictionToBones(prediction, config.hand);
    updateHud(prediction);
  };
  ws.onclose = () => { document.getElementById('status').textContent = 'Stopped.'; };
  ws.onerror = (e) => { console.error('WebSocket error', e); };
}

function stopSession() {
  if (ws) { ws.close(); ws = null; }
  document.getElementById('status').textContent = 'Stopped.';
}

document.getElementById('start-btn').addEventListener('click', startSession);
document.getElementById('stop-btn').addEventListener('click', stopSession);
loadTasks();

// ── Render loop ───────────────────────────────────────────────────────────────────────
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();
