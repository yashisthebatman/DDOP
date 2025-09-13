// FILE: web/main.js

// --- Global State ---
let viewer;
let DESTINATIONS_LIST = {}; // Will be populated from config
const droneEntities = new Map();

// --- WebSocket Communication ---
const socket = new WebSocket(`ws://${window.location.host}/ws`);

socket.onopen = () => console.log("WebSocket connection established.");
socket.onclose = () => console.log("WebSocket connection closed.");
socket.onerror = (error) => console.error("WebSocket Error:", error);
socket.onmessage = (event) => {
    const state = JSON.parse(event.data);
    updateUI(state);
};

function sendCommand(type, payload = {}) {
    if (socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type, payload }));
    }
}

// --- CesiumJS 3D Map Initialization ---
function initCesium(token) {
    Cesium.Ion.defaultAccessToken = token;
    viewer = new Cesium.Viewer('cesiumContainer', {
        timeline: false,
        animation: false,
        geocoder: false,
        homeButton: false,
        sceneModePicker: false,
        baseLayerPicker: false,
        navigationHelpButton: false,
        infoBox: false,
        selectionIndicator: false,
        fullscreenButton: false,
        requestRenderMode: true,
        maximumRenderTimeChange: Infinity,
    });

    const osmBuildings = Cesium.createOsmBuildings();
    viewer.scene.primitives.add(osmBuildings);

    viewer.camera.flyTo({
        destination: Cesium.Cartesian3.fromDegrees(-73.9854, 40.7484, 3000),
        orientation: {
            heading: Cesium.Math.toRadians(0.0),
            pitch: Cesium.Math.toRadians(-45.0),
        }
    });
}

// --- UI Update Functions ---
function updateUI(state) {
    document.getElementById('simTime').textContent = `Sim Time: ${state.simulation_time.toFixed(1)}s`;
    const drones = Object.values(state.drones);
    document.getElementById('dronesIdle').textContent = drones.filter(d => d.status === 'IDLE').length;
    document.getElementById('dronesActive').textContent = drones.filter(d => d.status !== 'IDLE' && d.status !== 'RECHARGING').length;
    document.getElementById('ordersPending').textContent = Object.keys(state.pending_orders).length;
    document.getElementById('ordersCompleted').textContent = state.completed_orders.length;

    const runPauseButton = document.getElementById('runPauseButton');
    runPauseButton.textContent = state.simulation_running ? '⏸️ Pause' : '▶ Run';
    runPauseButton.className = state.simulation_running ? 'pause' : 'run';

    updateDronesOnMap(state);
    updatePendingOrdersTable(state);

    const logContainer = document.getElementById('logContainer');
    logContainer.innerHTML = state.log.slice(0, 50).join('\n');
}

function updateDronesOnMap(state) {
    if (!viewer) return;
    const activeMissions = state.active_missions || {};
    const drones = Object.values(state.drones);
    const activeDroneIds = new Set(drones.map(d => d.id));

    for (const drone of drones) {
        const position = Cesium.Cartesian3.fromDegrees(drone.pos[0], drone.pos[1], drone.pos[2]);
        let entity = droneEntities.get(drone.id);

        if (!entity) {
            entity = viewer.entities.add({
                id: drone.id,
                position: position,
                model: {
                    uri: 'https://assets.cesium.com/models/CesiumDrone/CesiumDrone.glb',
                    minimumPixelSize: 64,
                },
                label: {
                    text: '', // Will be updated
                    font: '12pt monospace',
                    style: Cesium.LabelStyle.FILL_AND_OUTLINE,
                    outlineWidth: 2,
                    verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
                    pixelOffset: new Cesium.Cartesian2(0, -30),
                    fillColor: Cesium.Color.WHITE,
                },
                path: new Cesium.PolylineGraphics({
                    width: 2,
                    material: Cesium.Color.CYAN,
                    show: false,
                })
            });
            droneEntities.set(drone.id, entity);
        }

        entity.position = position;
        entity.label.text = `${drone.id}\n${drone.status}\n${(drone.battery / 200 * 100).toFixed(0)}%`;

        const mission = activeMissions[drone.mission_id];
        if (mission && mission.path_world_coords && mission.path_world_coords.length > 0) {
            const pathPositions = mission.path_world_coords.flat();
            entity.path.positions = Cesium.Cartesian3.fromDegreesArrayHeights(pathPositions);
            entity.path.show = true;
        } else {
            entity.path.show = false;
        }
    }

    for (const [droneId, entity] of droneEntities.entries()) {
        if (!activeDroneIds.has(droneId)) {
            viewer.entities.remove(entity);
            droneEntities.delete(droneId);
        }
    }
    viewer.scene.requestRender();
}

function updatePendingOrdersTable(state) {
    const tableBody = document.querySelector("#pendingOrdersTable tbody");
    const orders = Object.values(state.pending_orders);
    
    if (orders.length === 0) {
        tableBody.innerHTML = `<tr><td colspan="4" style="text-align:center;">No pending orders</td></tr>`;
        return;
    }
    
    orders.sort((a, b) => (b.high_priority || false) - (a.high_priority || false));

    let html = '';
    for (const order of orders) {
        html += `
            <tr>
                <td>${order.id.split('-')[1]}</td>
                <td>${order.dest_name}</td>
                <td>${order.eta_seconds ? `~${Math.round(order.eta_seconds)}` : 'N/A'}</td>
                <td>${order.high_priority ? 'High' : 'Normal'}</td>
            </tr>
        `;
    }
    tableBody.innerHTML = html;
}

// --- Event Listeners ---
function setupEventListeners() {
    document.getElementById('runPauseButton').addEventListener('click', () => sendCommand('toggle_simulation'));
    document.getElementById('dispatchButton').addEventListener('click', () => sendCommand('dispatch_missions'));
    document.getElementById('resetButton').addEventListener('click', () => {
        if (confirm("Are you sure you want to reset the entire simulation?")) {
            sendCommand('reset_simulation');
        }
    });

    const addOrderForm = document.getElementById('addOrderForm');
    addOrderForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const formData = new FormData(addOrderForm);
        const payload = {
            dest_name: formData.get('dest_name'),
            payload_kg: parseFloat(formData.get('payload_kg')),
            high_priority: formData.get('high_priority') === 'on'
        };
        sendCommand('add_order', payload);
    });
}

// --- Initialization ---
async function main() {
    setupEventListeners();

    try {
        // Step 1: Fetch the secure token
        const tokenResponse = await fetch('/api/token');
        if (!tokenResponse.ok) {
            throw new Error('Could not fetch Cesium token from server.');
        }
        const tokenData = await tokenResponse.json();
        
        // Step 2: Initialize Cesium with the token
        initCesium(tokenData.token);
        
        // Step 3: Fetch destinations for the dropdown
        const destResponse = await fetch('/api/destinations');
        DESTINATIONS_LIST = await destResponse.json();
        
        const select = document.getElementById('destinationSelect');
        select.innerHTML = ''; // Clear any existing options
        for (const destName in DESTINATIONS_LIST) {
            const option = document.createElement('option');
            option.value = destName;
            option.textContent = destName;
            select.appendChild(option);
        }

    } catch (error) {
        console.error("Initialization failed:", error);
        document.getElementById('controlPanel').innerHTML = `<h1>Error</h1><p>Could not initialize application. Check server logs for details, especially for a missing CESIUM_ION_TOKEN in the .env file.</p>`;
    }
}

document.addEventListener('DOMContentLoaded', main);