// FILE: web/main.js

let plotInitialized = false;

const socket = new WebSocket(`ws://${window.location.host}/ws`);

socket.onopen = () => console.log("WebSocket connection established.");
socket.onclose = () => console.log("WebSocket connection closed.");
socket.onerror = (error) => console.error("WebSocket Error:", error);
socket.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (!plotInitialized) {
        initPlot(message.plotly_data);
        plotInitialized = true;
    }
    updateUI(message);
    updatePlot(message.plotly_data);
};

function sendCommand(type, payload = {}) {
    if (socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type, payload }));
    }
}

function initPlot(data) {
    const layout = {
        title: 'Fleet Simulation', showlegend: false, autosize: true,
        paper_bgcolor: 'black', plot_bgcolor: 'black', // FIX: Black background
        font: { color: '#e0e0e0' }, margin: { l: 0, r: 0, b: 0, t: 40 },
        scene: {
            xaxis: { title: 'X (meters)', color: '#777', backgroundcolor: 'black', gridcolor: '#444', zerolinecolor: '#444' },
            yaxis: { title: 'Y (meters)', color: '#777', backgroundcolor: 'black', gridcolor: '#444', zerolinecolor: '#444' },
            zaxis: { title: 'Altitude (m)', color: '#777', backgroundcolor: 'black', gridcolor: '#444', zerolinecolor: '#444' },
            camera: { eye: { x: -1.5, y: -1.5, z: 1.0 } }
        }
    };
    Plotly.newPlot('plotContainer', data, layout, { responsive: true, displaylogo: false });
}

function updatePlot(data) {
    Plotly.react('plotContainer', data);
}

function updateUI(message) {
    const state = message.simulation_state;
    document.getElementById('simTime').textContent = `Sim Time: ${state.simulation_time.toFixed(1)}s`;
    
    const drones = message.drone_list || [];
    document.getElementById('dronesIdle').textContent = drones.filter(d => d.status === 'IDLE').length;
    document.getElementById('dronesActive').textContent = drones.filter(d => ['EN ROUTE', 'AVOIDING', 'EMERGENCY_RETURN', 'PERFORMING_DELIVERY'].includes(d.status)).length;
    document.getElementById('ordersPending').textContent = (message.pending_orders_list || []).length;
    document.getElementById('ordersCompleted').textContent = state.completed_orders.length;

    const runPauseButton = document.getElementById('runPauseButton');
    runPauseButton.textContent = state.simulation_running ? '⏸️ Pause' : '▶ Run';
    runPauseButton.className = state.simulation_running ? 'pause' : 'run';

    updateDroneStatusList(drones);
    updatePendingOrdersTable(message.pending_orders_list || []);
    updateMissionLog(message.mission_log || []);
}

function updateDroneStatusList(drones) {
    const container = document.getElementById('droneStatusContainer');
    drones.sort((a, b) => parseInt(a.id.split(' ')[1]) - parseInt(b.id.split(' ')[1]));
    let html = '';
    for (const drone of drones) {
        const batteryPercent = (drone.battery / 200) * 100;
        let batteryClass = '';
        if (batteryPercent < 40) batteryClass = 'low';
        if (batteryPercent < 20) batteryClass = 'critical';
        html += `
            <div class="drone-item">
                <span class="drone-name">${drone.id}</span>
                <span class="drone-status">${drone.status}</span>
                <div class="battery-bar" title="${batteryPercent.toFixed(1)}%">
                    <div class="battery-fill ${batteryClass}" style="width: ${batteryPercent}%;"></div>
                </div>
            </div>
        `;
    }
    container.innerHTML = html;
}

function updatePendingOrdersTable(orders) {
    const tableBody = document.querySelector("#pendingOrdersTable tbody");
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
                <td>${order.payload_kg.toFixed(1)}kg</td>
                <td>${order.high_priority ? 'High' : 'Normal'}</td>
            </tr>
        `;
    }
    tableBody.innerHTML = html;
}

function updateMissionLog(log) {
    const container = document.getElementById('logContainer');
    // Show most recent 50 entries, with newest at the top
    const reversedLog = [...log].reverse().slice(0, 50);
    container.innerHTML = reversedLog.map(entry => {
        const time = new Date(entry.completion_timestamp * 1000).toISOString().substr(11, 8);
        return `<div>[${time}] ${entry.drone_id} - Mission ${entry.mission_id.split('-')[1]} - ${entry.outcome}</div>`;
    }).join('');
}

function setupEventListeners() {
    document.getElementById('runPauseButton').addEventListener('click', () => sendCommand('toggle_simulation'));
    document.getElementById('dispatchButton').addEventListener('click', () => sendCommand('dispatch_missions'));
    document.getElementById('resetButton').addEventListener('click', () => {
        if (confirm("Are you sure? This will reset all simulation progress.")) {
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
        addOrderForm.reset();
    });
}

async function main() {
    setupEventListeners();
    try {
        // FIX: Removed the fetch for the non-existent token API
        const destResponse = await fetch('/api/destinations');
        const DESTINATIONS_LIST = await destResponse.json();
        const select = document.getElementById('destinationSelect');
        select.innerHTML = '';
        for (const destName in DESTINATIONS_LIST) {
            const option = document.createElement('option');
            option.value = destName;
            option.textContent = destName;
            select.appendChild(option);
        }
    } catch (error) {
        console.error("Initialization failed:", error);
        document.getElementById('leftPanel').innerHTML = `<h1>Error</h1><p>Could not initialize application. Check server logs.</p>`;
    }
}

document.addEventListener('DOMContentLoaded', main);