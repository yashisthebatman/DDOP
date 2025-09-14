// FILE: web/main.js

let plotInitialized = false;
const socket = new WebSocket(`ws://${window.location.host}/ws`);

socket.onopen = () => console.log("WebSocket connection established.");
socket.onclose = () => console.log("WebSocket connection closed.");
socket.onerror = (error) => console.error("WebSocket Error:", error);

socket.onmessage = (event) => {
    const message = JSON.parse(event.data);
    
    if (message.type === 'order_out_of_range') {
        showNotification(`Order ${message.payload.order_id} is out of range. No available drones have enough battery.`, 'error');
        return;
    }

    if (message.simulation_state) {
        if (!plotInitialized) {
            // Plotly is now data-driven and initialized via the main state update
            plotInitialized = true;
        }
        updateUI(message);
    }
};

function sendCommand(type, payload = {}) {
    if (socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type, payload }));
    }
}

function showNotification(text, type = 'info') {
    const notificationArea = document.getElementById('notificationArea');
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = text;
    notificationArea.appendChild(notification);
    setTimeout(() => {
        notification.classList.add('fade-out');
        setTimeout(() => notification.remove(), 500);
    }, 5000);
}

function updateUI(message) {
    const state = message.simulation_state;
    document.getElementById('simTime').textContent = `Sim Time: ${state.simulation_time.toFixed(1)}s`;
    
    const drones = message.drone_list || [];
    document.getElementById('dronesIdle').textContent = drones.filter(d => d.status === 'IDLE').length;
    document.getElementById('dronesActive').textContent = drones.filter(d => !['IDLE', 'RECHARGING'].includes(d.status)).length;
    document.getElementById('ordersPending').textContent = Object.keys(state.pending_orders).length;
    document.getElementById('ordersCompleted').textContent = state.completed_orders.length;

    const runPauseButton = document.getElementById('runPauseButton');
    runPauseButton.textContent = state.simulation_running ? '⏸️ Pause' : '▶ Run';
    runPauseButton.className = state.simulation_running ? 'pause' : 'run';

    updateDroneStatusList(drones);
    updatePendingOrdersTable(Object.values(state.pending_orders || {}));
    updateMissionLog(state.completed_missions_log || []);
}

let HUBS_BY_ID = {}; // Global store for hub data

function updateDroneStatusList(drones) {
    const container = document.getElementById('droneStatusContainer');
    drones.sort((a, b) => a.home_hub.localeCompare(b.home_hub) || parseInt(a.id.split(' ')[1]) - parseInt(b.id.split(' ')[1]));
    
    let html = '';
    let currentHub = null;
    for (const drone of drones) {
        if (drone.home_hub !== currentHub) {
            currentHub = drone.home_hub;
            html += `<div class="hub-header">${HUBS_BY_ID[currentHub]?.name || currentHub}</div>`;
        }
        const batteryPercent = (drone.battery / 200) * 100;
        const statusClass = drone.status.toLowerCase().replace(/_/g, '-');
        
        html += `
            <div class="drone-item" title="ID: ${drone.id}\nBattery: ${batteryPercent.toFixed(1)}%\nHealth: ${drone.battery_health}%\nCycles: ${drone.charge_cycles}">
                <span class="drone-icon ${statusClass}"></span>
                <span class="drone-name">${drone.id}</span>
                <span class="drone-status">${drone.status}</span>
                <div class="battery-bar">
                    <div class="battery-fill" style="width: ${batteryPercent}%;"></div>
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
    const reversedLog = [...log].reverse().slice(0, 50);
    container.innerHTML = reversedLog.map(entry => {
        const time = new Date(entry.completion_timestamp * 1000).toISOString().substr(11, 8);
        const outcomeClass = entry.outcome.startsWith('Failed') ? 'log-fail' : 'log-success';
        return `<div class="${outcomeClass}">[${time}] ${entry.drone_id} - Mission ${entry.mission_id.split('-')[1]} - ${entry.outcome}</div>`;
    }).join('');
}


function setupEventListeners() {
    document.getElementById('runPauseButton').addEventListener('click', () => sendCommand('toggle_simulation'));
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
            hub_id: formData.get('hub_id'),
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
        const [hubsResponse, destResponse] = await Promise.all([ fetch('/api/hubs'), fetch('/api/destinations') ]);
        const HUBS_LIST = await hubsResponse.json();
        const DESTINATIONS_LIST = await destResponse.json();

        const hubSelect = document.getElementById('hubSelect');
        HUBS_LIST.forEach(hub => {
            HUBS_BY_ID[hub.id] = hub;
            const option = document.createElement('option');
            option.value = hub.id;
            option.textContent = hub.name;
            hubSelect.appendChild(option);
        });

        const destSelect = document.getElementById('destinationSelect');
        for (const destName in DESTINATIONS_LIST) {
            const option = document.createElement('option');
            option.value = destName;
            option.textContent = destName;
            destSelect.appendChild(option);
        }
    } catch (error) {
        console.error("Initialization failed:", error);
    }
}

document.addEventListener('DOMContentLoaded', main);