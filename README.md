# Q-DOP: Quantum Drone Operations Platform

**An Enterprise-Grade Digital Twin for Autonomous Urban Air Mobility**

---

## The Future of Logistics is Not on the Ground. It's in the Sky.

**Q-DOP (Quantum Drone Operations Platform)** is not just a simulation; it's a digital twin for the future of last-mile delivery. This platform provides a comprehensive, end-to-end solution for managing a fleet of autonomous delivery drones navigating the complex, three-dimensional landscape of a dense city. From intelligent, AI-powered dispatching to sophisticated, multi-layered pathfinding and dynamic, real-world event handling, Q-DOP is engineered to tackle the most challenging aspects of urban drone logistics.

Our mission is to provide the critical infrastructure for planning, testing, and deploying drone fleets safely and efficiently, paving the way for the next generation of automated commerce.

---

## Core Features & Technological Edge

Q-DOP is built on a foundation of cutting-edge algorithms and a robust, real-time architecture. Here's what sets it apart:

### **Intelligent Fleet & Mission Control**

-   **Real-Time 3D Digital Twin:** A dynamic, interactive 3D environment built with **Plotly.js** provides unparalleled situational awareness, visualizing every drone, flight path, and obstacle in real-time.
-   **Multi-Depot Vehicle Routing (MDVRP):** Our "Dispatch Batch" system, powered by **Google OR-Tools**, transcends simple one-to-one assignments. It solves the complex Vehicle Routing Problem to create globally optimized, multi-stop delivery tours, maximizing fleet efficiency and minimizing delivery times.
-   **Automated Fleet Rebalancing:** A background **Demand Analyzer** perpetually monitors fleet distribution. It autonomously identifies logistical imbalances—too many drones at one hub, not enough at another—and dispatches drones on rebalancing missions to preemptively meet demand.
-   **ML-Powered Performance Prediction:** A sophisticated **Random Forest Regressor** model predicts flight time and energy consumption with remarkable accuracy, accounting for payload, distance, altitude changes, and dynamic weather. This intelligence is crucial for reliable dispatching and contingency planning.

### **Advanced Path Planning & Deconfliction**

-   **Hybrid Hierarchical Path Planning Engine:** We employ a two-tiered planning strategy for the perfect blend of speed and precision:
    1.  **Strategic Layer (A*):** A high-speed A* search on a cost-based coarse 3D grid maps out a safe, high-level "freeway in the sky."
    2.  **Tactical Layer (Anytime RRT*):** Within this strategic corridor, an Anytime Rapidly-exploring Random Tree Star (RRT*) algorithm generates a smooth, kinematically feasible flight path that elegantly weaves around obstacles.
-   **Guaranteed Collision Avoidance (Conflict-Based Search):** For multi-drone scenarios, Q-DOP uses a high-level **Conflict-Based Search (CBS)** solver. It systematically identifies and resolves potential path conflicts *before* drones are dispatched, guaranteeing collision-free routes by strategically inserting wait times or minor detours.

### **Resilience & Dynamic Simulation**

-   **Intelligent Contingency Management:** Drones are not just planners; they are problem-solvers.
    -   **Critical Battery Failsafe:** Each drone continuously calculates its "point of no return." If its battery level drops below the threshold needed to reach the *nearest* safe hub, it automatically aborts its current mission and initiates an emergency landing.
    -   **Dynamic Obstacle & NFZ Reaction:** The simulation can inject unexpected **No-Fly Zones (NFZs)**. Drones whose paths are invalidated by these new obstacles instantly trigger their contingency planners to find a safe way out.
-   **Dynamic Weather System:** A **Perlin noise-based** weather simulation creates realistic, evolving wind patterns that directly impact drone battery consumption and flight speed, forcing the entire system to adapt to real-world variability.
-   **Continuous Learning & Model Improvement:** The platform is built for the future. A dedicated retraining pipeline allows new flight data from completed missions to be fed back into the ML model, constantly improving its prediction accuracy over time.

---

## Technical Architecture

-   **Frontend:** A lightweight, responsive single-page application built with vanilla **HTML, CSS, and JavaScript**. It maintains a persistent **WebSocket** connection for real-time, low-latency updates from the server.
-   **Backend:** A high-performance asynchronous server built with **Python** and the **FastAPI** framework, running on **Uvico.rn**. It manages the core simulation loop, handles all API requests, and broadcasts state changes to all connected clients.
-   **Core Engine:**
    -   **Simulation Loop:** A time-stepped engine that updates drone physics, checks for events, and advances the state of the world.
    -   **Planning Stack:** A sophisticated combination of A*, RRT*, and Conflict-Based Search for multi-layered, deconflicted pathfinding.
    -   **State Management:** A robust system using **TinyDB** to persist the complete simulation state, allowing for pauses and resumes.
    -   **Optimization & ML:** Integrates **Google OR-Tools** for VRP and **Scikit-learn** for performance prediction.

---

## Getting Started

### Prerequisites
-   Python 3.10 or higher
-   `pip` package manager
-   `venv` module for virtual environments

### Installation & Launch

1.  **Clone the Repository**
    ```sh
    git clone https://github.com/yashisthebatman/Q-DOP.git
    cd Q-DOP
    ```

2.  **Set Up Virtual Environment**
    ```sh
    # Windows
    python -m venv venv
    .\venv\Scripts\Activate.ps1

    # macOS / Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```sh
    pip install -r requirements.txt
    ```

4.  **Launch the Server**
    ```sh
    uvicorn server:app --reload
    ```

5.  **Access the Platform**
    Open your web browser and navigate to **`http://127.0.0.1:8000`**.

---

## How to Use the Platform

1.  **Initialize:** The platform will load with an idle fleet and a 3D view of the city.
2.  **Start Simulation:** Click the green **`▶ Run Sim`** button to begin advancing simulation time.
3.  **Create Orders:** In the "Add New Order" panel, select a dispatch hub, a destination, and a payload weight. Click **`Add Order`**.
4.  **Dispatch:**
    -   Orders are dispatched **automatically** to the best-suited idle drone at the selected hub.
    -   Alternatively, add multiple orders and click **`Dispatch Batch`** to see the powerful VRP solver optimize multi-stop tours for the entire fleet.
5.  **Observe & Monitor:** Watch as drones transition from `IDLE` to `PLANNING` to `EN ROUTE`. Track their progress in the 3D view and monitor their status, battery, and logs in the side panels.
6.  **Simulate Events:** The system will randomly introduce dynamic No-Fly Zones. Watch how the affected drones react, aborting their missions and returning to the nearest hub.
7.  **Reset:** Click the **`⚠️ Reset`** button to clear all progress and return the simulation to its initial state.

---

## Running the Test Suite

Our platform is backed by a comprehensive test suite to ensure the reliability of our complex planning and simulation logic.

To run all tests, execute the following command from the project root:
```sh
pytest
```

---
