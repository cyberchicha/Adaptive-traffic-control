import os
import sys
import traci
import time

def setup_sumo():
    """
    Checks for the presence of SUMO and returns the command to launch the GUI
    """
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("Please declare the 'SUMO_HOME' environment variable")
    
    # Path to our configuration file
    sumocfg_path = os.path.join(os.path.dirname(__file__), "..", "data", "sumo", "cross.sumocfg")
    
    # Using sumo-gui for visualization
    return ["sumo-gui", "-c", sumocfg_path, "--start"]

def run_simulation():
    """
    Runs the simulation and controls it through TraCI
    """
    print("[INFO] Connecting to SUMO...")
    sumo_cmd = setup_sumo()
    traci.start(sumo_cmd)
    
    print("[INFO] Simulation started!")
    
    # Getting the ID of our traffic light (usually the same as the ID of the 'center' node)
    tls_ids = traci.trafficlight.getIDList()
    main_tls_id = tls_ids[0] if tls_ids else None
    
    if main_tls_id:
        print(f"[INFO] Found traffic light: {main_tls_id}")
    
    step = 0
    while step < 1000:
        traci.simulationStep()
        
        # Every 100 steps, reading the current traffic light phase
        if step % 100 == 0 and main_tls_id:
            current_phase = traci.trafficlight.getPhase(main_tls_id)
            current_state = traci.trafficlight.getRedYellowGreenState(main_tls_id)
            print(f"Step {step} | Phase: {current_phase} | State: {current_state}")
            
        step += 1
        time.sleep(0.05)

    print("[INFO] Simulation completed.")
    traci.close()

if __name__ == "__main__":
    run_simulation()