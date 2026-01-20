import subprocess
import argparse
import sys
from pathlib import Path

# ------ CONFIGURATION ------
RAW_DATA_ROOT = Path("/media/yiting/NewVolume/Data/Videos")
ANALYSIS_ROOT = Path("/media/yiting/NewVolume/Analysis")
LP_ROOT = Path("/home/yiting/Documents/GitHub/lightning-pose")
LP_CONFIG_NAME = "config.yaml"
AP_CONFIG_NAME = "config.toml"

# PATHS TO ENVIRONMENTS
ENV_PATHS = {
    "preprocessing": "/home/yiting/anaconda3/envs/hand-trk/bin/python",
    "training":      "/home/yiting/anaconda3/envs/litpose/bin/python",
    "inference":     "/home/yiting/anaconda3/envs/litpose/bin/python",      
    "triangulation": "/home/yiting/anaconda3/envs/anipose/bin/python", 
    "kinematics":    "/home/yiting/anaconda3/envs/hand-trk/bin/python"
}

PACKAGE_ROOT = Path(__file__).resolve().parent

SCRIPT_PATHS = {
    "preprocessing": PACKAGE_ROOT / "preprocessing" / "litpose_formatter.py",
    "training":      PACKAGE_ROOT / "modeling" / "lp_runner.py",
    "inference":     LP_ROOT / "scripts" / "predict_new_vids.py",
    "triangulation": PACKAGE_ROOT / "triangulation" / "anipose_runner.py",
    "kinematics":    PACKAGE_ROOT / "kinematics" / "feature_extractor.py"
}

def run_step(step_name, extra_args, session_name=None):
    """Generic function to trigger a subprocess in a specific env"""
    print(f"🚀 Starting Step: {step_name.upper()}...")
    
    python_executable = ENV_PATHS[step_name]
    script_path = SCRIPT_PATHS[step_name]
    
    if not Path(python_executable).exists():
        raise FileNotFoundError(f"Could not find python env for {step_name} at: {python_executable}")
    
    cmd = [python_executable, str(script_path)]

    # --- STEP SPECIFIC ARGUMENT HANDLING ---
    
    if step_name == "preprocessing":
        # Pass the global paths + the session args
        cmd.extend(["--data_dir", str(RAW_DATA_ROOT)])
        cmd.extend(["--analysis_dir", str(ANALYSIS_ROOT)])
        # Pass through the user's --session argument
        cmd.extend(extra_args)

    elif step_name == "inference":
        if not session_name:
            raise ValueError("Session name is required for inference path construction")
            
        # Construct the specific path to the config for this session
        lp_config_dir = ANALYSIS_ROOT / session_name / "litpose"
        
        # NOTE: Lightning Pose 'predict_new_vids.py' usually uses Hydra. 
        # Check if it expects --config-path or absolute paths. 
        # Here we assume standard Hydra syntax:
        cmd.extend(["--config-path", str(lp_config_dir)])
        cmd.extend(["--config-name", LP_CONFIG_NAME])
        
        # Add any other extra args (like which video to predict)
        cmd.extend(extra_args)

    elif step_name == "triangulation":    
        cmd.extend(["--analysis_dir", str(ANALYSIS_ROOT)])
        cmd.extend(extra_args)
        
    elif step_name == "kinematics":
        cmd.extend(["--analysis_dir", str(ANALYSIS_ROOT)])
        cmd.extend(extra_args)
    else:
        # For other steps, just pass whatever the user typed
        cmd.extend(extra_args)
    
    try:
        # check_call waits for the process to finish
        subprocess.check_call(cmd)
        print(f"✅ Finished Step: {step_name.upper()}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in step {step_name}: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Hand Tracking Pipeline Dispatcher")
    
    # 2. Added 'inference' to choices
    parser.add_argument("--stage", 
                        choices=["preprocess", "train", "inference", "triangulate", "kinematics", "all"], 
                        required=True)
    
    # 3. Explicitly parse session here so we can use it for path building
    parser.add_argument("--session", type=str, required=False, help="Session name (e.g., 2025-11-20)")

    # parse_known_args allows other flags to pass through to the subprocess
    args, unknown_args = parser.parse_known_args()

    # Re-construct the session arg for the subprocesses that need it in 'unknown_args' format
    session_arg_list = ["--session", args.session] if args.session else []
    
    # Combine unknown args with the session arg for the subprocesses
    pass_through_args = unknown_args + session_arg_list

    # --- DISPATCHER LOGIC ---

    if args.stage == "preprocess" or args.stage == "all":
        run_step("preprocessing", pass_through_args)
        
    if args.stage == "train" or args.stage == "all":
        # logic for training
        pass 

    if args.stage == "inference" or args.stage == "all":
        # We pass args.session explicitly for path construction
        run_step("inference", unknown_args, session_name=args.session)

    if args.stage == "triangulate" or args.stage == "all":
        run_step("triangulation", pass_through_args)

    if args.stage == "kinematics" or args.stage == "all":
        run_step("kinematics", pass_through_args)

if __name__ == "__main__":
    main()