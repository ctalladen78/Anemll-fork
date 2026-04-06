import coremltools as ct
import os
import sys

def profile_ane_residency(model_path):
    print(f"--- ANE Residency Profile: {os.path.basename(model_path)} ---")
    
    # Load model with ALL units to see fallback
    model = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.ALL)
    
    # Get the MIL program
    # Note: Modern coremltools models expose the MIL program during the compilation phase
    # or through the 'get_spec()' and then inspecting the mil_program if it was saved.
    
    # More direct: Check if it loads with NE_ONLY (using CPU_AND_NE as proxy for failure detection)
    try:
        model_ne = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        print("SUCCESS: Model loads with ANE preference (CPU_AND_NE).")
    except Exception as e:
        print(f"FAILURE: Model REJECTED by ANE. Fallback required. Error: {e}")

    # Use xcrun coremlcompiler to get a detailed report if possible
    print("\nAttempting to generate CoreML Compiler Optimization Report...")
    os.system(f"xcrun coremlcompiler compile {model_path} . --report")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python profile_residency.py <path_to_mlpackage>")
    else:
        profile_ane_residency(sys.argv[1])
