import jax
import sys

def verify_jax():
    print(f"Python version: {sys.version}")
    print(f"JAX version: {jax.__version__}")
    
    devices = jax.devices()
    print(f"Available devices: {devices}")
    
    gpu_found = any(d.device_kind == 'gpu' for d in devices)
    if gpu_found:
        print("SUCCESS: GPU detected!")
    else:
        print("INFO: Only CPU detected (expected if not in GPU-enabled environment/container).")

if __name__ == "__main__":
    try:
        verify_jax()
    except Exception as e:
        print(f"ERROR during JAX initialization: {e}")
        sys.exit(1)
