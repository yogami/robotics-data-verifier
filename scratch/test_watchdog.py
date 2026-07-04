import subprocess
import time
import os
import sys
import signal

def run_watchdog_test_case(name, loss_sequence, expect_kill):
    print(f"\nRunning Watchdog Test Case: {name}")
    log_path = f"scratch/test_watchdog_{name.lower().replace(' ', '_')}.log"
    if os.path.exists(log_path):
        os.remove(log_path)
        
    # Write initial normal loss log
    with open(log_path, "w") as f:
        f.write("Epoch   1/100 | loss=0.543210 | 1.0s\n")
        f.flush()
        
    # Start target process
    dummy = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    dummy_pid = dummy.pid
    
    # Launch watchdog
    watchdog = subprocess.Popen([
        sys.executable, 
        "scratch/runpod_watchdog.py", 
        log_path, 
        str(dummy_pid)
    ])
    
    try:
        time.sleep(1) # Let watchdog start
        
        # Append the loss sequence
        epoch = 2
        for loss in loss_sequence:
            print(f"  Writing: Epoch {epoch:3d}/100 | loss={loss} | {epoch:.1f}s")
            with open(log_path, "a") as f:
                f.write(f"Epoch {epoch:3d}/100 | loss={loss} | {epoch:.1f}s\n")
                f.flush()
            epoch += 1
            time.sleep(0.5)
            
            # Check if process died early
            dummy.poll()
            if dummy.returncode is not None:
                break
                
        # Wait a moment for watchdog to catch up
        time.sleep(2)
        
        dummy.poll()
        killed = dummy.returncode is not None
        
        if expect_kill:
            if killed:
                print(f"  ✅ PASSED: Process was successfully killed on sequence: {loss_sequence}")
            else:
                print(f"  ❌ FAILED: Process was NOT killed, but expected kill on sequence: {loss_sequence}")
                dummy.terminate()
                dummy.wait()
                sys.exit(1)
        else:
            if not killed:
                print(f"  ✅ PASSED: Process survived as expected on sequence: {loss_sequence}")
                dummy.terminate()
                dummy.wait()
            else:
                print(f"  ❌ FAILED: Process was killed, but expected survival on sequence: {loss_sequence}")
                sys.exit(1)
                
    finally:
        # Clean up watchdog
        watchdog.poll()
        if watchdog.returncode is None:
            watchdog.terminate()
            watchdog.wait()
        # Clean up log file
        if os.path.exists(log_path):
            os.remove(log_path)

def main():
    print("==============================================================")
    print("      Watchdog Test Suite: Verifying Budget Protection")
    print("==============================================================")
    
    # Case 1: 3 consecutive NaN losses must trigger kill
    run_watchdog_test_case(
        name="Consecutive NaN Kill",
        loss_sequence=["nan", "nan", "nan"],
        expect_kill=True
    )
    
    # Case 2: 3 consecutive signed -inf losses must trigger kill
    run_watchdog_test_case(
        name="Consecutive Signed Inf Kill",
        loss_sequence=["-inf", "-inf", "-inf"],
        expect_kill=True
    )
    
    # Case 3: Alternating NaN and normal loss must NOT trigger kill (reset logic check)
    run_watchdog_test_case(
        name="Alternating NaN Survival",
        loss_sequence=["nan", "0.500000", "nan", "0.450000", "nan"],
        expect_kill=False
    )
    
    print("\nWatchdog Test Suite completed successfully!")

if __name__ == "__main__":
    main()
